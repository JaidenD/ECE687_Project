import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from robomaster_msgs.action import GripperControl, MoveArm
from std_msgs.msg import ColorRGBA
from rclpy.callback_groups import ReentrantCallbackGroup

from robo_hockey_controller.helpers import wrap, yaw_from_pose

ROBOT1_ID = 1   # passing robot id
ROBOT2_ID = 2   # scoring robot id
ARM_RAISE_HEIGHT = 1.0 # TODO: tune in lab
##### simulator values ######################
SIM_STICK_X = 2.0
SIM_STICK_Y = 0.0
SIM_STICK_THETA = np.pi
SIM_PUCK_X = -0.5
SIM_PUCK_Y = -0.5
SIM_GOAL_X = 0.0
SIM_GOAL_Y = -2.0
#############################################

STICK_DIST = 0.5 # distance from the location of the passing robot to the stick tip

Kp = 0.5
Kd = 0.05
K_HEADING = 0.8

tol = 0.01
heading_tol = 0.1
ACTION_SERVER_TIMEOUT = 0.2
POSE_FILTER_ALPHA = 0.2 # higher = trust new mocap more, 1.0 = no filtering
DEBUG_PRINT_PERIOD = 0.1

class Controller(Node):
    def __init__(self, robot_id, passOrScore, sim=False):
        node_name = str(passOrScore)+'_id_'+str(robot_id)
        super().__init__(node_name)

        self.sim = sim
        self.passOrScore = passOrScore

        self.robot_id = robot_id
        self.robot_pose = None
        self.stick_pose = None
        self.puck_pose = None
        self.goal_pose = None
        self.state = 'open_gripper'
        self.state_started = self.get_clock().now()
        self.just_entered_state = True

        self.previous_PD_time = None
        self.previous_p_errx = 0.0
        self.previous_p_erry = 0.0
        self.next_debug_print = 0.0
        

        pose_topic = '/vrpn_mocap/dji_robot_'+str(self.robot_id)+'/pose'
        cmd_vel_topic = '/robot'+str(self.robot_id)+'/cmd_vel'
        move_arm_action = '/robot'+str(self.robot_id)+'/move_arm'
        gripper_action = '/robot'+str(self.robot_id)+'/gripper'

        self.CONTROL_FREQUENCY = 100
        self.waiting_for_pose_logging = True


        # subs
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        self.sub_robot_pose = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.save_robot_pose,
            qos
        )
        self.sub_robot_pose

        # stick location and orientation
        self.stick_x = None
        self.sitck_y = None
        self.stick_theta = None

        # puck location
        self.puck_x = None
        self.puck_y = None

        # goal location
        self.goal_x = None
        self.goal_y = None

        if not self.sim:
            stick_topic = '/vrpn_mocap/stick/pose'
            puck_topic = '/vrpn_mocap/puck/pose'
            goal_topic = '/vrpn_mocap/goal/pose'

            self.sub_stick = self.create_subscription(
                PoseStamped,
                stick_topic,
                self.save_stick_pose,
                qos
            )
            self.sub_puck = self.create_subscription(
                PoseStamped,
                puck_topic,
                self.save_puck_pose,
                qos
            )
            self.sub_goal = self.create_subscription(
                PoseStamped,
                goal_topic,
                self.save_goal_pose,
                qos
            )
            
        # pubs
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, qos)

        # action clients
        self._action_group = ReentrantCallbackGroup()

        if not self.sim:
            self.gripper_action_client = ActionClient(
                self,
                GripperControl,
                gripper_action,
                callback_group=self._action_group
            )
            self.gripper_action_client.wait_for_server()

            self.arm_action_client = ActionClient(
                self,
                MoveArm,
                move_arm_action,
                callback_group=self._action_group
            )

        # timer
        self.timer = self.create_timer(1. /self.CONTROL_FREQUENCY, self.stick_nav_and_pickup)


    def save_robot_pose(self, msg:PoseStamped):
        self.robot_pose = self.low_pass_pose(self.robot_pose, msg)

    def save_stick_pose(self, msg:PoseStamped):
        self.stick_pose = self.low_pass_pose(self.stick_pose, msg)

    def save_puck_pose(self, msg:PoseStamped):
        self.puck_pose = self.low_pass_pose(self.puck_pose, msg)
        
    def save_goal_pose(self, msg:PoseStamped):
        self.goal_pose = self.low_pass_pose(self.goal_pose, msg)

    def low_pass_pose(self, old_pose, new_pose):
        # simple mocap low pass filter
        if old_pose is None:
            return new_pose

        alpha = POSE_FILTER_ALPHA
        filtered_pose = PoseStamped()
        filtered_pose.header = new_pose.header

        filtered_pose.pose.position.x = (1.0 - alpha) * old_pose.pose.position.x + alpha * new_pose.pose.position.x
        filtered_pose.pose.position.y = (1.0 - alpha) * old_pose.pose.position.y + alpha * new_pose.pose.position.y
        filtered_pose.pose.position.z = (1.0 - alpha) * old_pose.pose.position.z + alpha * new_pose.pose.position.z

        old_theta = yaw_from_pose(old_pose)
        new_theta = yaw_from_pose(new_pose)
        theta = wrap(old_theta + alpha * wrap(new_theta - old_theta))

        filtered_pose.pose.orientation.z = np.sin(theta / 2.0)
        filtered_pose.pose.orientation.w = np.cos(theta / 2.0)

        return filtered_pose
    
    def go_to_state(self, new_state):
        self.state = new_state
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.next_debug_print = 0.0
        self.previous_PD_time = None
        self.previous_p_errx = 0.0
        self.previous_p_erry = 0.0
        self.get_logger().info(f'State: {new_state}')

    def seconds_in_state(self):
        dt = self.get_clock().now() - self.state_started
        return dt.nanoseconds / 1e9

    def unicycle_linear_controller(self, x, y, theta, px_des, py_des):
        # Take virtual outputs z1 = x + d cos(theta), z2 = y + d sin(theta), d = STICK_DIST
        z1 = x + STICK_DIST * np.cos(theta) 
        z2 = y + STICK_DIST * np.sin(theta)

        p_errx = px_des - z1
        p_erry = py_des - z2
        p_err = np.sqrt(p_errx**2 + p_erry**2)

        now = self.get_clock().now().nanoseconds / 1e9
        if self.previous_PD_time is None:
            dt = 0
        else:
            dt = max(now - self.previous_PD_time, 1e-6)

        if dt > 0.0:
            d_err_x = (p_errx - self.previous_p_errx) / dt
            d_err_y = (p_erry - self.previous_p_erry) / dt
        else:
            d_err_x = 0.0
            d_err_y = 0.0
        
        self.previous_p_errx = p_errx
        self.previous_p_erry = p_erry
        self.previous_PD_time = now

        vx = Kp * p_errx + Kd * d_err_x
        vy = Kp * p_erry + Kd * d_err_y

        v = np.cos(theta) * vx + np.sin(theta) * vy
        omega = (-np.sin(theta) * vx + np.cos(theta) * vy) / STICK_DIST

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega

        return cmd, p_err
    
    def send_gripper(self, state):
        if self.sim:
            return

        if not self.gripper_action_client.wait_for_server(timeout_sec=ACTION_SERVER_TIMEOUT):
            self.get_logger().warning('No gripper action server found')
            return

        gripper_goal_msg = GripperControl.Goal()
        gripper_goal_msg.target_state = state
        gripper_goal_msg.power = 0.5
        self.gripper_action_client.send_goal_async(gripper_goal_msg)

    def send_arm(self, x, z):
        if self.sim:
            return

        if not self.arm_action_client.wait_for_server(timeout_sec=ACTION_SERVER_TIMEOUT):
            self.get_logger().warning('No move_arm action server found')
            return

        arm_goal_msg = MoveArm.Goal()
        arm_goal_msg.x = x
        arm_goal_msg.z = z
        arm_goal_msg.relative = False
        self.arm_action_client.send_goal_async(arm_goal_msg)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def stick_nav_and_pickup(self):
        if self.robot_pose is None:
            if not self.waiting_for_pose_logging:
                self.waiting_for_pose_logging = True
                self.get_logger().info(f'Waiting for robot {self.robot_id} pose')
            return
        self.waiting_for_pose_logging = False

        # robot x, y, theta coordinates
        x = self.robot_pose.pose.position.x
        y = self.robot_pose.pose.position.y
        theta = yaw_from_pose(self.robot_pose)

        # stick x, y cooridnates
        if self.sim:
            # stick location and orientation
            self.stick_x = SIM_STICK_X
            self.stick_y = SIM_STICK_Y
            self.stick_theta = SIM_STICK_THETA

            # puck location
            self.puck_x = SIM_PUCK_X
            self.puck_y = SIM_PUCK_Y

            # goal location
            self.goal_x = SIM_GOAL_X
            self.goal_y = SIM_GOAL_Y

        elif not self.sim:
            if self.stick_pose is None:
                if not self.waiting_for_pose_logging:
                    self.waiting_for_pose_logging = True
                    self.get_logger().info(f'Waiting for stick pose')
                return
            self.waiting_for_pose_logging = False
            self.stick_x = self.stick_pose.pose.position.x
            self.stick_y = self.stick_pose.pose.position.y

            if self.puck_pose is None:
                if not self.waiting_for_pose_logging:
                    self.waiting_for_pose_logging = True
                    self.get_logger().info(f'Waiting for puck pose')
                return
            self.waiting_for_pose_logging = False
            self.stick_x = self.stick_pose.pose.position.x
            self.stick_y = self.stick_pose.pose.position.y

            # Only the scoring robot needs to know about the net
            if self.passOrScore == 'score':
                if self.goal_pose is None:
                    if not self.waiting_for_pose_logging:
                        self.waiting_for_pose_logging = True
                        self.get_logger().info(f'Waiting for goal pose')
                    return
                self.waiting_for_pose_logging = False
                self.goal_x = self.goal_pose.pose.position.x
                self.goal_y = self.goal_pose.pose.position.y
        if self.state == 'open_gripper':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_gripper(GripperControl.Goal.OPEN)
            if self.seconds_in_state() > 1.0:
                self.go_to_state('lift_arm')
            self.stop()
            return

        if self.state == 'lift_arm':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_arm(0.0, ARM_RAISE_HEIGHT)
            if self.seconds_in_state() > 1.5:
                self.go_to_state('navigate_to_stick')
            self.stop()
            return

        if self.state == 'navigate_to_stick':
            # pregrasp pose: robot is to the right of the stick and faces left
            pregrasp_x = self.stick_x + STICK_DIST
            pregrasp_y = self.stick_y
            pregrasp_theta = np.pi

            # unicycle_linear_controller commands the stick point, not the robot center
            stick_point_x = pregrasp_x + STICK_DIST * np.cos(pregrasp_theta)
            stick_point_y = pregrasp_y + STICK_DIST * np.sin(pregrasp_theta)

            cmd, stick_point_error = self.unicycle_linear_controller(x, y, theta, stick_point_x, stick_point_y)

            point_controller_omega = cmd.angular.z
            heading_error = wrap(pregrasp_theta - theta)

            self.cmd_pub.publish(cmd)

            center_error = np.sqrt((pregrasp_x - x)**2 + (pregrasp_y - y)**2)
            controlled_point_x = x + STICK_DIST * np.cos(theta)
            controlled_point_y = y + STICK_DIST * np.sin(theta)

            now = self.get_clock().now().nanoseconds / 1e9
            if now >= self.next_debug_print:
                self.next_debug_print = now + DEBUG_PRINT_PERIOD
                self.get_logger().info(
                    'nav debug: '
                    f'pose=({x:.2f}, {y:.2f}, {theta:.2f}), '
                    f'point=({controlled_point_x:.2f}, {controlled_point_y:.2f}), '
                    f'target_point=({stick_point_x:.2f}, {stick_point_y:.2f}), '
                    f'pregrasp=({pregrasp_x:.2f}, {pregrasp_y:.2f}, {pregrasp_theta:.2f}), '
                    f'errors(point={stick_point_error:.3f}, center={center_error:.3f}, heading={heading_error:.3f}), '
                    f'cmd(v={cmd.linear.x:.3f}, omega={cmd.angular.z:.3f}, '
                    f'point_omega={point_controller_omega:.3f})'
                )

            if stick_point_error < tol and center_error < tol and abs(heading_error) < heading_tol:
                self.stop()
                self.go_to_state('close_gripper')
            return

        if self.state == 'close_gripper':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_gripper(GripperControl.Goal.CLOSE)
            if self.sim or self.seconds_in_state() > 1.0:
                self.go_to_state('navigate_puck')
            self.stop()
            return

        if self.state == 'navigate_puck':
            self.stop()
            return

def main(args=None):
    rclpy.init(args=args)
    node = Controller(1, 'pass', sim=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
            # node.robot2_cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()



        
