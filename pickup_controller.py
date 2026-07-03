import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from robomaster_msgs.action import GripperControl, MoveArm
from std_msgs.msg import ColorRGBA

import numpy as np


ROBOT_ID = 1
SECOND_ROBOT_ID = 2

STICK_X = 0.50
STICK_Y = 0.50

STICK_THETA = 0.0          # Direction the robot should face when grabbing.

APPROACH_DISTANCE = 0.35   # Stop this far behind the stick first.
TOOL_OFFSET = 0.25         # Robot center to gripper/stick contact point.

USE_ARM_ACTIONS = False
USE_GRIPPER_ACTIONS = False

# These arm values need tuning on the real robot.
ARM_PICKUP_X = 0.10
ARM_PICKUP_Z = 0.02
ARM_CARRY_X = 0.00
ARM_CARRY_Z = 0.08

# Gains
KP_POSITION = 1.4
KP_HEADING = 1.8

MAX_LINEAR_SPEED = 0.35
MAX_ANGULAR_SPEED = 1.2
POSITION_TOLERANCE = 0.05
HEADING_TOLERANCE = 0.12

# TEMPORARY
ROBOT2_X = -0.5
ROBOT2_Y = -0.5

# Keep angle in [-pi, pi] range
def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PickupController(Node):
    def __init__(self):
        super().__init__('pickup_controller') # Create ROS node

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{ROBOT_ID}/pose',
            self.save_pose,
            qos,
        )
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{SECOND_ROBOT_ID}/pose',
            self.save_robot2_pose,
            qos,
        )
        self.cmd_pub = self.create_publisher(Twist, f'/robot{ROBOT_ID}/cmd_vel', qos)
        self.robot2_cmd_pub = self.create_publisher(Twist, f'/robot{SECOND_ROBOT_ID}/cmd_vel', qos)
        self.led_pub = self.create_publisher(ColorRGBA, f'/robot{ROBOT_ID}/leds/color', qos)

        self.gripper_client = ActionClient(
            self, GripperControl, f'/robot{ROBOT_ID}/gripper')
        self.arm_client = ActionClient(
            self, MoveArm, f'/robot{ROBOT_ID}/move_arm')

        self.pose = None
        self.robot2_pose = None
        self.state = 'position_scoring_robot'
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.waiting_for_pose_logged = False

        approach_x = math.cos(STICK_THETA)
        approach_y = math.sin(STICK_THETA)
        self.pregrasp_x = STICK_X - APPROACH_DISTANCE * approach_x
        self.pregrasp_y = STICK_Y - APPROACH_DISTANCE * approach_y

        self.set_led(0.0, 0.2, 1.0)
        self.timer = self.create_timer(0.05, self.control_step)

        self.get_logger().info(
            f'Pickup demo started for robot {ROBOT_ID}. '
            f'Stick=({STICK_X:.2f}, {STICK_Y:.2f}, {STICK_THETA:.2f}), '
            f'second_robot_pose=/vrpn_mocap/dji_robot_{SECOND_ROBOT_ID}/pose'
        )

    def save_pose(self, msg):
        self.pose = msg

    def save_robot2_pose(self, msg):
        self.robot2_pose = msg

    def go_to_state(self, new_state):
        self.state = new_state
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.get_logger().info(f'State: {new_state}')

    def seconds_in_state(self):
        dt = self.get_clock().now() - self.state_started
        return dt.nanoseconds / 1e9

    def control_step(self):
        if self.pose is None or self.robot2_pose is None:
            if not self.waiting_for_pose_logged:
                self.waiting_for_pose_logged = True
                self.get_logger().info('Waiting for robot 1 and robot 2 mocap poses...')
            return

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        robot2_x = self.robot2_pose.pose.position.x
        robot2_y = self.robot2_pose.pose.position.y

        # formula 2 from
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation?utm_source=chatgpt.com#Comparison_with_other_representations_of_rotations
        # We want to find the rotation angle about the z-axis of the robot.
        # R11 = cos(theta), R21 = sin(theta), so theta = atan2(R21, R11).
        q = self.pose.pose.orientation
        theta = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        q2 = self.robot2_pose.pose.orientation
        theta2 = math.atan2(
        2.0 * (q2.w * q2.z + q2.x * q2.y),
        1.0 - 2.0 * (q2.y * q2.y + q2.z * q2.z),
        )


        ##########################################
        #           Command sequence             #
        ##########################################

        if self.state == 'position_scoring_robot':
            cmd, error = self.point_command(robot2_x, robot2_y, theta2, ROBOT2_X, ROBOT2_Y)
            self.cmd_pub.publish(Twist())
            self.robot2_cmd_pub.publish(cmd)

            if error < POSITION_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.robot2_cmd_pub.publish(Twist())
                self.go_to_state('open_gripper')
            return

        # execute the following sequence of events with 1 second delays between each
        # open gripper -> lower arm -> drive to location of stick 
        # -> approach stick -> close gripper -> lift arm -> rotate to face robot 2
        if self.state == 'open_gripper':
            if self.just_entered_state:
                self.just_entered_state = False
                self.set_led(0.0, 0.2, 1.0) # for simulation to indicate gripper state
                self.send_gripper(GripperControl.Goal.OPEN)

            if self.seconds_in_state() > 1.0:
                self.go_to_state('lower_arm')

            # set linear and angular velocity to zero (check if this makes a diff in lab)
            self.cmd_pub.publish(Twist())
            return

        if self.state == 'lower_arm':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_arm(ARM_PICKUP_X, ARM_PICKUP_Z)

            if self.seconds_in_state() > 1.5:
                self.go_to_state('drive_to_pregrasp')

            self.cmd_pub.publish(Twist())
            return

        if self.state == 'drive_to_pregrasp':
            cmd, error = self.tool_point_command(x, y, theta, self.pregrasp_x, self.pregrasp_y)
            self.cmd_pub.publish(cmd)

            if error < POSITION_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('face_stick')
            return

        if self.state == 'face_stick':
            cmd = self.heading_command(theta, STICK_THETA)
            self.cmd_pub.publish(cmd)

            if abs(wrap(STICK_THETA - theta)) < HEADING_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('approach_stick')
            return

        if self.state == 'approach_stick':
            cmd, error = self.tool_point_command(x, y, theta, STICK_X, STICK_Y)
            cmd.linear.x = np.clip(cmd.linear.x, -0.18, 0.18)
            cmd.angular.z = np.clip(cmd.angular.z, -0.7, 0.7)
            self.cmd_pub.publish(cmd)

            if error < POSITION_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('close_gripper')
            return

        if self.state == 'close_gripper':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_gripper(GripperControl.Goal.CLOSE)
                self.set_led(0.0, 1.0, 0.0)

            if self.seconds_in_state() > 1.0:
                self.go_to_state('lift_arm')

            self.cmd_pub.publish(Twist())
            return

        if self.state == 'lift_arm':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_arm(ARM_CARRY_X, ARM_CARRY_Z)

            if self.seconds_in_state() > 1.5:
                self.go_to_state('face_second_robot')

            self.cmd_pub.publish(Twist())
            return

        if self.state == 'face_second_robot':
            robot2_direction = math.atan2(robot2_y - y, robot2_x - x)
            cmd = self.heading_command(theta, robot2_direction)
            self.cmd_pub.publish(cmd)

            if abs(wrap(robot2_direction - theta)) < HEADING_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('done')
            return

        if self.state == 'done':
            self.cmd_pub.publish(Twist())
            return

    def tool_point_command(self, x, y, theta, target_x, target_y):
        tool_x = x + TOOL_OFFSET * math.cos(theta)
        tool_y = y + TOOL_OFFSET * math.sin(theta)

        error_x = target_x - tool_x
        error_y = target_y - tool_y
        error = math.hypot(error_x, error_y) # euclidean length of error vector

        desired_tool_vx = KP_POSITION * error_x
        desired_tool_vy = KP_POSITION * error_y

        # velocity vector proportional to position error
        v = math.cos(theta) * desired_tool_vx + math.sin(theta) * desired_tool_vy

        # angular velocity vector proportional to rotation error
        omega = (-math.sin(theta) * desired_tool_vx + math.cos(theta) * desired_tool_vy) / TOOL_OFFSET

        cmd = Twist()
        cmd.linear.x = np.clip(v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        cmd.angular.z = np.clip(omega, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return cmd, error

    def point_command(self, x, y, theta, target_x, target_y):
        error_x = target_x - x
        error_y = target_y - y
        error = math.hypot(error_x, error_y)

        desired_vx_global = KP_POSITION * error_x
        desired_vy_global = KP_POSITION * error_y

        vx_body = math.cos(theta) * desired_vx_global + math.sin(theta) * desired_vy_global
        vy_body = -math.sin(theta) * desired_vx_global + math.cos(theta) * desired_vy_global

        cmd = Twist()
        cmd.linear.x = np.clip(vx_body, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        cmd.linear.y = np.clip(vy_body, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
        cmd.angular.z = 0.0
        return cmd, error

    # Change angle robot is facing 
    def heading_command(self, theta, target_theta):
        cmd = Twist()
        error = wrap(target_theta - theta)
        cmd.angular.z = np.clip(KP_HEADING * error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        return cmd

    def send_gripper(self, target_state):
        if not USE_GRIPPER_ACTIONS:
            self.get_logger().info(f'Sim mode: would command gripper state {target_state}')
            return

        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warning('No gripper action server found')
            return

        goal = GripperControl.Goal()
        goal.target_state = target_state
        goal.power = 0.5
        self.gripper_client.send_goal_async(goal)

    def send_arm(self, x, z):
        if not USE_ARM_ACTIONS:
            self.get_logger().info(f'Sim mode: would move arm to x={x:.2f}, z={z:.2f}')
            return

        if not self.arm_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warning('No move_arm action server found')
            return

        goal = MoveArm.Goal()
        goal.x = x
        goal.z = z
        goal.relative = False
        self.arm_client.send_goal_async(goal)

    def set_led(self, r, g, b):
        msg = ColorRGBA()
        msg.r = r
        msg.g = g
        msg.b = b
        msg.a = 1.0
        self.led_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PickupController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
            node.robot2_cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
