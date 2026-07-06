import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from robomaster_msgs.action import GripperControl, MoveArm
from std_msgs.msg import ColorRGBA


FIRST_ROBOT_ID = 5
# SECOND_ROBOT_ID = 2

# TEMPORARY: replace these with the stick mocap topic in the lab
STICK_X = 0.50
STICK_Y = 0.50
STICK_THETA = 0.0          # Direction the robot should face when grabbing.

# TEMPORARY: replace these with the puck mocap topic in the lab
PUCK_X = 0.80
PUCK_Y = 0.50

APPROACH_DISTANCE = 0.35   # Stop this far behind the stick first.
TOOL_OFFSET = 0.25         # Robot center to gripper/stick contact point.
STICK_LENGTH = 1.0         # TODO: tune in lab

WIND_UP_ANGLE = np.pi / 8
FOLLOW_THROUGH_ANGLE = 2 * np.pi / 8

USE_ARM_ACTIONS = True
USE_GRIPPER_ACTIONS = True

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


# Keep angle in [-pi, pi] range
def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PickupController(Node):
    def __init__(self):
        super().__init__('pickup_controller') # Create ROS node

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{FIRST_ROBOT_ID}/pose',
            self.save_pose,
            qos,
        )
        """
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{SECOND_ROBOT_ID}/pose',
            self.save_robot2_pose,
            qos,
        )
        """
        self.cmd_pub = self.create_publisher(Twist, f'/robot{FIRST_ROBOT_ID}/cmd_vel', qos)
        # self.robot2_cmd_pub = self.create_publisher(Twist, f'/robot{SECOND_ROBOT_ID}/cmd_vel', qos)
        self.led_pub = self.create_publisher(ColorRGBA, f'/robot{FIRST_ROBOT_ID}/leds/color', qos)

        self.gripper_client = ActionClient(
            self, GripperControl, f'/robot{FIRST_ROBOT_ID}/gripper')
        self.arm_client = ActionClient(
            self, MoveArm, f'/robot{FIRST_ROBOT_ID}/move_arm')

        self.pose = None
        self.robot2_pose = None
        self.state = 'open_gripper' # set initial state
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.waiting_for_pose_logged = False
        self.wind_angle = None
        self.follow_through_angle = None

        approach_x = math.cos(STICK_THETA)
        approach_y = math.sin(STICK_THETA)
        self.pregrasp_x = STICK_X - APPROACH_DISTANCE * approach_x
        self.pregrasp_y = STICK_Y - APPROACH_DISTANCE * approach_y

        self.set_led(0.0, 0.2, 1.0)
        self.timer = self.create_timer(0.05, self.control_step)

        self.get_logger().info(
            f'Pickup demo started for robot {FIRST_ROBOT_ID}. '
            f'Stick=({STICK_X:.2f}, {STICK_Y:.2f}, {STICK_THETA:.2f}), '
            f'Puck=({PUCK_X:.2f}, {PUCK_Y:.2f})'
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
        if self.pose is None:
            if not self.waiting_for_pose_logged:
                self.waiting_for_pose_logged = True
                self.get_logger().info('Waiting for robot 1 mocap pose...')
            return

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # formula 2 from
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation?utm_source=chatgpt.com#Comparison_with_other_representations_of_rotations
        # We want to find the rotation angle about the z-axis of the robot.
        # R11 = cos(theta), R21 = sin(theta), so theta = atan2(R21, R11).
        q = self.pose.pose.orientation
        theta = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        ##########################################
        #           Command sequence             #
        ##########################################

        # execute the following sequence of events with short delays between each
        # open gripper -> lower arm -> drive to location of stick
        # -> approach stick -> close gripper -> lift arm -> face puck -> wind up -> strike
        if self.state == 'open_gripper':
            if self.just_entered_state:
                self.just_entered_state = False
                self.set_led(0.0, 0.2, 1.0) # for simulation to indicate gripper state
                self.send_gripper(GripperControl.Goal.OPEN)

            if self.seconds_in_state() > 1.0:
                self.go_to_state('lift_arm')

            # set linear and angular velocity to zero (check if this makes a diff in lab)
            self.cmd_pub.publish(Twist())
            return

        if self.state == 'lift_arm':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_arm(ARM_CARRY_X, ARM_CARRY_Z)

            if self.seconds_in_state() > 1.5:
                self.go_to_state('drive_to_pregrasp')

            self.cmd_pub.publish(Twist())
            return

        # TODO: get correct location
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
                self.go_to_state('lift_arm_again')

            self.cmd_pub.publish(Twist())
            return

        # Lift arm to clear remove stick from slot
        if self.state == 'lift_arm_again':
            if self.just_entered_state:
                self.just_entered_state = False
                self.send_arm(ARM_CARRY_X, ARM_CARRY_Z)

            if self.seconds_in_state() > 1.5:
                self.go_to_state('face_puck')

            self.cmd_pub.publish(Twist())
            return

        # TODO: replace PUCK_X/PUCK_Y with mocap.
        # No navigation to the puck here; this assumes we are already in striking range.
        if self.state == 'face_puck':
            puck_direction = math.atan2(PUCK_Y - y, PUCK_X - x)
            cmd = self.heading_command(theta, puck_direction)
            self.cmd_pub.publish(cmd)

            if abs(wrap(puck_direction - theta)) < HEADING_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('wind_up_strike')
            return

        #TODO: Some drive to puck procedure

        if self.state == 'wind_up_strike':
            if self.just_entered_state:
                self.just_entered_state = False
                self.wind_angle = theta - WIND_UP_ANGLE # this sign will change depending on what side the pass is being done from

            cmd = self.heading_command(theta, self.wind_angle)
            self.cmd_pub.publish(cmd)

            if abs(wrap(self.wind_angle - theta)) < HEADING_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.go_to_state('strike_puck')
            return

        if self.state == 'strike_puck':
            if self.just_entered_state:
                self.just_entered_state = False
                self.follow_through_angle = self.wind_angle + FOLLOW_THROUGH_ANGLE # same here

            cmd = self.heading_command(theta, self.follow_through_angle, Kp=15) # gains should be more aggressive for a hard swing
            self.cmd_pub.publish(cmd)

            if abs(wrap(self.follow_through_angle - theta)) < HEADING_TOLERANCE:
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
        error = math.sqrt(error_x**2 + error_y**2) # euclidean length of error vector

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
    def heading_command(self, theta, target_theta, Kp=KP_HEADING):
        cmd = Twist()
        error = wrap(target_theta - theta)
        cmd.angular.z = np.clip(Kp * error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
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
            # node.robot2_cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
