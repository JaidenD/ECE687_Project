import os
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robomaster_msgs.action import GripperControl, MoveArm
from std_msgs.msg import Bool, Empty, String

from robo_hockey_controller.config import (
    ACTION_SERVER_TIMEOUT,
    CBF_GAIN,
    CLF_GAIN,
    CONTROL_FREQUENCY,
    CROSS_TRACK_KP,
    DEBUG_PRINT_PERIOD,
    GOAL_MOCAP_TOPIC,
    HEADING_KP,
    HOLDER_MOCAP_TOPIC,
    HOLDER_PLATFORM_RADIUS,
    LAB_OBSTACLES,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    MOCAP_TIMEOUT,
    NAVIGATION_ENVELOPE_RADIUS,
    NAV_KP,
    NAV_LOOKAHEAD_DIST,
    OTHER_ROBOT_SAFETY_DISTANCE,
    POINT_DRIVE_HEADING_LIMIT,
    PICKUP_MAX_ANGULAR_SPEED,
    POSE_FILTER_ALPHA,
    PUCK_MOCAP_TOPIC,
    PUCK_VELOCITY_WINDOW,
    QP_SLACK_PENALTY,
    QP_SOLVER,
    ROBOT1_ID,
    ROBOT2_ID,
    ROBOT_MOCAP_TO_BASE_OFFSET_BODY,
    SIM_ARM_ACTION_TIME,
    SIM_GRIPPER_ACTION_TIME,
    SIM_OBSTACLES,
    STRAIGHT_KP,
)
from robo_hockey_controller.helpers import (
    navigation_point,
    rotate_2d,
    wrap,
    yaw_from_pose,
)
from robo_hockey_controller.hockey_states import HockeyStateHandlers
from robo_hockey_controller.navigation_qp import solve_navigation_qp
from robo_hockey_controller.pickup_states import PickupStateHandlers


class Controller(PickupStateHandlers, HockeyStateHandlers, Node):
    """One ROS node that controls either the passing or scoring robot."""

    def __init__(self, robot_id, role, sim=False):
        super().__init__(f'{role}_id_{robot_id}')

        self.robot_id = robot_id
        self.role = role
        self.sim = sim
        self.other_robot_id = ROBOT2_ID if robot_id == ROBOT1_ID else ROBOT1_ID

        # Latest filtered mocap measurements.
        self.robot_pose = None
        self.other_robot_pose = None
        self.holder_pose = None
        self.puck_pose = None
        self.goal_pose = None

        # Local receipt times are used to reject stale mocap data.
        self.robot_pose_time = None
        self.other_robot_pose_time = None
        self.holder_pose_time = None
        self.puck_pose_time = None
        self.goal_pose_time = None

        # The state machine stores a state name and dispatches the matching
        # function once per control-loop iteration.
        self.state = 'wait_for_pickup_turn'
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.next_debug_print = 0.0
        self.next_waiting_print = 0.0
        self.next_qp_warning = 0.0

        # Obstacle locations are fixed during a run.
        self.static_obstacles = SIM_OBSTACLES if self.sim else LAB_OBSTACLES
        self.minimum_barrier_value = np.inf

        # Coordination information shared between the two controller nodes.
        self.pickup_clear = False
        self.other_pickup_clear = False
        self.pass_complete = False
        self.pass_complete_time = None
        # Puck velocity is estimated from a short history of mocap positions.
        self.puck_velocity = np.zeros(2)
        self.puck_pose_history = deque()
        self.puck_slow_since = None
        self.swing_plan = None
        self.swing_start_heading = None

        # An action has two asynchronous stages: goal acceptance and result.
        self.arm_goal_future = None
        self.arm_result_future = None
        self.arm_action_active = False
        self.arm_action_failed = False
        self.sim_arm_done_at = None

        self.gripper_goal_future = None
        self.gripper_result_future = None
        self.gripper_action_active = False
        self.gripper_action_failed = False
        self.sim_gripper_done_at = None

        # Build one lookup table without hiding the merge behind dictionary **.
        self.state_handlers = self.pickup_state_handlers()
        self.state_handlers.update(self.hockey_state_handlers())

        # Best effort avoids queuing old high-rate mocap measurements.
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        # Reliable delivery is appropriate for low-rate coordination messages.
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        # The physical RoboMaster driver requests reliable velocity commands.
        # A reliable publisher also remains compatible with the simulator.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        ########################################################################
        # subscriptions
        ########################################################################
        self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{self.robot_id}/pose',
            self.save_robot_pose,
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/dji_robot_{self.other_robot_id}/pose',
            self.save_other_robot_pose,
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            HOLDER_MOCAP_TOPIC,
            self.save_holder_pose,
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            PUCK_MOCAP_TOPIC,
            self.save_puck_pose,
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            GOAL_MOCAP_TOPIC,
            self.save_goal_pose,
            pose_qos,
        )
        self.create_subscription(
            Bool,
            f'/robo_hockey/robot{self.other_robot_id}/pickup_clear',
            self.save_other_pickup_clear,
            status_qos,
        )
        if self.role == 'score':
            self.create_subscription(
                Bool,
                '/robo_hockey/pass_complete',
                self.save_pass_complete,
                status_qos,
            )
        if self.sim:
            self.create_subscription(Empty, '/sim/reset', self.reset, status_qos)

        ########################################################################
        # publishers
        ########################################################################
        self.cmd_pub = self.create_publisher(
            Twist,
            f'/robot{self.robot_id}/cmd_vel',
            command_qos,
        )
        self.state_pub = self.create_publisher(
            String,
            f'/robo_hockey/robot{self.robot_id}/state',
            status_qos,
        )
        self.pickup_clear_pub = self.create_publisher(
            Bool,
            f'/robo_hockey/robot{self.robot_id}/pickup_clear',
            status_qos,
        )
        self.pass_complete_pub = self.create_publisher(
            Bool,
            '/robo_hockey/pass_complete',
            status_qos,
        )

        # Action responses may run concurrently, but only one control-loop
        # callback should run for this node at a time.
        self.action_group = ReentrantCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()

        ########################################################################
        # action clients
        ########################################################################
        # Arm and gripper actions are simulated with timers in simulator mode.
        if not self.sim:
            self.gripper_action_client = ActionClient(
                self,
                GripperControl,
                f'/robot{self.robot_id}/gripper',
                callback_group=self.action_group,
            )
            self.arm_action_client = ActionClient(
                self,
                MoveArm,
                f'/robot{self.robot_id}/move_arm',
                callback_group=self.action_group,
            )

        # Run the state machine and controller at CONTROL_FREQUENCY Hz.
        self.timer = self.create_timer(
            1.0 / CONTROL_FREQUENCY,
            self.control_loop,
            callback_group=self.control_group,
        )

        self.get_logger().info(
            f'Controller started: robot={self.robot_id}, role={self.role}, '
            f'sim={self.sim}'
        )

    def now_seconds(self):
        """Return the ROS clock time as a floating-point number of seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    ########################################################################
    # ROS subscription callbacks
    ########################################################################

    def save_robot_pose(self, msg):
        """Save the filtered robot pose and its local receipt time."""
        base_pose = self.robot_base_pose_from_mocap(msg, self.robot_id)
        self.robot_pose = self.low_pass_pose(self.robot_pose, base_pose)
        self.robot_pose_time = self.now_seconds()

    def save_other_robot_pose(self, msg):
        """Save the filtered pose of the other robot."""
        base_pose = self.robot_base_pose_from_mocap(msg, self.other_robot_id)
        self.other_robot_pose = self.low_pass_pose(
            self.other_robot_pose,
            base_pose,
        )
        self.other_robot_pose_time = self.now_seconds()

    def save_holder_pose(self, msg):
        """Save the filtered pose of the shared stick holder."""
        self.holder_pose = self.low_pass_pose(self.holder_pose, msg)
        self.holder_pose_time = self.now_seconds()

    def save_puck_pose(self, msg):
        """Save the filtered puck pose, timestamp, and windowed velocity."""
        self.puck_pose = self.low_pass_pose(self.puck_pose, msg)
        now = self.now_seconds()
        self.puck_pose_time = now

        position = np.array([
            self.puck_pose.pose.position.x,
            self.puck_pose.pose.position.y,
        ])
        self.puck_pose_history.append((now, position))

        # Keep only the recent position samples used for finite differences.
        while (
            len(self.puck_pose_history) > 2
            and now - self.puck_pose_history[0][0] > PUCK_VELOCITY_WINDOW
        ):
            self.puck_pose_history.popleft()

        # Estimate average velocity over the window. Waiting at least 0.2 s
        # keeps small mocap time/position errors from dominating the estimate.
        if len(self.puck_pose_history) >= 2:
            first_time, first_position = self.puck_pose_history[0]
            sample_time = now - first_time
            if sample_time > 0.20:
                self.puck_velocity = (position - first_position) / sample_time

    def save_goal_pose(self, msg):
        """Save the filtered goal pose."""
        self.goal_pose = self.low_pass_pose(self.goal_pose, msg)
        self.goal_pose_time = self.now_seconds()

    def save_other_pickup_clear(self, msg):
        """Save whether the other robot has cleared the pickup area."""
        self.other_pickup_clear = msg.data

    def save_pass_complete(self, msg):
        """Save the pass status and when the pass first completed."""
        if msg.data and not self.pass_complete:
            self.pass_complete_time = self.now_seconds()
        self.pass_complete = msg.data

    def reset(self, _msg):
        """Reset controller state when the simulator publishes a reset."""
        self.stop()
        self.robot_pose = None
        self.other_robot_pose = None
        self.holder_pose = None
        self.puck_pose = None
        self.goal_pose = None
        self.pickup_clear = False
        self.other_pickup_clear = False
        self.pass_complete = False
        self.pass_complete_time = None
        self.puck_velocity = np.zeros(2)
        self.puck_pose_history.clear()
        self.puck_slow_since = None
        self.swing_plan = None
        self.swing_start_heading = None
        self.minimum_barrier_value = np.inf
        self.arm_action_active = False
        self.gripper_action_active = False
        self.go_to_state('wait_for_pickup_turn')

    ########################################################################
    # end ROS subscription callbacks
    ########################################################################

    def robot_base_pose_from_mocap(self, mocap_pose, robot_id):
        """Convert the off-center mocap marker position to the base center."""
        # Simulator poses already describe the center of the robot body.
        if self.sim:
            return mocap_pose

        theta = yaw_from_pose(mocap_pose)
        offset_body = ROBOT_MOCAP_TO_BASE_OFFSET_BODY[robot_id]
        offset_world = rotate_2d(offset_body, theta)

        base_pose = PoseStamped()
        base_pose.header = mocap_pose.header
        base_pose.pose.position.x = (
            mocap_pose.pose.position.x + offset_world[0]
        )
        base_pose.pose.position.y = (
            mocap_pose.pose.position.y + offset_world[1]
        )
        base_pose.pose.position.z = mocap_pose.pose.position.z
        base_pose.pose.orientation = mocap_pose.pose.orientation
        return base_pose

    def low_pass_pose(self, old_pose, new_pose):
        """Apply a first-order low-pass filter to one mocap pose."""
        if old_pose is None:
            return new_pose

        alpha = POSE_FILTER_ALPHA

        filtered_pose = PoseStamped()
        filtered_pose.header = new_pose.header

        # Position update: x_hat[k] = (1-alpha)x_hat[k-1] + alpha*x[k].
        filtered_pose.pose.position.x = (
            (1.0 - alpha) * old_pose.pose.position.x
            + alpha * new_pose.pose.position.x
        )
        filtered_pose.pose.position.y = (
            (1.0 - alpha) * old_pose.pose.position.y
            + alpha * new_pose.pose.position.y
        )
        filtered_pose.pose.position.z = (
            (1.0 - alpha) * old_pose.pose.position.z
            + alpha * new_pose.pose.position.z
        )

        # Filter the shortest angular difference so crossing +/-pi does not
        # create an artificial full rotation.
        old_theta = yaw_from_pose(old_pose)
        new_theta = yaw_from_pose(new_pose)
        theta = wrap(old_theta + alpha * wrap(new_theta - old_theta))

        filtered_pose.pose.orientation.z = np.sin(theta / 2.0)
        filtered_pose.pose.orientation.w = np.cos(theta / 2.0)

        return filtered_pose

    def wait_for_pose(self, name, pose, pose_time):
        """Return False and stop the robot when a required pose is stale."""
        # check that the pose exists and was received recently enough to use
        now = self.now_seconds()
        pose_is_fresh = (
            pose is not None
            and pose_time is not None
            and now - pose_time <= MOCAP_TIMEOUT
        )
        if pose_is_fresh:
            return True

        if now >= self.next_waiting_print:
            self.next_waiting_print = now + 1.0
            self.get_logger().info(f'Waiting for fresh {name} mocap')

        self.stop()

        return False

    def go_to_state(self, new_state):
        """Change state and mark its next callback as the first iteration."""
        self.state = new_state
        self.state_started = self.get_clock().now()
        self.just_entered_state = True
        self.next_debug_print = 0.0
        self.get_logger().info(f'State: {new_state}')

    def publish_status(self):
        """Publish state and the two robot-to-robot coordination flags."""
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        clear_msg = Bool()
        clear_msg.data = self.pickup_clear
        self.pickup_clear_pub.publish(clear_msg)

        pass_msg = Bool()
        pass_msg.data = self.pass_complete if self.role == 'pass' else False
        if self.role == 'pass':
            self.pass_complete_pub.publish(pass_msg)

    @staticmethod
    def point_velocity_to_twist(point_velocity, theta):
        # Invert the approximate-linearization map. If u is the desired
        # navigation-point velocity, then
        #   v     = [ cos(theta), sin(theta)] u
        #   omega = [-sin(theta), cos(theta)] u / l.
        c = np.cos(theta)
        s = np.sin(theta)

        cmd = Twist()
        cmd.linear.x = float(c * point_velocity[0] + s * point_velocity[1])
        cmd.angular.z = float(
            (-s * point_velocity[0] + c * point_velocity[1])
            / NAV_LOOKAHEAD_DIST
        )
        return cmd

    ########################################################################
    # Start of safe navigation functions
    ########################################################################
    def navigation_obstacles(self, include_other_robot, include_holder=False):
        """Return static obstacle centers and their required clearances."""
        centers = []
        safety_distances = []

        for obstacle in self.static_obstacles:
            centers.append(obstacle['center'])
            # The QP controls one point, so inflate the physical obstacle by
            # the radius assigned to the robot/stick navigation envelope.
            safe_distance = obstacle['radius'] + NAVIGATION_ENVELOPE_RADIUS
            safety_distances.append(safe_distance)

        if include_holder:
            # The holder position comes from mocap, so it cannot be entered in
            # LAB_OBSTACLES as a fixed world coordinate. Treat its rectangular
            # platform as a conservative circle during the long approach. The
            # later straight pickup states intentionally do not use this CBF.
            holder_center = np.array([
                self.holder_pose.pose.position.x,
                self.holder_pose.pose.position.y,
            ])
            centers.append(holder_center)
            safety_distances.append(
                HOLDER_PLATFORM_RADIUS + NAVIGATION_ENVELOPE_RADIUS
            )

        if include_other_robot:
            other_x = self.other_robot_pose.pose.position.x
            other_y = self.other_robot_pose.pose.position.y
            other_theta = yaw_from_pose(self.other_robot_pose)
            
            # Use the other robot's navigation point as an additional static
            # obstacle. The state machine normally moves only one robot at once.
            other_point = navigation_point(other_x, other_y, other_theta)
            centers.append(other_point)
            safety_distances.append(OTHER_ROBOT_SAFETY_DISTANCE)

        centers = np.asarray(centers, dtype=float).reshape((-1, 2))
        safety_distances = np.asarray(safety_distances, dtype=float)
        return centers, safety_distances

    def safe_navigation_controller(
        self,
        x,
        y,
        theta,
        point_desired,
        include_other_robot,
        include_holder=False,
    ):
        # The nominal controller points directly at the goal. The QP changes
        # this velocity only when needed for the CLF, CBFs, or speed limits.
        point = navigation_point(x, y, theta)
        point_error_vector = point_desired - point
        point_error = np.linalg.norm(point_error_vector)
        nominal_velocity = NAV_KP * point_error_vector
        
        centers, safety_distances = self.navigation_obstacles(
            include_other_robot,
            include_holder,
        )

        try:
            result = solve_navigation_qp(
                point=point,
                target=point_desired,
                theta=theta,
                lookahead_distance=NAV_LOOKAHEAD_DIST,
                obstacle_centers=centers,
                safety_distances=safety_distances,
                nominal_velocity=nominal_velocity,
                clf_gain=CLF_GAIN,
                cbf_gain=CBF_GAIN,
                slack_penalty=QP_SLACK_PENALTY,
                max_linear_speed=MAX_LINEAR_SPEED,
                max_angular_speed=MAX_ANGULAR_SPEED,
                solver=QP_SOLVER,
            )
        except Exception as error:
            result = None
            failure_message = f'QP solver error: {error}'
        else:
            failure_message = 'QP is infeasible'

        if result is None:
            now = self.now_seconds()
            if now >= self.next_qp_warning:
                self.next_qp_warning = now + 1.0
                self.get_logger().error(
                    f'{failure_message}; publishing zero Twist'
                )
            return Twist(), point_error, None

        if len(result['barrier_values']) > 0:
            minimum_barrier = float(np.min(result['barrier_values']))
            self.minimum_barrier_value = min(
                self.minimum_barrier_value,
                minimum_barrier,
            )

        cmd = self.point_velocity_to_twist(result['point_velocity'], theta)
        return cmd, point_error, result

    @staticmethod
    def qp_debug_text(result):
        """Format the CLF-CBF-QP diagnostics printed during navigation."""
        if result is None:
            return 'qp=failed'

        if len(result['barrier_values']) > 0:
            minimum_barrier = float(np.min(result['barrier_values']))
        else:
            minimum_barrier = np.inf

        return (
            f"clf={result['clf_value']:.3f}, min_h={minimum_barrier:.3f}, "
            f"delta={result['slack']:.3f}, "
            f"qp_time={1000.0 * result['solve_time']:.2f}ms"
        )

    ########################################################################
    # End of safe navigation functions
    ########################################################################

    ########################################################################
    # Start of control commands
    ########################################################################
    def heading_command(self, theta, desired_theta, max_angular_speed):
        """Rotate in place using proportional heading feedback."""
        heading_error = wrap(desired_theta - theta)

        cmd = Twist()
        angular_speed = HEADING_KP * heading_error
        cmd.angular.z = float(np.clip(
            angular_speed,
            -max_angular_speed,
            max_angular_speed,
        ))
        return cmd, heading_error

    def point_drive_command(self, x, y, theta, target_position, max_linear_speed):
        """Turn toward a point while driving forward to it."""
        position_error = target_position - np.array([x, y])
        distance = np.linalg.norm(position_error)

        if distance > 1e-9:
            target_heading = np.arctan2(position_error[1], position_error[0])
        else:
            target_heading = theta

        heading_error = wrap(target_heading - theta)

        # Use full speed outside 0.25 m and ramp linearly to zero near the goal.
        speed_scale = min(1.0, distance / 0.25)

        # A large heading error means forward motion would make the robot drive
        # in a circle around the point. Rotate in place until it faces the point.
        if abs(heading_error) > POINT_DRIVE_HEADING_LIMIT:
            forward_scale = 0.0
        else:
            forward_scale = np.cos(heading_error)

        cmd = Twist()
        cmd.linear.x = float(max_linear_speed * speed_scale * forward_scale)
        angular_speed = HEADING_KP * heading_error
        cmd.angular.z = float(np.clip(
            angular_speed,
            -PICKUP_MAX_ANGULAR_SPEED,
            PICKUP_MAX_ANGULAR_SPEED,
        ))

        return cmd, distance, heading_error

    def straight_command(
        self,
        x,
        y,
        theta,
        target_position,
        desired_theta,
        max_linear_speed,
        allow_reverse,
    ):
        """Move along a fixed heading and correct lateral displacement."""

        # direction is along the desired path; normal points to its left.
        direction = np.array([np.cos(desired_theta), np.sin(desired_theta)])
        normal = np.array([-direction[1], direction[0]])
        position_error = target_position - np.array([x, y])
        forward_error = float(position_error @ direction)
        lateral_error = float(position_error @ normal)
        heading_error = wrap(desired_theta - theta)

        # The sign of forward_error says whether the target is ahead or behind.
        linear_speed = STRAIGHT_KP * forward_error
        if allow_reverse:
            linear_speed = np.clip(linear_speed, -max_linear_speed, max_linear_speed)
        else:
            linear_speed = np.clip(linear_speed, 0.0, max_linear_speed)

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        angular_speed = (
            HEADING_KP * heading_error
            + CROSS_TRACK_KP * lateral_error
        )
        cmd.angular.z = float(np.clip(
            angular_speed,
            -PICKUP_MAX_ANGULAR_SPEED,
            PICKUP_MAX_ANGULAR_SPEED,
        ))

        distance = np.linalg.norm(position_error)
        return cmd, distance, lateral_error, heading_error
    ########################################################################
    # End of safe navigation commands
    ########################################################################


    ########################################################################
    # Start of gripper functions
    ########################################################################
    def start_gripper_action(self, target_state):
        # This function sends the goal once. poll_gripper_action() checks the
        # asynchronous acceptance/result futures on later control iterations.
        self.gripper_action_active = True
        self.gripper_action_failed = False
        self.gripper_goal_future = None
        self.gripper_result_future = None

        if self.sim:
            # The simulator has no physical action server, so use a fixed delay.
            self.sim_gripper_done_at = self.now_seconds() + SIM_GRIPPER_ACTION_TIME
            return

        if not self.gripper_action_client.wait_for_server(
            timeout_sec=ACTION_SERVER_TIMEOUT
        ):
            self.get_logger().error('No gripper action server found')
            self.gripper_action_failed = True
            return

        goal = GripperControl.Goal()
        goal.target_state = target_state
        goal.power = 1.0

        self.gripper_goal_future = (
            self.gripper_action_client.send_goal_async(goal)
        )

    def poll_gripper_action(self):
        # Return one of three simple states for run_action_state():
        # pending, done, or failed.
        if self.gripper_action_failed:
            return 'failed'
        if not self.gripper_action_active:
            return 'done'
        if self.sim:
            if self.now_seconds() >= self.sim_gripper_done_at:
                self.gripper_action_active = False
                return 'done'
            return 'pending'

        if self.gripper_goal_future is not None:
            # First wait for the action server to accept or reject the goal.
            if not self.gripper_goal_future.done():
                return 'pending'
            goal_handle = self.gripper_goal_future.result()
            self.gripper_goal_future = None
            if not goal_handle.accepted:
                self.get_logger().error('Gripper action was rejected')
                self.gripper_action_failed = True
                return 'failed'
            self.gripper_result_future = goal_handle.get_result_async()

        if self.gripper_result_future is not None:
            # Once accepted, wait for the physical action to finish.
            if not self.gripper_result_future.done():
                return 'pending'
            self.gripper_result_future = None
            self.gripper_action_active = False
            return 'done'
        return 'pending'

    ########################################################################
    # End of gripper functions
    ########################################################################


    ########################################################################
    # Start of arm functions
    ########################################################################
    def start_arm_action(self, x, z):
        # MoveArm coordinates are absolute x/z values in arm_base_link.
        self.arm_action_active = True
        self.arm_action_failed = False
        self.arm_goal_future = None
        self.arm_result_future = None

        if self.sim:
            self.sim_arm_done_at = self.now_seconds() + SIM_ARM_ACTION_TIME
            return

        if not self.arm_action_client.wait_for_server(
            timeout_sec=ACTION_SERVER_TIMEOUT
        ):
            self.get_logger().error('No move_arm action server found')
            self.arm_action_failed = True
            return

        goal = MoveArm.Goal()
        goal.x = x
        goal.z = z
        goal.relative = False
        self.arm_goal_future = self.arm_action_client.send_goal_async(goal)

    def poll_arm_action(self):
        # This follows the same two-stage asynchronous flow as the gripper:
        # wait for goal acceptance, then wait for the result.
        if self.arm_action_failed:
            return 'failed'
        if not self.arm_action_active:
            return 'done'
        if self.sim:
            if self.now_seconds() >= self.sim_arm_done_at:
                self.arm_action_active = False
                return 'done'
            return 'pending'

        if self.arm_goal_future is not None:
            if not self.arm_goal_future.done():
                return 'pending'
            goal_handle = self.arm_goal_future.result()
            self.arm_goal_future = None
            if not goal_handle.accepted:
                self.get_logger().error('MoveArm action was rejected')
                self.arm_action_failed = True
                return 'failed'
            self.arm_result_future = goal_handle.get_result_async()

        if self.arm_result_future is not None:
            if not self.arm_result_future.done():
                return 'pending'
            self.arm_result_future = None
            self.arm_action_active = False
            return 'done'
        return 'pending'

    ########################################################################
    # End of arm functions
    ########################################################################
    
    ########################################################################
    # Start state machine functions
    ########################################################################
    def stop(self):
        # A default Twist has every linear and angular component equal to zero.
        # External shutdown can invalidate ROS before finally runs, in which
        # case publishing is no longer possible.
        if rclpy.ok():
            try:
                self.cmd_pub.publish(Twist())
            except rclpy._rclpy_pybind11.RCLError:
                # Ctrl+C can request publisher destruction just before cleanup.
                pass

    def stop_and_go_to_state(self, new_state):
        self.stop()
        self.go_to_state(new_state)

    def publish_motion(self, label, pose, error_text, cmd):
        self.cmd_pub.publish(cmd)

        # print motion diagnostics at a slower rate than the control loop
        now = self.now_seconds()
        if now >= self.next_debug_print:
            self.next_debug_print = now + DEBUG_PRINT_PERIOD
            x, y, theta = pose
            self.get_logger().info(
                f'{label}: pose=({x:.2f}, {y:.2f}, {theta:.2f}), '
                f'{error_text}, '
                f'cmd=(v={cmd.linear.x:.2f}, omega={cmd.angular.z:.2f})'
            )

    def run_action_state(self, start_action, poll_action, next_state, action_name):
        # State handlers run at 50 Hz. Start the ROS action only on the first
        # iteration, then poll without blocking until it finishes.
        self.stop()
        if self.just_entered_state:
            self.just_entered_state = False
            start_action()

        status = poll_action()
        if status == 'done':
            self.go_to_state(next_state)
        elif status == 'failed':
            # stop immediately and enter the terminal error state
            self.stop()
            self.get_logger().error(
                f'{action_name} failed; controller stopped'
            )
            self.go_to_state('error')

    ########################################################################
    # End of state machine functions
    ########################################################################

    ########################################################################
    # main control-loop timer callback
    ########################################################################

    def control_loop(self):
        # This timer callback is the state-machine dispatcher.
        self.publish_status()

        if self.state in ('error', 'done', 'pass_complete'):
            self.stop()
            return

        # wait for current mocap data before constructing the planar pose
        if not self.wait_for_pose('robot', self.robot_pose, self.robot_pose_time):
            return
        pose = (
            self.robot_pose.pose.position.x,
            self.robot_pose.pose.position.y,
            yaw_from_pose(self.robot_pose),
        )

        # Map the current state string to its function and execute one step.
        handler = self.state_handlers.get(self.state)
        if handler is None:
            self.get_logger().error(f'Unknown state: {self.state}')
            self.go_to_state('error')
            return

        handler(pose)

    ########################################################################
    # end main control-loop timer callback
    ########################################################################


def main(args=None):
    # Initialize ROS before creating any nodes, publishers, or timers.
    rclpy.init(args=args)

    # the simulator launcher sets this to 1; an unset value means hardware mode
    sim = os.environ.get('ROBO_HOCKEY_SIM') == '1'

    # One process contains both robot controllers. Each object has its own
    # state machine, topics, timer, and action clients.
    nodes = [
        Controller(ROBOT1_ID, 'pass', sim=sim),
        Controller(ROBOT2_ID, 'score', sim=sim),
    ]

    # The executor waits for ROS work and assigns ready callbacks to its
    # worker threads. spin() blocks here until Ctrl+C or external shutdown.
    executor = MultiThreadedExecutor(num_threads=4)
    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Always publish zero velocity before destroying ROS resources.
        for node in nodes:
            node.stop()
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
