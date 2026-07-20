import os
from math import cos, sin

import matplotlib
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Empty, String

# Tests set this to 1 so Matplotlib does not try to open a desktop window.
HEADLESS = os.environ.get('ROBO_HOCKEY_HEADLESS') == '1'
matplotlib.use('Agg' if HEADLESS else 'Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from multi_robomaster_ros_sim.config import (
    BASE_TO_GRIPPER,
    GOAL_MOCAP_TOPIC,
    HOLDER_MOCAP_TOPIC,
    NAV_LOOKAHEAD_DIST,
    NAVIGATION_ENVELOPE_RADIUS,
    PUCK_FRICTION_COEFFICIENT,
    PUCK_LINEAR_DRAG,
    PUCK_MASS,
    PUCK_MOCAP_TOPIC,
    PUCK_QUADRATIC_DRAG,
    PUCK_RADIUS,
    ROBOT1_ID,
    ROBOT2_ID,
    SCORING_RECEIVE_OFFSET_BODY,
    SIM_GOAL_X,
    SIM_GOAL_Y,
    SIM_HOLDER_THETA,
    SIM_HOLDER_X,
    SIM_HOLDER_Y,
    SIM_OBSTACLES,
    SIM_PUCK_X,
    SIM_PUCK_Y,
    STICK_EFFECTIVE_MASS,
    STICK_HEADING_OFFSET_FROM_ROBOT,
    STICK_PUCK_RESTITUTION,
    STICK_SLOT_OFFSETS,
    STICK_TIP_FROM_BASE,
)
from multi_robomaster_ros_sim.helpers import rotate_2d


# Process noise perturbs the simulated true state while a robot is moving.
PROCESS_POSITION_NOISE_STD = 0.0005  # metres per simulation step
PROCESS_YAW_NOISE_STD = 0.0005       # radians per simulation step

# Mocap noise perturbs only the published measurement, not the true state.
MOCAP_POSITION_NOISE_STD = 0.05  # metres per measurement
MOCAP_YAW_NOISE_STD = 0.06       # radians per measurement

STICK_TIP_RADIUS = 0.035
IMPACT_COOLDOWN = 0.25  # prevents one contact from becoming several impacts


class MultiRoboMasterSim(Node):
    """Simple planar simulator with the same ROS topics as the lab system."""

    def __init__(self):
        super().__init__('multi_robomaster_sim')

        self.ROBOT_IDS = [ROBOT1_ID, ROBOT2_ID]
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 20  # milliseconds
        self.TIMEOUT_GET_POSES = 10  # milliseconds
        self.TIMEOUT_CHASSIS_SPEED = 500  # milliseconds
        # One dynamics step combines the command and pose-update periods used
        # by the original simulator.
        self.DT = (
            self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES
        ) / 1000.0
        self.MAX_LINEAR_SPEED = 1.0  # meters / second
        self.MAX_ANGULAR_SPEED = 2.0 * np.pi  # radians / second
        self.ENV = [-2.0, -2.0, 4.0, 4.0]
        self.ROBOT_SIZE = [0.24, 0.32]  # [width, length]
        self.GRIPPER_SIZE = 0.10

        seed = int(os.environ.get('ROBO_HOCKEY_SIM_SEED', '7'))
        self.rng = np.random.default_rng(seed)

        # states[id] = [world x, world y, yaw].
        # velocities[id] = [world x velocity, world y velocity, yaw rate].
        self.states = {}
        self.velocities = {}
        self.last_cmd_time = {}
        self.leds = {}
        self.controller_states = {}
        self.sticks_attached = {}
        # Previous tip locations are needed to detect a swept impact between
        # simulation frames instead of checking only the current point.
        self.previous_stick_tips = {}
        self.last_impact_time = {}
        self.minimum_swing_tip_distance = {}
        self.puck_position = np.zeros(2)
        self.puck_velocity = np.zeros(2)

        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.robot_pose_pubs = {}
        self.velocity_subs = {}
        self.led_subs = {}
        self.state_subs = {}
        for robot_id in self.ROBOT_IDS:
            self.robot_pose_pubs[robot_id] = self.create_publisher(
                PoseStamped,
                f'/vrpn_mocap/dji_robot_{robot_id}/pose',
                pose_qos,
            )
            # rid=robot_id captures the current loop value in each callback.
            self.velocity_subs[robot_id] = self.create_subscription(
                Twist,
                f'/robot{robot_id}/cmd_vel',
                lambda msg, rid=robot_id: self.velocity_callback(msg, rid),
                pose_qos,
            )
            self.led_subs[robot_id] = self.create_subscription(
                ColorRGBA,
                f'/robot{robot_id}/leds/color',
                lambda msg, rid=robot_id: self.led_callback(msg, rid),
                pose_qos,
            )
            self.state_subs[robot_id] = self.create_subscription(
                String,
                f'/robo_hockey/robot{robot_id}/state',
                lambda msg, rid=robot_id: self.controller_state_callback(msg, rid),
                status_qos,
            )

        self.holder_pose_pub = self.create_publisher(
            PoseStamped,
            HOLDER_MOCAP_TOPIC,
            pose_qos,
        )
        self.puck_pose_pub = self.create_publisher(
            PoseStamped,
            PUCK_MOCAP_TOPIC,
            pose_qos,
        )
        self.goal_pose_pub = self.create_publisher(
            PoseStamped,
            GOAL_MOCAP_TOPIC,
            pose_qos,
        )
        self.create_subscription(Empty, '/sim/reset', self.reset_callback, status_qos)

        self.figure = None
        self.axes = None
        self.robot_patches = {}
        self.gripper_patches = {}
        self.nav_point_patches = {}
        self.nav_point_lines = {}
        self.robot_labels = {}
        self.attached_stick_lines = {}
        self.slot_patches = {}
        self.slot_labels = {}
        self.obstacle_patches = {}
        self.obstacle_safety_patches = {}
        self.puck_patch = None
        self.pass_target_patch = None
        self.pass_target_line = None

        self.reset_world()
        if not HEADLESS:
            self.initialize_plot()
            self.update_plot()

        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(
            f'Simulator started for robots: {self.ROBOT_IDS}, seed={seed}, '
            f'headless={HEADLESS}'
        )

    def random_robot_state(self, existing_positions):
        # Random starts make the controller solve a different navigation
        # problem each run while keeping robots out of unsafe initial states.
        fixed_objects = [
            np.array([SIM_HOLDER_X, SIM_HOLDER_Y]),
            np.array([SIM_PUCK_X, SIM_PUCK_Y]),
            np.array([SIM_GOAL_X, SIM_GOAL_Y]),
        ]
        for _ in range(1000):
            position = np.array([
                self.rng.uniform(
                    self.ENV[0] + 0.35,
                    self.ENV[0] + self.ENV[2] - 0.35,
                ),
                self.rng.uniform(
                    self.ENV[1] + 0.35,
                    self.ENV[1] + self.ENV[3] - 0.35,
                ),
            ])
            theta = self.rng.uniform(-np.pi, np.pi)
            # The CLF-CBF controller acts on this point, so it must also start
            # outside every CBF safety circle.
            nav_point = position + NAV_LOOKAHEAD_DIST * np.array([
                np.cos(theta),
                np.sin(theta),
            ])
            separated_from_objects = all(
                np.linalg.norm(position - obstacle) > 0.65
                for obstacle in fixed_objects
            )
            inside_navigation_safe_set = all(
                np.linalg.norm(nav_point - obstacle['center'])
                > obstacle['radius'] + NAVIGATION_ENVELOPE_RADIUS + 0.05
                for obstacle in SIM_OBSTACLES
            )
            separated_from_robots = all(
                np.linalg.norm(position - other) > 0.75
                for other in existing_positions
            )
            if (
                separated_from_objects
                and inside_navigation_safe_set
                and separated_from_robots
            ):
                return np.array([
                    position[0],
                    position[1],
                    theta,
                ])
        raise RuntimeError('Could not generate separated robot positions')

    def reset_world(self):
        # Reset all dynamic state. The configured holder, puck, goal, and
        # obstacles remain fixed while the robot poses are randomized.
        existing_positions = []
        for robot_id in self.ROBOT_IDS:
            state = self.random_robot_state(existing_positions)
            self.states[robot_id] = state
            existing_positions.append(state[:2].copy())
            self.velocities[robot_id] = np.zeros(3)
            self.last_cmd_time[robot_id] = self.get_clock().now()
            self.leds[robot_id] = np.zeros(3)
            self.controller_states[robot_id] = 'wait_for_pickup_turn'
            self.sticks_attached[robot_id] = False
            self.previous_stick_tips[robot_id] = None
            self.last_impact_time[robot_id] = -np.inf
            self.minimum_swing_tip_distance[robot_id] = np.inf

        self.puck_position = np.array([SIM_PUCK_X, SIM_PUCK_Y], dtype=float)
        self.puck_velocity = np.zeros(2)

    ########################################################################
    # ROS subscription callbacks
    ########################################################################

    def reset_callback(self, _msg):
        self.reset_world()
        if not HEADLESS and self.figure is not None:
            self.update_plot()
        self.get_logger().info('Simulator state reset')

    def velocity_callback(self, msg, robot_id):
        # Twist.linear.x is forward speed in the robot body frame. Convert it
        # to world-frame x/y components before integrating the state.
        linear_x = float(np.clip(
            msg.linear.x,
            -self.MAX_LINEAR_SPEED,
            self.MAX_LINEAR_SPEED,
        ))
        angular_z = float(np.clip(
            msg.angular.z,
            -self.MAX_ANGULAR_SPEED,
            self.MAX_ANGULAR_SPEED,
        ))
        theta = self.states[robot_id][2]
        self.velocities[robot_id] = np.array([
            cos(theta) * linear_x,
            sin(theta) * linear_x,
            angular_z,
        ])
        self.last_cmd_time[robot_id] = self.get_clock().now()

    def led_callback(self, msg, robot_id):
        self.leds[robot_id] = np.clip(
            np.array([msg.r, msg.g, msg.b]),
            0.0,
            1.0,
        )

    def controller_state_callback(self, msg, robot_id):
        # The visual simulator uses controller states to decide when a stick is
        # attached and when tip-puck collision detection should be active.
        previous_state = self.controller_states[robot_id]
        self.controller_states[robot_id] = msg.data
        not_attached_states = {
            'wait_for_pickup_turn',
            'open_gripper',
            'arm_to_pickup_height',
            'navigate_to_pickup_alignment',
            'turn_toward_pickup_alignment',
            'drive_to_pickup_alignment',
            'align_for_pickup',
            'drive_to_pregrasp',
            'insert_gripper',
            'close_gripper',
        }
        if msg.data in not_attached_states:
            self.sticks_attached[robot_id] = False
        elif msg.data == 'lift_stick':
            if not self.sticks_attached[robot_id]:
                self.get_logger().info(f'Robot {robot_id} stick attached')
            self.sticks_attached[robot_id] = True

        if msg.data == 'swing' and previous_state != 'swing':
            self.minimum_swing_tip_distance[robot_id] = np.inf
        elif previous_state == 'swing' and msg.data != 'swing':
            distance = self.minimum_swing_tip_distance[robot_id]
            self.get_logger().info(
                f'Robot {robot_id} swing minimum tip distance: {distance:.3f} m'
            )

    ########################################################################
    # end ROS subscription callbacks
    ########################################################################

    def slot_position(self, robot_id):
        # Convert the selected holder-body slot offset into world coordinates.
        holder_position = np.array([SIM_HOLDER_X, SIM_HOLDER_Y])
        return holder_position + rotate_2d(
            STICK_SLOT_OFFSETS[robot_id],
            SIM_HOLDER_THETA,
        )

    def nav_point(self, robot_id):
        # Approximate-linearization point p = base + l*[cos(theta), sin(theta)].
        state = self.states[robot_id]
        return state[:2] + NAV_LOOKAHEAD_DIST * np.array([
            cos(state[2]),
            sin(state[2]),
        ])

    def stick_tip(self, robot_id):
        # Physical stick-tip location used only for drawing and puck impacts.
        state = self.states[robot_id]
        stick_heading = state[2] + STICK_HEADING_OFFSET_FROM_ROBOT
        return state[:2] + STICK_TIP_FROM_BASE * np.array([
            cos(stick_heading),
            sin(stick_heading),
        ])

    def receiving_point(self):
        # The desired pass point is fixed in Robot 2's body frame, so it moves
        # and rotates with Robot 2.
        state = self.states[ROBOT2_ID]
        return state[:2] + rotate_2d(
            SCORING_RECEIVE_OFFSET_BODY,
            state[2],
        )

    def publish_pose(self, publisher, position, theta, noisy=True):
        # Publish a VRPN-like PoseStamped. The controller sees this noisy pose,
        # while the simulator continues integrating the noise-free true state.
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        if noisy:
            position_noise = self.rng.normal(0.0, MOCAP_POSITION_NOISE_STD, 2)
            yaw_noise = self.rng.normal(0.0, MOCAP_YAW_NOISE_STD)
        else:
            position_noise = np.zeros(2)
            yaw_noise = 0.0

        msg.pose.position.x = float(position[0] + position_noise[0])
        msg.pose.position.y = float(position[1] + position_noise[1])
        msg.pose.position.z = 0.0
        # A planar yaw quaternion has x=y=0, z=sin(theta/2), w=cos(theta/2).
        half_yaw = (theta + yaw_noise) / 2.0
        msg.pose.orientation.z = sin(half_yaw)
        msg.pose.orientation.w = cos(half_yaw)
        publisher.publish(msg)

    @staticmethod
    def point_to_segment_distance(point, segment_start, segment_end):
        # Project the point onto the infinite line, clamp the projection to the
        # segment, then measure the distance to that closest point.
        segment = segment_end - segment_start
        length_squared = float(segment @ segment)
        if length_squared <= 1e-12:
            return np.linalg.norm(point - segment_start)
        fraction = float(
            (point - segment_start) @ segment
            / length_squared
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = segment_start + fraction * segment
        return np.linalg.norm(point - closest)

    def apply_stick_impacts(self):
        # Test the entire line swept by the tip during this time step. A point
        # test could miss the puck when a fast tip jumps across it in one frame.
        now = self.get_clock().now().nanoseconds / 1e9
        for robot_id in self.ROBOT_IDS:
            current_tip = self.stick_tip(robot_id)
            previous_tip = self.previous_stick_tips[robot_id]
            self.previous_stick_tips[robot_id] = current_tip.copy()

            if self.controller_states[robot_id] == 'swing':
                tip_distance = np.linalg.norm(current_tip - self.puck_position)
                self.minimum_swing_tip_distance[robot_id] = min(
                    self.minimum_swing_tip_distance[robot_id],
                    tip_distance,
                )

            if (
                not self.sticks_attached[robot_id]
                or self.controller_states[robot_id] != 'swing'
                or previous_tip is None
            ):
                continue
            if now - self.last_impact_time[robot_id] < IMPACT_COOLDOWN:
                continue

            swept_distance = self.point_to_segment_distance(
                self.puck_position,
                previous_tip,
                current_tip,
            )
            if swept_distance > PUCK_RADIUS + STICK_TIP_RADIUS:
                continue

            tip_velocity = (current_tip - previous_tip) / self.DT
            tip_speed = np.linalg.norm(tip_velocity)
            if tip_speed < 0.05:
                continue

            # Only the relative velocity along the impact direction contributes
            # to the one-dimensional collision impulse.
            impact_direction = tip_velocity / tip_speed
            relative_speed = float(
                (tip_velocity - self.puck_velocity) @ impact_direction
            )
            if relative_speed <= 0.0:
                continue

            # Restitution impulse for two masses:
            # J = (1+e)*v_relative / (1/m_puck + 1/m_stick).
            impulse = (
                (1.0 + STICK_PUCK_RESTITUTION)
                * relative_speed
                / (1.0 / PUCK_MASS + 1.0 / STICK_EFFECTIVE_MASS)
            )
            self.puck_velocity += impulse / PUCK_MASS * impact_direction
            self.last_impact_time[robot_id] = now
            self.get_logger().info(
                f'Robot {robot_id} hit puck: tip_speed={tip_speed:.2f} m/s, '
                f'impulse={impulse:.2f} N s, '
                f'puck_speed={np.linalg.norm(self.puck_velocity):.2f} m/s'
            )

    def update_puck(self):
        # Friction and drag reduce speed but do not change travel direction.
        speed = np.linalg.norm(self.puck_velocity)
        if speed > 0.0:
            deceleration = (
                PUCK_FRICTION_COEFFICIENT * 9.81
                + PUCK_LINEAR_DRAG * speed / PUCK_MASS
                + PUCK_QUADRATIC_DRAG * speed * speed / PUCK_MASS
            )
            new_speed = max(0.0, speed - deceleration * self.DT)
            self.puck_velocity *= new_speed / speed

        # Forward Euler position update.
        self.puck_position += self.puck_velocity * self.DT

        x_min = self.ENV[0] + PUCK_RADIUS
        x_max = self.ENV[0] + self.ENV[2] - PUCK_RADIUS
        y_min = self.ENV[1] + PUCK_RADIUS
        y_max = self.ENV[1] + self.ENV[3] - PUCK_RADIUS
        # Stop the velocity component normal to a wall instead of simulating a
        # bounce, since wall impacts are not part of the project controller.
        if not x_min <= self.puck_position[0] <= x_max:
            self.puck_position[0] = np.clip(self.puck_position[0], x_min, x_max)
            self.puck_velocity[0] = 0.0
        if not y_min <= self.puck_position[1] <= y_max:
            self.puck_position[1] = np.clip(self.puck_position[1], y_min, y_max)
            self.puck_velocity[1] = 0.0

    ########################################################################
    # simulation timer callback
    ########################################################################

    def update_and_publish(self):
        # This timer performs one simulation step and then publishes mocap.
        current_time = self.get_clock().now()
        for robot_id in self.ROBOT_IDS:
            time_since_command = (
                current_time - self.last_cmd_time[robot_id]
            ).nanoseconds / 1e9
            if time_since_command > self.TIMEOUT_CHASSIS_SPEED / 1000.0:
                # Match the real driver's command timeout: an old command must
                # not continue moving the robot forever.
                command = np.zeros(3)
            else:
                command = self.velocities[robot_id]

            # Process noise changes the true simulated state and is applied
            # only while moving. Stationary robots therefore do not drift.
            moving = np.linalg.norm(command) > 1e-6
            position_noise = (
                self.rng.normal(0.0, PROCESS_POSITION_NOISE_STD, 2)
                if moving else np.zeros(2)
            )
            yaw_noise = (
                self.rng.normal(0.0, PROCESS_YAW_NOISE_STD)
                if moving else 0.0
            )
            # Forward Euler integration of world velocity and yaw rate.
            self.states[robot_id][:2] += (
                command[:2] * self.DT
                + position_noise
            )
            self.states[robot_id][2] += command[2] * self.DT + yaw_noise

            # Keep yaw in [-pi, pi] so it cannot grow without bound.
            self.states[robot_id][2] = np.arctan2(
                np.sin(self.states[robot_id][2]),
                np.cos(self.states[robot_id][2]),
            )

        # Impacts use the updated stick locations, then the puck advances.
        self.apply_stick_impacts()
        self.update_puck()

        for robot_id in self.ROBOT_IDS:
            self.publish_pose(
                self.robot_pose_pubs[robot_id],
                self.states[robot_id][:2],
                self.states[robot_id][2],
            )
        self.publish_pose(
            self.holder_pose_pub,
            np.array([SIM_HOLDER_X, SIM_HOLDER_Y]),
            SIM_HOLDER_THETA,
        )
        self.publish_pose(self.puck_pose_pub, self.puck_position, 0.0)
        self.publish_pose(
            self.goal_pose_pub,
            np.array([SIM_GOAL_X, SIM_GOAL_Y]),
            0.0,
        )

        if not HEADLESS:
            self.update_plot()

    ########################################################################
    # end simulation timer callback
    ########################################################################

    def initialize_plot(self):
        # The plot is only a visualization; controller calculations use the
        # ROS pose messages published above.
        self.figure, self.axes = plt.subplots()
        environment = patches.Rectangle(
            (self.ENV[0], self.ENV[1]),
            self.ENV[2],
            self.ENV[3],
            edgecolor='black',
            fill=False,
            linewidth=4,
        )
        self.axes.add_patch(environment)

        for obstacle in SIM_OBSTACLES:
            obstacle_patch = patches.Circle(
                obstacle['center'],
                radius=obstacle['radius'],
                facecolor='tab:red',
                edgecolor='black',
                linewidth=1.5,
                zorder=3,
            )
            safety_patch = patches.Circle(
                obstacle['center'],
                radius=obstacle['radius'] + NAVIGATION_ENVELOPE_RADIUS,
                facecolor='none',
                edgecolor='tab:red',
                linestyle='--',
                linewidth=1.2,
                zorder=2,
            )
            self.obstacle_patches[obstacle['name']] = obstacle_patch
            self.obstacle_safety_patches[obstacle['name']] = safety_patch
            self.axes.add_patch(safety_patch)
            self.axes.add_patch(obstacle_patch)
            self.axes.text(
                obstacle['center'][0],
                obstacle['center'][1] + obstacle['radius'] + 0.08,
                obstacle['name'],
                color='tab:red',
                ha='center',
                fontsize=9,
            )

        # Construct the holder polygon in its local frame, then rotate every
        # corner into the world frame.
        holder_width = 0.14
        holder_length = 0.48
        holder_corners = np.array([
            [-holder_width / 2.0, -holder_length / 2.0],
            [holder_width / 2.0, -holder_length / 2.0],
            [holder_width / 2.0, holder_length / 2.0],
            [-holder_width / 2.0, holder_length / 2.0],
        ])
        rotated_holder = np.array([SIM_HOLDER_X, SIM_HOLDER_Y]) + np.array([
            rotate_2d(corner, SIM_HOLDER_THETA) for corner in holder_corners
        ])
        holder_patch = patches.Polygon(
            rotated_holder,
            facecolor='lightgray',
            edgecolor='saddlebrown',
            linewidth=2.0,
        )
        self.axes.add_patch(holder_patch)
        self.axes.text(
            SIM_HOLDER_X,
            SIM_HOLDER_Y + 0.34,
            'stick holder',
            color='saddlebrown',
            ha='center',
        )

        for robot_id in self.ROBOT_IDS:
            slot = self.slot_position(robot_id)
            slot_patch = patches.Circle(
                slot,
                radius=0.045,
                facecolor='saddlebrown',
                edgecolor='black',
                linewidth=1.0,
                zorder=5,
            )
            slot_label = self.axes.text(
                slot[0],
                slot[1] + 0.09,
                f'stick {robot_id}',
                color='saddlebrown',
                ha='center',
                fontsize=9,
            )
            self.slot_patches[robot_id] = slot_patch
            self.slot_labels[robot_id] = slot_label
            self.axes.add_patch(slot_patch)

            state = self.states[robot_id]
            robot_patch = patches.Polygon(
                self.robot_polygon(state),
                facecolor='black',
            )
            gripper_patch = patches.Polygon(
                self.gripper_polygon(state),
                facecolor='black',
            )
            nav_point = self.nav_point(robot_id)
            nav_patch = patches.Circle(
                nav_point,
                radius=0.035,
                facecolor='tab:orange',
                edgecolor='black',
                linewidth=1.0,
                zorder=5,
            )
            nav_line, = self.axes.plot(
                [state[0], nav_point[0]],
                [state[1], nav_point[1]],
                color='tab:orange',
                linestyle='--',
                linewidth=1.2,
            )
            stick_line, = self.axes.plot(
                [],
                [],
                color='saddlebrown',
                linewidth=5.0,
                solid_capstyle='round',
                zorder=4,
            )
            label = self.axes.text(
                state[0] + 0.20,
                state[1] + 0.20,
                str(robot_id),
                color='red',
            )

            self.robot_patches[robot_id] = robot_patch
            self.gripper_patches[robot_id] = gripper_patch
            self.nav_point_patches[robot_id] = nav_patch
            self.nav_point_lines[robot_id] = nav_line
            self.attached_stick_lines[robot_id] = stick_line
            self.robot_labels[robot_id] = label
            self.axes.add_patch(robot_patch)
            self.axes.add_patch(gripper_patch)
            self.axes.add_patch(nav_patch)

        self.puck_patch = patches.Circle(
            self.puck_position,
            radius=PUCK_RADIUS,
            facecolor='tab:blue',
            edgecolor='black',
            linewidth=1.0,
            zorder=5,
        )
        self.axes.add_patch(self.puck_patch)
        self.axes.text(
            SIM_PUCK_X,
            SIM_PUCK_Y + 0.14,
            'puck',
            color='tab:blue',
            ha='center',
        )

        goal_width = 0.70
        goal_patch = patches.Rectangle(
            (SIM_GOAL_X - goal_width / 2.0, SIM_GOAL_Y - 0.04),
            goal_width,
            0.08,
            facecolor='none',
            edgecolor='tab:green',
            linewidth=3.0,
        )
        self.axes.add_patch(goal_patch)
        self.axes.text(
            SIM_GOAL_X,
            SIM_GOAL_Y + 0.13,
            'goal',
            color='tab:green',
            ha='center',
        )

        pass_target = self.receiving_point()
        self.pass_target_patch = patches.Circle(
            pass_target,
            radius=0.055,
            facecolor='none',
            edgecolor='tab:purple',
            linewidth=2.0,
            zorder=4,
        )
        self.axes.add_patch(self.pass_target_patch)
        self.pass_target_line, = self.axes.plot(
            [self.puck_position[0], pass_target[0]],
            [self.puck_position[1], pass_target[1]],
            color='tab:purple',
            linestyle=':',
            linewidth=1.2,
        )

        margin = max(self.ROBOT_SIZE)
        self.axes.set_xlim(
            self.ENV[0] - margin,
            self.ENV[0] + self.ENV[2] + margin,
        )
        self.axes.set_ylim(
            self.ENV[1] - margin,
            self.ENV[1] + self.ENV[3] + margin,
        )
        self.axes.grid()
        self.axes.axis('equal')
        plt.ion()
        plt.show()

    def robot_polygon(self, state):
        # Define body corners in the robot frame, rotate them by yaw, and then
        # translate them by the robot's world position.
        rotation = np.array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]), cos(state[2])],
        ])
        body = np.array([
            [self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
            [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
            [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
            [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
        ])
        return state[:2] + body @ rotation.T

    def gripper_polygon(self, state):
        # The same body-to-world transform is used for the gripper outline.
        rotation = np.array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]), cos(state[2])],
        ])
        front = self.ROBOT_SIZE[1] / 2.0
        gripper = np.array([
            [front, -self.GRIPPER_SIZE / 2.0],
            [front, self.GRIPPER_SIZE / 2.0],
            [front + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
            [front + self.GRIPPER_SIZE, 0.4 * self.GRIPPER_SIZE],
            [front, 0.4 * self.GRIPPER_SIZE],
            [front, -0.4 * self.GRIPPER_SIZE],
            [front + self.GRIPPER_SIZE, -0.4 * self.GRIPPER_SIZE],
            [front + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
        ])
        return state[:2] + gripper @ rotation.T

    def update_plot(self):
        # Move existing Matplotlib artists rather than recreating them each step.
        for robot_id in self.ROBOT_IDS:
            state = self.states[robot_id]
            self.robot_patches[robot_id].xy = self.robot_polygon(state)
            self.gripper_patches[robot_id].xy = self.gripper_polygon(state)

            led = self.leds[robot_id]
            if np.linalg.norm(led) < 1e-6:
                led = np.zeros(3)
            self.robot_patches[robot_id].set_facecolor(led)

            nav_point = self.nav_point(robot_id)
            self.nav_point_patches[robot_id].center = nav_point
            self.nav_point_lines[robot_id].set_data(
                [state[0], nav_point[0]],
                [state[1], nav_point[1]],
            )
            self.robot_labels[robot_id].set_position((
                state[0] + 0.20,
                state[1] + 0.20,
            ))

            if self.sticks_attached[robot_id]:
                stick_heading = state[2] + STICK_HEADING_OFFSET_FROM_ROBOT
                stick_start = state[:2] + BASE_TO_GRIPPER * np.array([
                    cos(stick_heading),
                    sin(stick_heading),
                ])
                stick_end = self.stick_tip(robot_id)
                self.attached_stick_lines[robot_id].set_data(
                    [stick_start[0], stick_end[0]],
                    [stick_start[1], stick_end[1]],
                )
            else:
                self.attached_stick_lines[robot_id].set_data([], [])

            stick_is_attached = self.sticks_attached[robot_id]
            self.slot_patches[robot_id].set_visible(not stick_is_attached)
            self.slot_labels[robot_id].set_visible(not stick_is_attached)

        self.puck_patch.center = self.puck_position
        pass_target = self.receiving_point()
        self.pass_target_patch.center = pass_target
        self.pass_target_line.set_data(
            [self.puck_position[0], pass_target[0]],
            [self.puck_position[1], pass_target[1]],
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = MultiRoboMasterSim()
    try:
        # rclpy.spin runs subscriptions and the simulation timer until Ctrl+C.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
