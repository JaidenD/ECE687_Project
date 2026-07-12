import numpy as np
from geometry_msgs.msg import Twist

from robo_hockey_controller.config import (
    ARM_CARRY_X,
    ARM_CARRY_Z,
    ARM_PLAY_X,
    ARM_PLAY_Z,
    AUTO_RUN_SCORING_SWING,
    GOAL_TARGET_OFFSET_WORLD,
    IMPACT_CONTACT_TIME,
    NAV_LOOKAHEAD_DIST,
    NAV_POINT_TOL,
    PASS_SETTLE_TIME,
    PICKUP_APPROACH_SPEED,
    PICKUP_INSERT_SPEED,
    PICKUP_LATERAL_TOL,
    PICKUP_MAX_ANGULAR_SPEED,
    PUCK_MASS,
    PUCK_RADIUS,
    PUCK_RECEIVE_RADIUS,
    PUCK_STOP_HOLD_TIME,
    PUCK_STOP_SPEED_TOL,
    PUCK_TARGET_ARRIVAL_SPEED,
    PUCK_FRICTION_COEFFICIENT,
    PUCK_LINEAR_DRAG,
    PUCK_QUADRATIC_DRAG,
    SCORING_RECEIVE_OFFSET_BODY,
    STICK_EFFECTIVE_MASS,
    STICK_HEADING_OFFSET_FROM_ROBOT,
    STICK_PUCK_RESTITUTION,
    STICK_TIP_FROM_BASE,
    SWING_ANGLE_BEFORE_IMPACT,
    SWING_FOLLOW_THROUGH_ANGLE,
    SWING_MAX_ANGULAR_SPEED,
    SWING_SETUP_APPROACH_DISTANCE,
    SWING_SETUP_HEADING_TOL,
    SWING_SETUP_POSITION_TOL,
    SWING_SETTLE_TIME,
    SWING_SIGN_BY_ROBOT,
)
from robo_hockey_controller.helpers import (
    required_puck_launch_speed,
    rotate_2d,
    stick_heading_for_impact,
    wrap,
    yaw_from_pose,
)


class HockeyStateHandlers:
    """State handlers for planning and executing the pass or shot."""

    def hockey_state_handlers(self):
        # After pickup, the passing robot plans immediately. The scoring robot
        # first waits for the puck to arrive. Both then use the same sequence:
        # plan -> QP navigation -> precise setup -> lower arm -> swing.
        return {
            'wait_for_both_pickups': self.state_wait_for_both_pickups,
            'wait_for_pass': self.state_wait_for_pass,
            'raise_arm_for_navigation': self.state_raise_arm_for_navigation,
            'plan_swing': self.state_plan_swing,
            'navigate_to_swing': self.state_navigate_to_swing,
            'turn_toward_swing_standoff': self.state_turn_toward_swing_standoff,
            'drive_to_swing_standoff': self.state_drive_to_swing_standoff,
            'align_for_swing': self.state_align_for_swing,
            'drive_to_swing_base': self.state_drive_to_swing_base,
            'lower_arm_for_swing': self.state_lower_arm_for_swing,
            'wait_before_swing': self.state_wait_before_swing,
            'swing': self.state_swing,
        }

    def receiving_point(self, scoring_robot_pose):
        # SCORING_RECEIVE_OFFSET_BODY is measured in Robot 2's body frame.
        # Rotate it by Robot 2's yaw and add Robot 2's world position:
        #   r_receive = r_robot2 + R(theta_robot2) * offset_body.
        scoring_position = np.array([
            scoring_robot_pose.pose.position.x,
            scoring_robot_pose.pose.position.y,
        ])
        scoring_theta = yaw_from_pose(scoring_robot_pose)

        receive_offset_world = rotate_2d(
            SCORING_RECEIVE_OFFSET_BODY,
            scoring_theta,
        )
        return scoring_position + receive_offset_world

    def target_for_swing(self):
        # Robot 1 passes toward a live point attached to Robot 2.
        if self.role == 'pass':
            if not self.wait_for_pose(
                'scoring robot',
                self.other_robot_pose,
                self.other_robot_pose_time,
            ):
                return None
            return self.receiving_point(self.other_robot_pose)

        # Robot 2 shoots toward the measured goal plus any chosen world offset.
        if not self.wait_for_pose('goal', self.goal_pose, self.goal_pose_time):
            return None
        return np.array([
            self.goal_pose.pose.position.x,
            self.goal_pose.pose.position.y,
        ]) + GOAL_TARGET_OFFSET_WORLD

    def build_swing_plan(self):
        # A swing plan is a snapshot. It is rebuilt from the latest puck and
        # target poses before navigation begins.
        if not self.wait_for_pose('puck', self.puck_pose, self.puck_pose_time):
            return False

        puck = np.array([
            self.puck_pose.pose.position.x,
            self.puck_pose.pose.position.y,
        ])

        target = self.target_for_swing()
        if target is None:
            return False

        # Unit vector n points in the desired puck-travel direction.
        target_delta = target - puck
        travel_distance = np.linalg.norm(target_delta)
        if travel_distance < 0.10:
            self.get_logger().warning('Puck target is too close to define a swing')
            return False
        impact_direction = target_delta / travel_distance
        swing_sign = SWING_SIGN_BY_ROBOT[self.robot_id]

        # A rotating tip moves perpendicular to the stick. swing_sign selects
        # the clockwise or counterclockwise solution that produces velocity n.
        stick_heading_at_impact = stick_heading_for_impact(
            impact_direction,
            swing_sign,
        )
        robot_heading_at_impact = wrap(
            stick_heading_at_impact
            - STICK_HEADING_OFFSET_FROM_ROBOT
        )

        # Start on the opposite side of the impact heading, then rotate through
        # the puck and continue by SWING_FOLLOW_THROUGH_ANGLE.
        robot_heading_at_start = wrap(
            robot_heading_at_impact
            - swing_sign * SWING_ANGLE_BEFORE_IMPACT
        )

        # Contact the rear edge of the puck so the impulse points along n.
        #   r_contact = r_puck - R_puck*n.
        contact_point = puck - PUCK_RADIUS * impact_direction
        stick_direction_at_impact = np.array([
            np.cos(stick_heading_at_impact),
            np.sin(stick_heading_at_impact),
        ])
        # Place the base so its stick tip reaches the contact point at impact:
        #   r_base + L_stick*t_stick = r_contact.
        robot_position = (
            contact_point
            - STICK_TIP_FROM_BASE * stick_direction_at_impact
        )
        standoff_direction = np.array([
            np.cos(robot_heading_at_start),
            np.sin(robot_heading_at_start),
        ])

        # Work backward through the friction/drag model to find the puck speed
        # required immediately after impact.
        puck_launch_speed = required_puck_launch_speed(
            travel_distance,
            PUCK_TARGET_ARRIVAL_SPEED,
            PUCK_MASS,
            PUCK_FRICTION_COEFFICIENT,
            PUCK_LINEAR_DRAG,
            PUCK_QUADRATIC_DRAG,
        )

        # One-dimensional impact model for a stationary puck:
        #   v_puck+ = (1+e)m_stick/(m_stick+m_puck) * v_tip-.
        # Rearrange it to find the required incoming stick-tip speed.
        impact_denominator = (
            (1.0 + STICK_PUCK_RESTITUTION)
            * STICK_EFFECTIVE_MASS
        )
        tip_speed = (
            puck_launch_speed
            * (STICK_EFFECTIVE_MASS + PUCK_MASS)
            / impact_denominator
        )

        # Tangential speed is v_tip = omega*L. Store the requested value and
        # the value that the robot can actually command.
        requested_omega = tip_speed / STICK_TIP_FROM_BASE
        commanded_omega = min(requested_omega, SWING_MAX_ANGULAR_SPEED)

        # From impulse J = Delta(momentum) = F_average*Delta(t). This is only
        # an estimate for logging; the robot is not doing force control.
        average_force = (
            PUCK_MASS
            * puck_launch_speed
            / IMPACT_CONTACT_TIME
        )

        # Save one plan so every following state uses the same geometry.
        self.swing_plan = {
            'puck': puck,
            'target': target,
            'direction': impact_direction,
            'base_position': robot_position,
            'standoff_position': (
                robot_position
                - SWING_SETUP_APPROACH_DISTANCE * standoff_direction
            ),
            'start_heading': robot_heading_at_start,
            'impact_heading': robot_heading_at_impact,
            'swing_sign': swing_sign,
            'requested_omega': requested_omega,
            'commanded_omega': commanded_omega,
            'puck_launch_speed': puck_launch_speed,
            'tip_speed': tip_speed,
            'average_force': average_force,
        }

        # Print enough information to compare the requested and limited swing.
        limit_text = ''
        if requested_omega > SWING_MAX_ANGULAR_SPEED:
            limit_text = ' (limited by SWING_MAX_ANGULAR_SPEED)'
        self.get_logger().info(
            'Swing plan: '
            f'target=({target[0]:.2f}, {target[1]:.2f}), '
            f'distance={travel_distance:.2f} m, '
            f'puck_speed={puck_launch_speed:.2f} m/s, '
            f'tip_speed={tip_speed:.2f} m/s, '
            f'omega={commanded_omega:.2f} rad/s{limit_text}, '
            f'estimated_average_force={average_force:.1f} N'
        )
        return True

    ########################################################################
    # Hockey state handlers
    ########################################################################

    def state_wait_for_both_pickups(self, _pose):
        self.stop()
        if self.other_pickup_clear:
            self.go_to_state('raise_arm_for_navigation')

    def state_wait_for_pass(self, _pose):
        # Robot 2 requires the puck to remain slow inside the receiving region
        # for PUCK_STOP_HOLD_TIME. This avoids reacting to one noisy sample.
        self.stop()
        if not AUTO_RUN_SCORING_SWING or not self.pass_complete:
            return
        if self.pass_complete_time is None:
            return
        if self.now_seconds() - self.pass_complete_time < PASS_SETTLE_TIME:
            return
        if not self.wait_for_pose('puck', self.puck_pose, self.puck_pose_time):
            return

        receive_point = self.receiving_point(self.robot_pose)
        puck_position = np.array([
            self.puck_pose.pose.position.x,
            self.puck_pose.pose.position.y,
        ])
        puck_distance = np.linalg.norm(puck_position - receive_point)
        puck_speed = np.linalg.norm(self.puck_velocity)
        if (
            puck_distance < PUCK_RECEIVE_RADIUS
            and puck_speed < PUCK_STOP_SPEED_TOL
        ):
            if self.puck_slow_since is None:
                self.puck_slow_since = self.now_seconds()
        else:
            self.puck_slow_since = None

        puck_has_stopped = (
            self.puck_slow_since is not None
            and self.now_seconds() - self.puck_slow_since >= PUCK_STOP_HOLD_TIME
        )
        if puck_has_stopped:
            self.go_to_state('raise_arm_for_navigation')

    def state_raise_arm_for_navigation(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_arm_action(ARM_CARRY_X, ARM_CARRY_Z),
            poll_action=self.poll_arm_action,
            next_state='plan_swing',
            action_name='Raise arm for navigation',
        )

    def state_plan_swing(self, _pose):
        self.stop()
        if self.build_swing_plan():
            self.go_to_state('navigate_to_swing')

    def state_navigate_to_swing(self, pose):
        if not self.wait_for_pose(
            'other robot',
            self.other_robot_pose,
            self.other_robot_pose_time,
        ):
            return

        start_heading = self.swing_plan['start_heading']
        start_direction = np.array([
            np.cos(start_heading),
            np.sin(start_heading),
        ])
        # As in pickup navigation, shift the QP target because the QP controls
        # the navigation point rather than the robot base origin.
        target_point = (
            self.swing_plan['standoff_position']
            + NAV_LOOKAHEAD_DIST * start_direction
        )
        x, y, theta = pose
        cmd, point_error, qp_result = self.safe_navigation_controller(
            x,
            y,
            theta,
            target_point,
            include_other_robot=True,
        )
        self.publish_motion(
            'swing navigation',
            pose,
            f'point_error={point_error:.3f}, {self.qp_debug_text(qp_result)}',
            cmd,
        )
        if qp_result is not None and point_error < NAV_POINT_TOL:
            self.stop_and_go_to_state('turn_toward_swing_standoff')

    def state_turn_toward_swing_standoff(self, pose):
        x, y, theta = pose
        position_error = (
            self.swing_plan['standoff_position'] - np.array([x, y])
        )
        desired_heading = np.arctan2(
            position_error[1],
            position_error[0],
        )
        cmd, heading_error = self.heading_command(
            theta,
            desired_heading,
            PICKUP_MAX_ANGULAR_SPEED,
        )
        self.publish_motion(
            'turn toward swing stand-off',
            pose,
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if abs(heading_error) < SWING_SETUP_HEADING_TOL:
            self.stop_and_go_to_state('drive_to_swing_standoff')

    def state_drive_to_swing_standoff(self, pose):
        x, y, theta = pose
        cmd, position_error, heading_error = self.point_drive_command(
            x,
            y,
            theta,
            self.swing_plan['standoff_position'],
            PICKUP_APPROACH_SPEED,
        )
        self.publish_motion(
            'drive to swing stand-off',
            pose,
            f'position_error={position_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if position_error < SWING_SETUP_POSITION_TOL:
            self.stop_and_go_to_state('align_for_swing')

    def state_align_for_swing(self, pose):
        cmd, heading_error = self.heading_command(
            pose[2],
            self.swing_plan['start_heading'],
            PICKUP_MAX_ANGULAR_SPEED,
        )
        self.publish_motion(
            'swing alignment',
            pose,
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if abs(heading_error) < SWING_SETUP_HEADING_TOL:
            self.stop_and_go_to_state('drive_to_swing_base')

    def state_drive_to_swing_base(self, pose):
        x, y, theta = pose
        cmd, position_error, lateral_error, heading_error = self.straight_command(
            x,
            y,
            theta,
            self.swing_plan['base_position'],
            self.swing_plan['start_heading'],
            PICKUP_INSERT_SPEED,
            allow_reverse=False,
        )
        self.publish_motion(
            'final swing approach',
            pose,
            f'position_error={position_error:.3f}, '
            f'lateral_error={lateral_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        # Lower the arm only after the final base pose is accurate enough.
        swing_ready = (
            position_error < SWING_SETUP_POSITION_TOL
            and abs(lateral_error) < PICKUP_LATERAL_TOL
            and abs(heading_error) < SWING_SETUP_HEADING_TOL
        )
        if swing_ready:
            self.stop_and_go_to_state('lower_arm_for_swing')

    def state_lower_arm_for_swing(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_arm_action(ARM_PLAY_X, ARM_PLAY_Z),
            poll_action=self.poll_arm_action,
            next_state='wait_before_swing',
            action_name='Lower arm for swing',
        )

    def state_wait_before_swing(self, _pose):
        self.stop()
        elapsed = self.get_clock().now() - self.state_started
        elapsed_seconds = elapsed.nanoseconds / 1e9
        if elapsed_seconds >= SWING_SETTLE_TIME:
            self.go_to_state('swing')

    def state_swing(self, pose):
        theta = pose[2]
        if self.just_entered_state:
            self.just_entered_state = False
            self.swing_start_heading = theta

        # Multiplying by swing_sign makes progress positive for either a
        # clockwise or counterclockwise swing.
        swing_sign = self.swing_plan['swing_sign']
        progress = swing_sign * wrap(theta - self.swing_start_heading)
        total_angle = SWING_ANGLE_BEFORE_IMPACT + SWING_FOLLOW_THROUGH_ANGLE
        if progress >= total_angle:
            self.stop()
            if self.role == 'pass':
                self.pass_complete = True
                self.go_to_state('pass_complete')
            else:
                self.go_to_state('done')
            return

        cmd = Twist()
        cmd.angular.z = float(
            swing_sign * self.swing_plan['commanded_omega']
        )
        self.publish_motion(
            'swing',
            pose,
            f'progress={progress:.3f}/{total_angle:.3f}',
            cmd,
        )
