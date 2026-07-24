import numpy as np
from robomaster_msgs.action import GripperControl

from robo_hockey_controller.config import (
    ARM_LIFT_X,
    ARM_LIFT_Z,
    ARM_PICKUP_X,
    ARM_PICKUP_Z,
    ARM_PLAY_X,
    ARM_PLAY_Z,
    BASE_TO_GRIPPER,
    HOLDER_PLATFORM_RADIUS,
    NAVIGATION_ENVELOPE_RADIUS,
    NAV_LOOKAHEAD_DIST,
    NAV_POINT_TOL,
    PICKUP_ALIGNMENT_DISTANCE,
    PICKUP_APPROACH_SPEED,
    PICKUP_HEADING_TOL,
    PICKUP_INSERT_DISTANCE,
    PICKUP_INSERT_SPEED,
    PICKUP_LATERAL_TOL,
    PICKUP_MAX_ANGULAR_SPEED,
    PICKUP_POSITION_TOL,
    PICKUP_RETREAT_DISTANCES,
    PICKUP_RETREAT_SPEED,
    ROBOT1_ID,
    STICK_SLOT_OFFSETS,
)
from robo_hockey_controller.helpers import rotate_2d, wrap, yaw_from_pose


class PickupStateHandlers:
    """State handlers for picking up a stick and clearing the holder."""

    def pickup_state_handlers(self):
        # The dispatcher in controller.py uses this table to call the function
        # belonging to the current state. The pickup sequence is:
        # wait -> open/lower arm -> QP navigation -> final alignment/insertion
        # -> close/lift -> reverse out -> lower arm.
        return {
            'wait_for_pickup_turn': self.state_wait_for_pickup_turn,
            'open_gripper': self.state_open_gripper,
            'arm_to_pickup_height': self.state_arm_to_pickup_height,
            'navigate_to_pickup_alignment': self.state_navigate_to_pickup_alignment,
            'turn_toward_pickup_alignment': self.state_turn_toward_pickup_alignment,
            'drive_to_pickup_alignment': self.state_drive_to_pickup_alignment,
            'align_for_pickup': self.state_align_for_pickup,
            'drive_to_pregrasp': self.state_drive_to_pregrasp,
            'insert_gripper': self.state_insert_gripper,
            'close_gripper': self.state_close_gripper,
            'lift_stick': self.state_lift_stick,
            'retreat_from_holder': self.state_retreat_from_holder,
            'lower_arm_to_play_height': self.state_lower_arm_to_play_height,
            'pickup_complete': self.state_pickup_complete,
        }

    def holder_slot_geometry(self):
        # The holder mocap pose gives the origin r_h and heading theta_h.
        holder_position = np.array([
            self.holder_pose.pose.position.x,
            self.holder_pose.pose.position.y,
        ])
        holder_theta = 3.14+yaw_from_pose(self.holder_pose)
        # Slot offsets are measured in the holder body frame. Rotate the chosen
        # offset into the world frame before adding the holder position:
        #   r_slot = r_holder + R(theta_holder) * slot_offset.
        slot_position = holder_position + rotate_2d(
            STICK_SLOT_OFFSETS[self.robot_id],
            holder_theta,
        )

        # pickup_direction points from the robot toward the slot.
        # The holder frame's positive x-axis points out of the front. The robot
        # starts on that side and faces back along negative x toward the slot.
        pickup_theta = wrap(holder_theta + np.pi)
        pickup_direction = np.array([
            np.cos(pickup_theta),
            np.sin(pickup_theta),
        ])
        # Work backward from the slot to find the three base positions. At
        # grasp_base, the gripper reaches the slot. pregrasp and alignment add
        # extra straight-line clearance in front of that final position.
        grasp_base = slot_position - BASE_TO_GRIPPER * pickup_direction
        pregrasp = grasp_base - PICKUP_INSERT_DISTANCE * pickup_direction
        alignment = pregrasp - PICKUP_ALIGNMENT_DISTANCE * pickup_direction

        return {
            'slot': slot_position,
            'theta': pickup_theta,
            'direction': pickup_direction,
            'grasp_base': grasp_base,
            'pregrasp': pregrasp,
            'alignment': alignment,
        }

    def current_pickup_geometry(self):
        if not self.wait_for_pose(
            'stick holder',
            self.holder_pose,
            self.holder_pose_time,
        ):
            return None
        return self.holder_slot_geometry()

    ########################################################################
    # Pickup state handlers
    ########################################################################

    def state_wait_for_pickup_turn(self, _pose):
        # Robot 1 goes first. Robot 2 waits until Robot 1 publishes pickup_clear.
        self.stop()
        if self.robot_id == ROBOT1_ID or self.other_pickup_clear:
            self.go_to_state('open_gripper')

    def state_open_gripper(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_gripper_action(
                GripperControl.Goal.OPEN
            ),
            poll_action=self.poll_gripper_action,
            next_state='arm_to_pickup_height',
            action_name='Open gripper',
        )

    def state_arm_to_pickup_height(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_arm_action(
                ARM_PICKUP_X,
                ARM_PICKUP_Z,
            ),
            poll_action=self.poll_arm_action,
            next_state='navigate_to_pickup_alignment',
            action_name='Move arm to pickup height',
        )

    def state_navigate_to_pickup_alignment(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        x, y, theta = pose
        # The QP controls the navigation point, not the base origin. Shift the
        # QP target forward by the same lookahead distance so the base finishes
        # at pickup['alignment'] with the desired heading.
        target_point = (
            pickup['alignment']
            + NAV_LOOKAHEAD_DIST * pickup['direction']
        )

        # The CBF cannot drive to a target inside its own holder safety circle.
        # Detect a bad combination of pickup distances before commanding motion.
        holder_position = np.array([
            self.holder_pose.pose.position.x,
            self.holder_pose.pose.position.y,
        ])
        target_clearance = np.linalg.norm(target_point - holder_position)
        required_clearance = (
            HOLDER_PLATFORM_RADIUS + NAVIGATION_ENVELOPE_RADIUS
        )
        if target_clearance <= required_clearance:
            self.get_logger().error(
                'Pickup target is inside the holder safety envelope: '
                f'target_clearance={target_clearance:.2f} m, '
                f'required_clearance={required_clearance:.2f} m'
            )
            self.stop_and_go_to_state('error')
            return

        cmd, point_error, qp_result = self.safe_navigation_controller(
            x,
            y,
            theta,
            target_point,
            include_other_robot=False,
            include_holder=True,
        )
        self.publish_motion(
            'pickup navigation',
            pose,
            f'point_error={point_error:.3f}, {self.qp_debug_text(qp_result)}',
            cmd,
        )
        if qp_result is not None and point_error < NAV_POINT_TOL:
            self.stop_and_go_to_state('turn_toward_pickup_alignment')

    def state_turn_toward_pickup_alignment(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        x, y, theta = pose
        position_error = pickup['alignment'] - np.array([x, y])
        # First face the stand-off point so the next state can drive forward.
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
            'turn toward pickup stand-off',
            pose,
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if abs(heading_error) < PICKUP_HEADING_TOL:
            self.stop_and_go_to_state('drive_to_pickup_alignment')

    def state_drive_to_pickup_alignment(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        x, y, theta = pose
        cmd, position_error, heading_error = self.point_drive_command(
            x,
            y,
            theta,
            pickup['alignment'],
            PICKUP_APPROACH_SPEED,
        )
        self.publish_motion(
            'drive to pickup stand-off',
            pose,
            f'position_error={position_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if position_error < PICKUP_POSITION_TOL:
            self.stop_and_go_to_state('align_for_pickup')

    def state_align_for_pickup(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        # Rotate in place before entering the narrow straight pickup path.
        cmd, heading_error = self.heading_command(
            pose[2],
            pickup['theta'],
            PICKUP_MAX_ANGULAR_SPEED,
        )
        self.publish_motion(
            'pickup alignment',
            pose,
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if abs(heading_error) < PICKUP_HEADING_TOL:
            self.stop_and_go_to_state('drive_to_pregrasp')

    def state_drive_to_pregrasp(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        x, y, theta = pose
        cmd, position_error, lateral_error, heading_error = self.straight_command(
            x,
            y,
            theta,
            pickup['pregrasp'],
            pickup['theta'],
            PICKUP_APPROACH_SPEED,
            allow_reverse=False,
        )
        self.publish_motion(
            'pregrasp drive',
            pose,
            f'position_error={position_error:.3f}, '
            f'lateral_error={lateral_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        # All three errors must be small before moving closer to the holder.
        pickup_ready = (
            position_error < PICKUP_POSITION_TOL
            and abs(lateral_error) < PICKUP_LATERAL_TOL
            and abs(heading_error) < PICKUP_HEADING_TOL
        )
        if pickup_ready:
            self.stop_and_go_to_state('insert_gripper')

    def state_insert_gripper(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        x, y, theta = pose
        cmd, position_error, lateral_error, heading_error = self.straight_command(
            x,
            y,
            theta,
            pickup['grasp_base'],
            pickup['theta'],
            PICKUP_INSERT_SPEED,
            allow_reverse=False,
        )
        self.publish_motion(
            'straight insertion',
            pose,
            f'position_error={position_error:.3f}, '
            f'lateral_error={lateral_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        # Do not close the gripper until position, line tracking, and heading
        # are all within their calibrated tolerances.
        grasp_ready = (
            position_error < PICKUP_POSITION_TOL
            and abs(lateral_error) < PICKUP_LATERAL_TOL
            and abs(heading_error) < PICKUP_HEADING_TOL
        )
        if grasp_ready:
            self.stop_and_go_to_state('close_gripper')

    def state_close_gripper(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_gripper_action(
                GripperControl.Goal.CLOSE
            ),
            poll_action=self.poll_gripper_action,
            next_state='lift_stick',
            action_name='Close gripper',
        )

    def state_lift_stick(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_arm_action(ARM_LIFT_X, ARM_LIFT_Z),
            poll_action=self.poll_arm_action,
            next_state='retreat_from_holder',
            action_name='Lift stick',
        )

    def state_retreat_from_holder(self, pose):
        pickup = self.current_pickup_geometry()
        if pickup is None:
            return

        # Keep the pickup heading and reverse along the same line. Robot 1 has
        # a longer retreat because it must clear the area for Robot 2.
        retreat_target = (
            pickup['grasp_base']
            - PICKUP_RETREAT_DISTANCES[self.robot_id] * pickup['direction']
        )
        x, y, theta = pose
        cmd, position_error, lateral_error, heading_error = self.straight_command(
            x,
            y,
            theta,
            retreat_target,
            pickup['theta'],
            PICKUP_RETREAT_SPEED,
            allow_reverse=True,
        )
        self.publish_motion(
            'holder retreat',
            pose,
            f'position_error={position_error:.3f}, '
            f'lateral_error={lateral_error:.3f}, '
            f'heading_error={heading_error:.3f}',
            cmd,
        )
        if position_error < PICKUP_POSITION_TOL:
            self.stop_and_go_to_state('lower_arm_to_play_height')

    def state_lower_arm_to_play_height(self, _pose):
        self.run_action_state(
            start_action=lambda: self.start_arm_action(ARM_PLAY_X, ARM_PLAY_Z),
            poll_action=self.poll_arm_action,
            next_state='pickup_complete',
            action_name='Lower arm to play height',
        )

    def state_pickup_complete(self, _pose):
        # This flag releases Robot 2 and lets the hockey sequence begin.
        self.stop()
        self.pickup_clear = True
        if self.role == 'pass':
            self.go_to_state('wait_for_both_pickups')
        else:
            self.go_to_state('wait_for_pass')
