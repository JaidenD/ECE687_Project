import numpy as np

from multi_robomaster_ros_sim.config import NAV_LOOKAHEAD_DIST


def wrap(angle):
    # atan2(sin, cos) gives the equivalent angle in [-pi, pi].
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_pose(msg):
    # Mocap gives orientation as a quaternion. This is the standard quaternion
    # to yaw formula after assuming the robot moves in the horizontal plane.
    q = msg.pose.orientation
    return np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def rotate_2d(vector, angle):
    # Apply the 2D rotation matrix R(angle) to a vector.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        c * vector[0] - s * vector[1],
        s * vector[0] + c * vector[1],
    ])


def required_puck_launch_speed(
    distance,
    arrival_speed,
    puck_mass,
    friction_coefficient,
    linear_drag,
    quadratic_drag,
    steps=1000,
):
    # The puck model is
    #   m dv/dt = -mu*m*g - c1*v - c2*v^2.
    # This function starts at the target and integrates backward over distance
    # to recover the speed needed immediately after impact.
    if distance <= 0.0:
        return max(arrival_speed, 0.0)
    if puck_mass <= 0.0:
        raise ValueError('puck_mass must be positive')

    number_of_steps = max(steps, 1)
    distance_step = distance / number_of_steps

    # Use w = v^2. In the backward-distance direction,
    #   dw/ds = 2*mu*g + 2*c1*sqrt(w)/m + 2*c2*w/m.
    speed_squared = max(arrival_speed, 0.0) ** 2

    for _ in range(number_of_steps):
        speed = np.sqrt(max(speed_squared, 0.0))
        recovered_loss = (
            2.0 * friction_coefficient * 9.81
            + 2.0 * linear_drag * speed / puck_mass
            + 2.0 * quadratic_drag * speed_squared / puck_mass
        )
        speed_squared += distance_step * recovered_loss

    return np.sqrt(max(speed_squared, 0.0))


def stick_heading_for_impact(direction, swing_sign):
    # A rotating stick tip moves perpendicular to the stick. Put the stick
    # 90 degrees clockwise or counterclockwise from the desired puck direction.
    target_heading = np.arctan2(direction[1], direction[0])
    return wrap(target_heading - swing_sign * np.pi / 2.0)


def navigation_point(x, y, theta):
    # Virtual point used for approximate linearization:
    # p = [x, y] + l*[cos(theta), sin(theta)].
    return np.array([
        x + NAV_LOOKAHEAD_DIST * np.cos(theta),
        y + NAV_LOOKAHEAD_DIST * np.sin(theta),
    ])
