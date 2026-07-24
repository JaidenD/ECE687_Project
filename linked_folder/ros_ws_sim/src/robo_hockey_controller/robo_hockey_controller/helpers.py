import numpy as np
import math

from robo_hockey_controller.config import NAV_LOOKAHEAD_DIST


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

def check_segment_circle_intersection(p1, p2, center, radius):
    """
    check if the path between p1 and p2 intersects a circle (obstacle) centered 
    at center with the given radius.
    p1, p2 are numpy arrays [x, y].
    """
    path_vector = p2 - p1
    center_vector = center - p1

    path_length_sq = np.dot(path_vector, path_vector)
    # Project center_vector onto path_vector to find the closest point
    t = np.dot(center_vector, path_vector) / path_length_sq
    
    # Clamp t to [0, 1] to ensure the closest point lies on the segment
    #points close to obstacle but beyond the segment will not matter
    t_clamped = max(0.0, min(1.0, t)) #clamp t to [0,1]
    closest_point = p1 + t_clamped * path_vector
    # Check if the closest point is within the safe radius
    distance = np.linalg.norm(center - closest_point)
    return distance <= radius

def is_point_in_rect(point, rect_center, rect_yaw, length, width):
    """
    Checks if a 2D point is inside a rotated rectangle.
    
    :param point: [x, y] of the target point
    :param rect_center: [x, y] of the rectangle's center
    :param rect_yaw: yaw of the rectangle
    :param length: length of the rectangle along x
    :param width: width of the rectangle along y
    :return: True if inside, False otherwise
    """
    px, py = point
    cx, cy = rect_center

    # method: convert the point to the rectangle's local frame
    dx = px - cx
    dy = py - cy
    local_x = dx * math.cos(rect_yaw) + dy * math.sin(rect_yaw)
    local_y = -dx * math.sin(rect_yaw) + dy * math.cos(rect_yaw)

    return abs(local_x) <= length / 2.0 and abs(local_y) <= width / 2.0
