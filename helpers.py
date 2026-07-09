import numpy as np

# Keep angle in [-pi, pi] range
def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def yaw_from_pose(msg):
    q = msg.pose.orientation
    return np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )