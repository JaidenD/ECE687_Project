from time import perf_counter

import numpy as np
from qpsolvers import solve_qp


def solve_navigation_qp(
    point,
    target,
    theta,
    lookahead_distance,
    obstacle_centers,
    safety_distances,
    nominal_velocity,
    clf_gain,
    cbf_gain,
    slack_penalty,
    max_linear_speed,
    max_angular_speed,
    solver='quadprog',
):
    """Solve the CLF-CBF QP for the navigation-point velocity."""
    point = np.asarray(point, dtype=float)
    target = np.asarray(target, dtype=float)
    nominal_velocity = np.asarray(nominal_velocity, dtype=float)
    obstacle_centers = np.asarray(
        obstacle_centers,
        dtype=float,
    ).reshape((-1, 2))
    safety_distances = np.asarray(safety_distances, dtype=float)

    # Check array sizes here so a malformed QP fails with a useful message.
    if point.shape != (2,) or target.shape != (2,):
        raise ValueError('point and target must be 2D vectors')
    if nominal_velocity.shape != (2,):
        raise ValueError('nominal_velocity must be a 2D vector')
    if len(obstacle_centers) != len(safety_distances):
        raise ValueError('each obstacle must have one safety distance')
    if lookahead_distance <= 0.0:
        raise ValueError('lookahead_distance must be positive')
    if slack_penalty <= 0.0:
        raise ValueError('slack_penalty must be positive')

    # qpsolvers minimizes (1/2) z^T P z + q^T z.
    # The decision is z = [u_x, u_y, delta], where u is the desired velocity
    # of the navigation point and delta is the nonnegative CLF slack.
    objective = np.diag([2.0, 2.0, 2.0 * slack_penalty])
    objective_linear = np.array([
        -2.0 * nominal_velocity[0],
        -2.0 * nominal_velocity[1],
        0.0,
    ])

    inequalities = []
    inequality_bounds = []

    # CLF convergence constraint:
    #   V = ||p - p_des||^2
    #   V_dot = 2 (p - p_des)^T u
    #   V_dot + gamma*V <= delta.
    position_error = point - target
    clf_value = float(position_error @ position_error)
    inequalities.append(np.array([
        2.0 * position_error[0],
        2.0 * position_error[1],
        -1.0,
    ]))
    inequality_bounds.append(-clf_gain * clf_value)

    # One CBF is added for each circular obstacle:
    #   h = ||p - o||^2 - d_safe^2.
    # h >= 0 means the navigation point is outside the safety circle.
    # The static-obstacle CBF is 2 (p - o)^T u + gamma*h >= 0.
    barrier_values = []
    for center, safe_distance in zip(
        obstacle_centers,
        safety_distances,
    ):
        relative_position = point - center
        barrier_value = float(
            relative_position @ relative_position - safe_distance**2
        )
        barrier_values.append(barrier_value)
        # qpsolvers uses Gz <= h, so multiply the CBF inequality by -1.
        inequalities.append(np.array([
            -2.0 * relative_position[0],
            -2.0 * relative_position[1],
            0.0,
        ]))
        inequality_bounds.append(cbf_gain * barrier_value)

    # delta >= 0 becomes -delta <= 0 in the solver's Gz <= h form.
    inequalities.append(np.array([0.0, 0.0, -1.0]))
    inequality_bounds.append(0.0)

    # Approximate-linearization inverse input map:
    # v = [cos(theta), sin(theta)] u
    # omega = [-sin(theta), cos(theta)] u / lookahead_distance
    # Add both the positive and negative rows to impose absolute-value limits.
    c = np.cos(theta)
    s = np.sin(theta)
    velocity_rows = (
        (np.array([c, s, 0.0]), max_linear_speed),
        (np.array([-c, -s, 0.0]), max_linear_speed),
        (
            np.array([-s / lookahead_distance, c / lookahead_distance, 0.0]),
            max_angular_speed,
        ),
        (
            np.array([s / lookahead_distance, -c / lookahead_distance, 0.0]),
            max_angular_speed,
        ),
    )
    for row, bound in velocity_rows:
        inequalities.append(row)
        inequality_bounds.append(bound)

    start_time = perf_counter()
    inequality_matrix = np.vstack(inequalities)
    inequality_limit = np.asarray(inequality_bounds)
    solution = solve_qp(
        objective,
        objective_linear,
        inequality_matrix,
        inequality_limit,
        solver=solver,
    )
    solve_time = perf_counter() - start_time

    if solution is None or not np.all(np.isfinite(solution)):
        return None

    return {
        'point_velocity': np.asarray(solution[:2]),
        'slack': max(float(solution[2]), 0.0),
        'clf_value': clf_value,
        'barrier_values': np.asarray(barrier_values),
        'solve_time': solve_time,
    }
