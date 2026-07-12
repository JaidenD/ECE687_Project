import numpy as np


########################################################################
# lab calibration values - measure these before running on the robots
########################################################################

ROBOT1_ID = 1  # passing robot id
ROBOT2_ID = 2  # scoring robot id

HOLDER_MOCAP_TOPIC = '/vrpn_mocap/stick/pose'
PUCK_MOCAP_TOPIC = '/vrpn_mocap/puck/pose'
GOAL_MOCAP_TOPIC = '/vrpn_mocap/goal/pose'

# the two sticks are fixed side by side in the holder frame
STICK_SLOT_FORWARD_OFFSET = 0.0  # TODO: measure in the lab
STICK_SLOT_SEPARATION = 0.30     # TODO: measure in the lab
STICK_SLOT_OFFSETS = {
    ROBOT1_ID: np.array([
        STICK_SLOT_FORWARD_OFFSET,
        -STICK_SLOT_SEPARATION / 2.0,
    ]),
    ROBOT2_ID: np.array([
        STICK_SLOT_FORWARD_OFFSET,
        STICK_SLOT_SEPARATION / 2.0,
    ]),
}

# zero means the robots approach along the holder's positive x-axis
HOLDER_INSERT_HEADING_OFFSET = 0.0  # TODO: verify the holder mocap frame
BASE_TO_GRIPPER = 0.42              # TODO: base mocap origin to grasp point
PICKUP_INSERT_DISTANCE = 0.18       # straight insertion distance
PICKUP_ALIGNMENT_DISTANCE = 0.45    # rotate this far before the pregrasp point
PICKUP_RETREAT_DISTANCES = {
    ROBOT1_ID: 1.20,  # robot 1 clears the shared pickup area before robot 2 starts
    ROBOT2_ID: 0.45,
}

# MoveArm uses absolute x/z coordinates in meters relative to arm_base_link
ARM_PICKUP_X = 0.18         # TODO: measure the arm pose at the stick handle
ARM_PICKUP_Z = -0.02        # TODO: measure the arm pose at the stick handle
ARM_LIFT_X = ARM_PICKUP_X
ARM_LIFT_Z = 0.12           # TODO: high enough to clear the foam slot
ARM_PLAY_X = 0.18           # TODO: arm extension while playing hockey
ARM_PLAY_Z = -0.04          # TODO: stick height at the puck
ARM_CARRY_X = ARM_PLAY_X
ARM_CARRY_Z = ARM_LIFT_Z    # keep the stick above the puck while navigating

# Held-stick geometry. All distances are measured from the robot mocap origin.
STICK_HEADING_OFFSET_FROM_ROBOT = 0.0  # zero means the stick points forward
STICK_TIP_FROM_BASE = 0.65             # TODO: base origin to impact point

# Desired puck center relative to Robot 2. The current 0.55 m value is close
# to STICK_TIP_FROM_BASE minus one puck radius and a small clearance.
SCORING_RECEIVE_OFFSET_BODY = np.array([0.55, 0.0])
GOAL_TARGET_OFFSET_WORLD = np.array([0.0, 0.0])

# puck and impact model
PUCK_MASS = 0.17                    # kg, TODO: weigh the puck
PUCK_RADIUS = 0.08                  # m, TODO: measure the puck
PUCK_FRICTION_COEFFICIENT = 0.08    # TODO: identify experimentally
PUCK_LINEAR_DRAG = 0.0              # N/(m/s), TODO: identify experimentally
PUCK_QUADRATIC_DRAG = 0.0           # N/(m/s)^2, TODO: identify experimentally
PUCK_TARGET_ARRIVAL_SPEED = 0.20
STICK_EFFECTIVE_MASS = 5.0          # TODO: identify from impact tests
STICK_PUCK_RESTITUTION = 0.55       # TODO: identify from impact tests
IMPACT_CONTACT_TIME = 0.02  # seconds, only used to estimate average force

# known obstacle locations in the lab frame
# TODO: replace these with the measured obstacle centers and radii
LAB_OBSTACLES = ()


########################################################################
# simulator values
########################################################################

SIM_HOLDER_X = 1.25
SIM_HOLDER_Y = 0.0
SIM_HOLDER_THETA = 0.0

SIM_PUCK_X = -0.50
SIM_PUCK_Y = -0.50

SIM_GOAL_X = 0.0
SIM_GOAL_Y = -1.80

# known circular obstacles used to test the CBF constraints
SIM_OBSTACLES = (
    {
        'name': 'obstacle_1',
        'center': np.array([-0.20, 1.10]),
        'radius': 0.15,
    },
    {
        'name': 'obstacle_2',
        'center': np.array([0.75, -0.75]),
        'radius': 0.15,
    },
)

SIM_ARM_ACTION_TIME = 0.40
SIM_GRIPPER_ACTION_TIME = 0.30


########################################################################
# control values
########################################################################

CONTROL_FREQUENCY = 50.0
POSE_FILTER_ALPHA = 0.2  # alpha = 1.0 means no filtering
MOCAP_TIMEOUT = 1.0      # maximum mocap age before a measurement is stale
DEBUG_PRINT_PERIOD = 0.20

# approximate linearization and CLF-CBF-QP controller
NAV_LOOKAHEAD_DIST = 0.25  # distance from base origin to controlled point
NAV_KP = 0.80
NAV_POINT_TOL = 0.08
MAX_LINEAR_SPEED = 0.35
MAX_ANGULAR_SPEED = 1.40
CLF_GAIN = 1.00          # requested convergence rate
CBF_GAIN = 0.50          # how early the barrier slows inward motion
QP_SLACK_PENALTY = 1000.0  # high cost makes CLF relaxation a last resort
QP_SOLVER = 'quadprog'
NAVIGATION_ENVELOPE_RADIUS = 0.45
OTHER_ROBOT_SAFETY_DISTANCE = 0.70

# slow pickup controllers
HEADING_KP = 1.50
CROSS_TRACK_KP = 1.50
STRAIGHT_KP = 1.00
PICKUP_HEADING_TOL = 0.08
PICKUP_POSITION_TOL = 0.04
PICKUP_LATERAL_TOL = 0.05
PICKUP_APPROACH_SPEED = 0.12
PICKUP_INSERT_SPEED = 0.06
PICKUP_RETREAT_SPEED = 0.10
PICKUP_MAX_ANGULAR_SPEED = 0.35

# swing setup and execution
SWING_SIGN_BY_ROBOT = {
    ROBOT1_ID: 1.0,  # +1 is counterclockwise
    ROBOT2_ID: -1.0,
}
SWING_ANGLE_BEFORE_IMPACT = 0.55
SWING_FOLLOW_THROUGH_ANGLE = 0.25
SWING_MAX_ANGULAR_SPEED = 2.50
SWING_SETUP_APPROACH_DISTANCE = 0.35
SWING_SETUP_POSITION_TOL = 0.04
SWING_SETUP_HEADING_TOL = 0.06
SWING_SETTLE_TIME = 0.50
PUCK_RECEIVE_RADIUS = 0.45
PASS_SETTLE_TIME = 0.20
PUCK_VELOCITY_WINDOW = 0.50
PUCK_STOP_SPEED_TOL = 0.10
PUCK_STOP_HOLD_TIME = 0.40
AUTO_RUN_SCORING_SWING = True

ACTION_SERVER_TIMEOUT = 0.50
