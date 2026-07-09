import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import ColorRGBA
from math import cos, sin, pi
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from robo_hockey_controller.controller import (
    SIM_GOAL_X,
    SIM_GOAL_Y,
    SIM_PUCK_X,
    SIM_PUCK_Y,
    SIM_STICK_THETA,
    SIM_STICK_X,
    SIM_STICK_Y,
    STICK_DIST,
)

PROCESS_NOISE_STD = 0.002
MOCAP_POSITION_NOISE_STD = 0.02
MOCAP_YAW_NOISE_STD = 0.03

class MultiRoboMasterSim(Node):
    def __init__(self):
        super().__init__('multi_robomaster_sim')

        # constants
        # robots
        self.ROBOT_IDS = [1, 2]
        self.N = len(self.ROBOT_IDS)
        # time
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 20 # milliseconds
        self.TIMEOUT_GET_POSES = 10 # milliseconds
        self.TIMEOUT_CHASSIS_SPEED = 500 # milliseconds
        self.DT = (self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES) / 1000.
        # robot control
        self.MAX_LINEAR_SPEED = 1.0 # meters / second
        self.MAX_ANGULAR_SPEED = 360 * np.pi / 180 # radians / second
        # dimensions
        self.ENV = [-2., -2., 4., 4.] # (x, y) can vary from (ENV[0], ENV[1]) to (ENV[0]+ENV[2], ENV[1]+ENV[3])
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        self.GRIPPER_SIZE = 0.1
        
        # State: [x, y, theta]
        self.states = {}
        self.leds = {}
        self.velocities = {rid: np.array([0.0, 0.0, 0.0]) for rid in self.ROBOT_IDS}
        self.last_cmd_time = {rid: self.get_clock().now() for rid in self.ROBOT_IDS}
        
        # Initialize robots randomly
        for i, rid in enumerate(self.ROBOT_IDS):
            x = np.random.uniform(self.ENV[0], self.ENV[0] + self.ENV[2])
            y = np.random.uniform(self.ENV[1], self.ENV[1] + self.ENV[3])
            theta = np.random.random() * 2 * np.pi
            
            self.states[rid] = np.array([x, y, theta])
            self.leds[rid] = np.array([0., 0., 0.])

        # Pubs and Subs
        self.pubs = {}
        self.subs_vel = {}
        self.subs_led = {}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        for rid in self.ROBOT_IDS:
            # Publisher: Mimics VRPN motion capture system
            self.pubs[rid] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/dji_robot_{rid}/pose', qos)
            
            # Subscriber: Listen to the controller's cmd_vel
            self.subs_vel[rid] = self.create_subscription(
                Twist, f'/robot{rid}/cmd_vel', 
                lambda msg, rid=rid: self.vel_callback(msg, rid), qos)
            
            # Subscriber: Listen to the controller's leds
            self.subs_led[rid] = self.create_subscription(
                ColorRGBA, f'/robot{rid}/leds/color', 
                lambda msg, rid=rid: self.led_callback(msg, rid), qos)

        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(f"Simulator started for robots: {self.ROBOT_IDS}")

        # Plots
        self.figure = []
        self.axes = []
        self.patches_robots = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_grippers = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_stick_points = {rid: [] for rid in self.ROBOT_IDS}
        self.lines_stick_points = {rid: [] for rid in self.ROBOT_IDS}
        self.text_ids = {rid: [] for rid in self.ROBOT_IDS}
        self.__init_plot()
        self.__update_plot()

    def stick_overlay_point(self, rid):
        # point the controller drives to using the stick distance
        return np.array([
            self.states[rid][0] + STICK_DIST * cos(self.states[rid][2]),
            self.states[rid][1] + STICK_DIST * sin(self.states[rid][2]),
        ])
    
    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3], edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)
        self.__add_object_markers()

        for i, rid in enumerate(self.ROBOT_IDS):
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            p_robot = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                                     [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T),
                                                     facecolor='k')
            p_gripper = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T),
                                                       facecolor='k')
            stick_point = self.stick_overlay_point(rid)
            p_stick_point = patches.Circle(
                stick_point,
                radius=0.04,
                facecolor='tab:orange',
                edgecolor='black',
                linewidth=1.0,
                zorder=5,
            )
            line_stick_point, = self.axes.plot(
                [t[0], stick_point[0]],
                [t[1], stick_point[1]],
                color='tab:orange',
                linestyle='--',
                linewidth=1.5,
                zorder=4,
            )
            text_id = plt.text(self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0, s=str(self.ROBOT_IDS[i]), color="red")
            self.patches_robots[rid] = p_robot
            self.patches_grippers[rid] = p_gripper
            self.patches_stick_points[rid] = p_stick_point
            self.lines_stick_points[rid] = line_stick_point
            self.text_ids[rid] = text_id
            self.axes.add_patch(p_robot)
            self.axes.add_patch(p_gripper)
            self.axes.add_patch(p_stick_point)
        
        self.axes.set_xlim(self.ENV[0] - max(self.ROBOT_SIZE), self.ENV[0] + self.ENV[2] + max(self.ROBOT_SIZE))
        self.axes.set_ylim(self.ENV[1] - max(self.ROBOT_SIZE), self.ENV[1] + self.ENV[3] + max(self.ROBOT_SIZE))
        self.axes.grid()
        # self.axes.set_axis_off()
        self.axes.axis('equal')

        plt.ion()
        plt.show()

    def __add_object_markers(self):
        # fixed sim object positions used by controller.py
        stick_half_length = 0.35
        stick_dx = stick_half_length * cos(SIM_STICK_THETA)
        stick_dy = stick_half_length * sin(SIM_STICK_THETA)
        self.axes.plot(
            [SIM_STICK_X - stick_dx, SIM_STICK_X + stick_dx],
            [SIM_STICK_Y - stick_dy, SIM_STICK_Y + stick_dy],
            color='saddlebrown',
            linewidth=5.0,
            solid_capstyle='round',
            zorder=3,
        )
        self.axes.text(SIM_STICK_X, SIM_STICK_Y + 0.12, 'stick', color='saddlebrown', ha='center')

        p_puck = patches.Circle(
            (SIM_PUCK_X, SIM_PUCK_Y),
            radius=0.08,
            facecolor='tab:blue',
            edgecolor='black',
            linewidth=1.0,
            zorder=3,
        )
        self.axes.add_patch(p_puck)
        self.axes.text(SIM_PUCK_X, SIM_PUCK_Y + 0.12, 'puck', color='tab:blue', ha='center')

        goal_width = 0.7
        p_goal = patches.Rectangle(
            (SIM_GOAL_X - goal_width / 2.0, SIM_GOAL_Y - 0.04),
            goal_width,
            0.08,
            facecolor='none',
            edgecolor='tab:green',
            linewidth=3.0,
            zorder=3,
        )
        self.axes.add_patch(p_goal)
        self.axes.text(SIM_GOAL_X, SIM_GOAL_Y + 0.12, 'goal', color='tab:green', ha='center')
    
    def __update_plot(self):
        for rid in self.ROBOT_IDS:
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            xy_robot = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                      [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T)
            xy_gripper = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T)
        
            self.patches_robots[rid].xy = xy_robot
            self.patches_grippers[rid].xy = xy_gripper

            self.patches_robots[rid].set_facecolor(self.leds[rid])

            stick_point = self.stick_overlay_point(rid)
            self.patches_stick_points[rid].center = stick_point
            self.lines_stick_points[rid].set_data(
                [self.states[rid][0], stick_point[0]],
                [self.states[rid][1], stick_point[1]],
            )

            self.text_ids[rid].set_position((self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0))

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
    
    @staticmethod
    def transform_velocity_local_to_global(robots_speeds, theta):
        # robots_speeds : list of 3
        # theta : scalar
        robots_speeds_global = [0] * 3
        x_dot = robots_speeds[0]
        y_dot = robots_speeds[1]
        th_dot = robots_speeds[2]
        c_th = cos(theta)
        s_th = sin(theta)
        robots_speeds_global[0] = c_th * x_dot - s_th * y_dot
        robots_speeds_global[1] = s_th * x_dot + c_th * y_dot
        robots_speeds_global[2] = robots_speeds[2]
        return robots_speeds_global

    def vel_callback(self, msg, rid):
        # Store commanded velocities
        robot_speeds = MultiRoboMasterSim.transform_velocity_local_to_global([msg.linear.x, msg.linear.y, msg.angular.z], self.states[rid][2])
        self.velocities[rid] = np.array(robot_speeds)
        self.last_cmd_time[rid] = self.get_clock().now() # Update heartbeat

    def led_callback(self, msg, rid):
        # Store commanded velocities
        self.leds[rid] = np.array([msg.r, msg.g, msg.b])
        
    def update_and_publish(self):
        current_time = self.get_clock().now()

        for rid in self.ROBOT_IDS:
            elapsed_time_since_last_command_received = (current_time - self.last_cmd_time[rid]).nanoseconds / 1e9    
            if elapsed_time_since_last_command_received > self.TIMEOUT_CHASSIS_SPEED / 1e3:
                v_cmd = np.array([0.0, 0.0, 0.0])
            else:
                v_cmd = self.velocities[rid]
                
            # Integrate velocity
            # Global X/Y update
            # The controller sends local velocities, which is what the robots are expected to receive.
            # These are then converted to global in the velocity callback in the simulator
            #ideal
            # self.states[rid][0] += v_cmd[0] * self.DT
            # self.states[rid][1] += v_cmd[1] * self.DT 
            # self.states[rid][2] += v_cmd[2] * self.DT 
            # noisy
            self.states[rid][0] += v_cmd[0] * self.DT + np.random.normal(0, PROCESS_NOISE_STD)
            self.states[rid][1] += v_cmd[1] * self.DT + np.random.normal(0, PROCESS_NOISE_STD)
            self.states[rid][2] += v_cmd[2] * self.DT + np.random.normal(0, PROCESS_NOISE_STD)

            # Create PoseStamped message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            
            msg.pose.position.x = self.states[rid][0] + np.random.normal(0, MOCAP_POSITION_NOISE_STD)
            msg.pose.position.y = self.states[rid][1] + np.random.normal(0, MOCAP_POSITION_NOISE_STD)
            msg.pose.position.z = 0.0
            
            # Euler to Quaternion (simplified for 2D Z-axis rotation)
            noisy_yaw = self.states[rid][2] + np.random.normal(0, MOCAP_YAW_NOISE_STD)
            half_yaw = noisy_yaw * 0.5
            msg.pose.orientation.z = sin(half_yaw)
            msg.pose.orientation.w = cos(half_yaw)
            
            self.pubs[rid].publish(msg)

        self.__update_plot()

def main(args=None):
    rclpy.init(args=args)
    node = MultiRoboMasterSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
