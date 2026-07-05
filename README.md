# ECE687 Project

## How to Run in Simulator

From the folder `/multi_robomaster_ros_sim`, start the simulator:

```bash
open -a XQuartz   # macOS only
./run.sh
```

In a new terminal, enter the simulator Docker container:

```bash
docker exec -it dji_robomaster_ros_simulator bash
```

Source the ROS environments:

```bash
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash
source /linked_folder/ros_ws_sim/install/setup.bash
```

Build and run the controller:

```bash
cd /linked_folder/ros_ws_sim
colcon build --packages-select robo_hockey_controller
source install/setup.bash

ros2 run robo_hockey_controller pickup_controller
```

To exit:

```bash
Ctrl + C
```

To reset the simulator state:

```bash
ros2 topic pub /sim/reset std_msgs/msg/Empty "{}" --once
```

---

## How to Run on the Physical Robot

Connect to the `brushbotarium` network.

If you were already connected to Colima before changing networks, reset Colima first:

```bash
colima stop
```

Start Colima:

```bash
colima start --network-address --network-mode=bridged
colima list
```

Start the Docker container:

```bash
docker run -it --rm \
  --network=host \
  --pid=host \
  --ipc=host \
  --name="dji_robomaster_ros" \
  dji_robomaster_ros:1.0
```

Once inside the Docker container, source ROS and launch the RoboMaster driver:

```bash
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash

ros2 launch robomaster_ros main.launch
```

Open a second terminal and enter the same Docker container:

```bash
docker exec -it dji_robomaster_ros bash
```

Source ROS again:

```bash
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash
```

Test the connection by rotating the robot slightly:

```bash
ros2 topic pub /robot9/cmd_vel geometry_msgs/msg/Twist \
"{angular: {z: 0.3}}" --once
```

Build the controller inside the container.

> **Note:** Change the `ROBOT_ID` value in the script before running.

```bash
cd /linked_folder/ros_ws_sim
colcon build --packages-select robo_hockey_controller
source install/setup.bash

ros2 run robo_hockey_controller pickup_controller
```

Check ROS topics with:

```bash
ros2 topic list
```