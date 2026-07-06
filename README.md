# ECE687 Project

## How to Run in Simulator

In a Mac terminal, go to the folder `multi_robomaster_ros_sim` and start the
simulator:

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

In a Mac terminal, run the setup commands below.

If Colima was already running before changing networks, stop it first:

```bash
colima stop
```

Start Colima in bridged mode:

```bash
colima start --network-address --network-mode=bridged
colima list
```

If an old robot container is already running, stop it:

```bash
docker stop dji_robomaster_ros 2>/dev/null || true
```

From the project root, start the Docker container with this project mounted into
`/linked_folder`:

```bash
cd "/Users/jaiden/Desktop/Robotics/Robot Dynamics/FinalProject"

docker run -it --rm \
  --network=host \
  --pid=host \
  --ipc=host \
  --name="dji_robomaster_ros" \
  -v "$PWD/simulator/multi_robomaster_ros_sim/linked_folder:/linked_folder:rw" \
  dji_robomaster_ros:1.0
```

Once inside the Docker container, source ROS and launch the RoboMaster driver.
Leave this terminal running:

```bash
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash

ros2 launch robomaster_ros main.launch
```

Open a second Mac terminal and enter the same Docker container:

```bash
docker exec -it dji_robomaster_ros bash
```

Source ROS again:

```bash
source /opt/ros/humble/setup.bash
source /opt/ros/ws/setup.bash
```

Check which robot topics exist:

```bash
ros2 topic list
```

Look for the real robot number in topics like `/robot5/cmd_vel`. Use that robot
number in the commands below.

Test the connection by rotating the robot slightly:

```bash
ros2 topic pub /robot5/cmd_vel geometry_msgs/msg/Twist \
"{angular: {z: 0.3}}" --once
```

Stop the robot:

```bash
ros2 topic pub /robot5/cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once
```

Build the controller inside the container.

> **Note:** Change the robot ID value in the controller script before running.
> If `/linked_folder/ros_ws_sim` does not exist, the container was started
> without the `-v ...:/linked_folder:rw` mount above.

```bash
cd /linked_folder/ros_ws_sim
colcon build --packages-select robo_hockey_controller
source install/setup.bash

ros2 run robo_hockey_controller pickup_controller
```

To stop the controller:

```bash
Ctrl+C
```
