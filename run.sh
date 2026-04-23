xhost +

docker run -it --rm \
	--network=host --pid=host --ipc=host \
	--volume ./linked_folder:/linked_folder:rw \
	--volume "$HOME/.Xauthority:/root/.Xauthority:rw" \
	--env="DISPLAY" \
	--name="dji_robomaster_ros_simulator" dji_robomaster_ros:1.0 \
	/bin/bash -c "cd linked_folder/ros_ws_sim && colcon build && source install/setup.bash && ros2 run multi_robomaster_ros_sim simulator"
