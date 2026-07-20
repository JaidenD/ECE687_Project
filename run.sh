xhost +local:

docker run -it --rm \
    --network=host \
    --pid=host \
    --ipc=host \
    --device=/dev/dri:/dev/dri \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ./linked_folder:/linked_folder \
    --name dji_robomaster_ros_simulator \
    dji_robomaster_ros:1.0\
	/bin/bash -c "cd linked_folder/ros_ws_sim && colcon build && source install/setup.bash && ros2 run multi_robomaster_ros_sim simulator"
