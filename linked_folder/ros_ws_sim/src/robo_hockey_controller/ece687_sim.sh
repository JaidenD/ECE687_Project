#start the first container
sudo docker run -it --rm \
    --network=host \
    --pid=host \
    --ipc=host \
    --volume /home/ellie/DockerShared/ECE687:/ECE687 \
    --volume "$HOME/.Xauthority:/root/.Xauthority:rw" \
    --env DISPLAY \
    --name dji_robomaster_ros_simulator \
    dji_robomaster_ros:1.1 \
    /bin/bash -c '
        export XDG_RUNTIME_DIR=/tmp/runtime-root
        export __GLX_VENDOR_LIBRARY_NAME=nvidia
        export __NV_PRIME_RENDER_OFFLOAD=1

        source /opt/ros/humble/setup.bash
        source /opt/ros/ws/setup.bash

        cd /ECE687/multi_robomaster_ros_sim/linked_folder/ros_ws_sim

        colcon build
        source install/setup.bash
        ros2 run multi_robomaster_ros_sim simulator
        exec bash
    '

 

