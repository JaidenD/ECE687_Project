sudo docker exec -it dji_robomaster_ros_simulator /bin/bash -c '
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export __NV_PRIME_RENDER_OFFLOAD=1

    source /opt/ros/humble/setup.bash
    source /opt/ros/ws/setup.bash

    cd /ECE687/multi_robomaster_ros_sim/linked_folder/ros_ws_sim
    source install/setup.bash

    exec bash
'
