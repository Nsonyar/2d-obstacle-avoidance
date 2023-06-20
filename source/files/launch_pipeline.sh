#!/bin/bash
#call with bash ./launch_pipeline.sh

clear
echo "............pipeline launched............"
sleep 1
source /root/catkin_ws/devel/setup.bash

python3 launch_pipeline.py

echo "....pipeline sucessfully terminated...."
