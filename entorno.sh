#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [ -f "$HOME/grupo_05_tm5_900_ws/install/setup.bash" ]; then
  source "$HOME/grupo_05_tm5_900_ws/install/setup.bash"
fi
