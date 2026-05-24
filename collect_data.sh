#!/bin/bash


echo "Starting automated data collection loop..."
ros2 service call /isaacsim/GetSimulationState simulation_interfaces/srv/GetSimulationState
sleep 1
while true; do
    echo "=========================================="
    echo "Resuming Simulation..."
    ros2 service call /isaacsim/SetSimulationState simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"
    sleep 1
    
    echo "Launching biped_motion_planner..."
    ros2 launch biped_motion_planner biped_motion.launch.py
    
    echo "Launch process exited. Stopping Simulation..."
    sleep 0.5
    ros2 service call /isaacsim/SetSimulationState simulation_interfaces/srv/SetSimulationState "{state: {state: 0}}"
    sleep 0.5
done