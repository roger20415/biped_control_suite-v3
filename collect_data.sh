#!/bin/bash

SIGNAL_FILE="/shared_shutdown_signal/restart_isaac.flag"
TIMEOUT_SEC=10

send_restart_signal() {
    echo "Isaac Sim is unresponsive. Sending restart signal..."
    if ! [ -f "$SIGNAL_FILE" ]; then
        touch "$SIGNAL_FILE"
        chmod 777 "$SIGNAL_FILE"
    fi
}

call_with_timeout() {
    local t_sec=$1
    shift
    timeout "$t_sec" "$@"
    if [ $? -eq 124 ]; then
        send_restart_signal
        return 1
    fi
    return 0
}

echo "Starting automated data collection loop..."

call_with_timeout 5 ros2 service call /isaacsim/GetSimulationState simulation_interfaces/srv/GetSimulationState
sleep 1

while true; do
    echo "=========================================="
    
    if [ -f "$SIGNAL_FILE" ]; then
        echo "Waiting for Isaac Sim to restart..."
        sleep 20
        continue
    fi

    echo "Resuming Simulation..."
    if ! call_with_timeout $TIMEOUT_SEC ros2 service call /isaacsim/SetSimulationState simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"; then
        continue
    fi
    sleep 1
    
    ros2 launch biped_motion_planner biped_motion.launch.py

    echo "Launch process exited. Stopping Simulation..."
    sleep 0.5
    
    if ! call_with_timeout $TIMEOUT_SEC ros2 service call /isaacsim/SetSimulationState simulation_interfaces/srv/SetSimulationState "{state: {state: 0}}"; then
        continue
    fi
    sleep 0.5
done