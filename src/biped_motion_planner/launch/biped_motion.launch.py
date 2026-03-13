"""
Launch file to start biped_motion_planner nodes sequentially.
This script utilizes the ROS 2 launch system with TimerActions to 
initialize multiple nodes with a strict 0.5-second delay between each.
"""
import time

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    """
   src/biped_motion_planner/test Generates the launch description for the biped_motion_planner nodes.
    
    Returns:
        LaunchDescription: The launch description containing the nodes and timers.
    """
    
    launch_start_time_sec = time.monotonic()

    # Define the nodes
    stance_leg_node = Node(
        package='biped_motion_planner',
        executable='stance_leg_control_node',
        name='stance_leg_control_node',
        output='screen'
    )
    
    swing_leg_node = Node(
        package='biped_motion_planner',
        executable='swing_leg_control_node',
        name='swing_leg_control_node',
        output='screen'
    )
    
    counterweight_node = Node(
        package='biped_motion_planner',
        executable='counterweight_control_node',
        name='counterweight_control_node',
        output='screen'
    )
    
    planner_node = Node(
        package='biped_motion_planner',
        executable='biped_motion_planner_node',
        name='biped_motion_planner',
        parameters=[{'launch_start_time_sec': launch_start_time_sec}],
        output='screen'
    )

    return LaunchDescription([
        stance_leg_node,
        TimerAction(
            period=0.5,
            actions=[swing_leg_node]
        ),
        TimerAction(
            period=1.0,
            actions=[counterweight_node]
        ),
        TimerAction(
            period=1.5,
            actions=[planner_node]
        )
    ])