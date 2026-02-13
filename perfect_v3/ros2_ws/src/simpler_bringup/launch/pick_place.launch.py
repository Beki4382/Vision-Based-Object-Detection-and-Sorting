"""
Pick-Place Controller Launch File for Perfect V2

Launches only the pick-place controller node.
This should be run AFTER scene.launch.py is running.

The controller will:
  1. Wait for the scene_manager to be ready
  2. Get cube positions from scene_manager
  3. Execute pick-and-place operations

Usage:
  # Terminal 1: Start the scene
  ros2 launch simpler_bringup scene.launch.py

  # Terminal 2: Start the controller (after scene is ready)
  ros2 launch simpler_bringup pick_place.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time',
        ),
    ]

    use_sim_time = LaunchConfiguration('use_sim_time')

    # Pick-Place Controller Node
    pick_place_controller_node = Node(
        package='simpler_pick_place',
        executable='pick_place_controller.py',
        name='pick_place_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription(
        declared_arguments
        + [
            pick_place_controller_node,
        ]
    )
