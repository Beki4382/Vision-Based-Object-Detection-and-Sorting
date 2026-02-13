"""Launch file to spawn the red cube into Gazebo Sim."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Cube pose arguments
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='0.4'),
        DeclareLaunchArgument('y', default_value='0.15'),
        DeclareLaunchArgument('z', default_value='0.725'),

        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=[
                '-file', PathJoinSubstitution([
                    FindPackageShare('simpler_description'),
                    'models', 'red_cube', 'model.sdf'
                ]),
                '-name', 'red_cube',
                '-x', x,
                '-y', y,
                '-z', z,
            ],
        ),
    ])
