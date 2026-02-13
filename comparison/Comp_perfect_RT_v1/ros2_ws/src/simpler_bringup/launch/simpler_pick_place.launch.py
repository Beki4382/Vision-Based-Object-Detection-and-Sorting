"""
Combined Launch File for Perfect V2

Launches everything in one command:
  1. Gazebo Sim with custom world (table + cubes)
  2. UR5e + Robotiq 2F-85 gripper
  3. MoveIt 2 move_group
  4. RViz (optional)
  5. Scene Manager node (publishes cube poses)
  6. Pick-Place Controller (waits for scene manager, then executes)

This is a convenience launch file that runs both scene and controller.
For separate execution, use scene.launch.py and pick_place.launch.py.

Usage:
  ros2 launch simpler_bringup simpler_pick_place.launch.py
"""

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    world_file = LaunchConfiguration('world_file')
    launch_rviz = LaunchConfiguration('launch_rviz')
    gazebo_gui = LaunchConfiguration('gazebo_gui')

    declared_arguments = [
        DeclareLaunchArgument(
            'ur_type',
            default_value='ur5e',
            description='Type of UR robot to spawn',
        ),
        DeclareLaunchArgument(
            'world_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('simpler_description'),
                'worlds',
                'pick_place.sdf',
            ]),
            description='Path to Gazebo world file',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz?',
        ),
        DeclareLaunchArgument(
            'gazebo_gui',
            default_value='true',
            description='Launch Gazebo with GUI?',
        ),
    ]

    # Controller config file
    controllers_file = PathJoinSubstitution([
        FindPackageShare('simpler_bringup'),
        'config',
        'ur5e_robotiq_controllers.yaml',
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # Robot description
    # ──────────────────────────────────────────────────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('simpler_description'),
            'urdf',
            'ur5e_robotiq.urdf.xacro',
        ]),
        ' name:=ur',
        ' ur_type:=', ur_type,
        ' tf_prefix:=""',
        ' safety_limits:=true',
        ' simulation_controllers:=', controllers_file,
    ])
    robot_description = {'robot_description': robot_description_content}

    srdf_path = os.path.join(
        get_package_share_directory('simpler_description'),
        'config',
        'ur5e_robotiq.srdf'
    )
    with open(srdf_path, 'r') as srdf_file:
        robot_description_semantic_content = srdf_file.read()
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_content}

    # ──────────────────────────────────────────────────────────────────────────
    # Gazebo Sim
    # ──────────────────────────────────────────────────────────────────────────
    gz_args = IfElseSubstitution(
        gazebo_gui,
        if_value=['-r -v 4 ', world_file],
        else_value=['-s -r --headless-rendering -v 4 ', world_file],
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'),
            '/launch/gz_sim.launch.py',
        ]),
        launch_arguments={
            'gz_args': gz_args,
            'on_exit_shutdown': 'true',
        }.items(),
    )

    gz_spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-string', robot_description_content,
            '-name', 'ur',
            '-allow_renaming', 'true',
            '-x', '0',
            '-y', '-0.5',
            '-z', '0.74',
        ],
    )

    gz_clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    gz_attach_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/gripper/attach_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_red@std_msgs/msg/Bool[gz.msgs.Boolean',
            '/gripper/attach_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_green@std_msgs/msg/Bool[gz.msgs.Boolean',
        ],
        output='screen',
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Robot state publisher
    # ──────────────────────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'use_sim_time': True}, robot_description],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Controllers
    # ──────────────────────────────────────────────────────────────────────────
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller', '-c', '/controller_manager'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_forward_position_controller', '-c', '/controller_manager'],
        output='screen',
    )

    # ──────────────────────────────────────────────────────────────────────────
    # MoveIt 2
    # ──────────────────────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(file_path=Path("urdf") / "ur.urdf.xacro", mappings={"ur_type": "ur5e", "name": "ur"})
        .robot_description_semantic(file_path=srdf_path)
        .robot_description_kinematics(file_path=Path("config") / "kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution(file_path=Path("config") / "moveit_controllers.yaml")
        .to_moveit_configs()
    )

    moveit_config.robot_description = robot_description

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': True},
            # Increase trajectory tolerance to reduce failures
            {'trajectory_execution.allowed_start_tolerance': 0.05},
        ],
    )

    # RViz - use our custom config with correct Fixed Frame (base_link)
    simpler_bringup_pkg = get_package_share_directory('simpler_bringup')
    rviz_config = os.path.join(simpler_bringup_pkg, 'rviz', 'view_robot.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[
            # Pass robot_description directly (our custom URDF with gripper)
            robot_description,
            robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': True},
        ],
        condition=IfCondition(launch_rviz),
    )

    delay_moveit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[move_group_node, rviz_node],
        )
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Scene Manager Node
    # ──────────────────────────────────────────────────────────────────────────
    scene_manager_node = Node(
        package='simpler_pick_place',
        executable='scene_manager.py',
        name='scene_manager',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    delay_scene_manager = TimerAction(
        period=10.0,
        actions=[scene_manager_node],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Pick-Place Controller Node
    # ──────────────────────────────────────────────────────────────────────────
    pick_place_node = Node(
        package='simpler_pick_place',
        executable='pick_place_controller.py',
        name='pick_place_controller',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Wait for scene to be fully ready before starting controller
    delay_pick_place = TimerAction(
        period=20.0,
        actions=[pick_place_node],
    )

    return LaunchDescription(
        declared_arguments
        + [
            gz_sim_launch,
            gz_clock_bridge,
            gz_attach_bridge,
            robot_state_publisher,
            gz_spawn_robot,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
            delay_moveit,
            delay_scene_manager,
            delay_pick_place,
        ]
    )
