"""
Scene Launch File for Perfect YOLO V1

Launches the simulation environment with vision-based object detection:
  1. Gazebo Sim with custom world (table + cubes + RGBD camera)
  2. UR5e + Robotiq 2F-85 gripper
  3. MoveIt 2 move_group
  4. RViz (optional)
  5. Camera image bridge (RGB + Depth from Gazebo to ROS2)
  6. Vision node (YOLOv8 object detection)
  7. Scene Manager node (receives detected cube poses)

This launch file does NOT start the pick-place controller.
Run pick_place.launch.py separately to start the controller.

Usage:
  ros2 launch simpler_bringup scene.launch.py
  # Then in another terminal:
  ros2 launch simpler_bringup pick_place.launch.py
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
    # Robot description (xacro -> URDF)
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

    # Load SRDF
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

    # Spawn robot into Gazebo
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

    # Clock bridge
    gz_clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # DetachableJoint bridges for all 4 cubes (small + big)
    gz_attach_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Small red cube
            '/gripper/attach_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_red@std_msgs/msg/Bool[gz.msgs.Boolean',
            # Small green cube
            '/gripper/attach_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_green@std_msgs/msg/Bool[gz.msgs.Boolean',
            # Big red cube
            '/gripper/attach_big_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_big_red@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_big_red@std_msgs/msg/Bool[gz.msgs.Boolean',
            # Big green cube
            '/gripper/attach_big_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/detach_big_green@std_msgs/msg/Empty]gz.msgs.Empty',
            '/gripper/state_big_green@std_msgs/msg/Bool[gz.msgs.Boolean',
        ],
        output='screen',
    )

    # Dynamic pose bridge (for tracking cube positions)
    gz_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/pick_place_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen',
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Camera Image Bridge (Gazebo -> ROS2)
    # Bridges RGB and Depth images from the overhead RGBD camera
    # ──────────────────────────────────────────────────────────────────────────
    gz_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/camera/image',        # RGB image topic
            '/camera/depth_image',  # Depth image topic
        ],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Camera info bridge (for intrinsics)
    gz_camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
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
    # Initial Detach Node
    # The DetachableJoint plugin attaches cubes to gripper by default!
    # This node runs early to send detach commands and free the cubes.
    # ──────────────────────────────────────────────────────────────────────────
    initial_detach_node = Node(
        package='simpler_pick_place',
        executable='initial_detach.py',
        name='initial_detach',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Delay initial detach until after robot is spawned and bridge is ready
    delay_initial_detach = TimerAction(
        period=5.0,  # Wait for robot spawn and detach bridge to be ready
        actions=[initial_detach_node],
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

    # Delay MoveIt until joint_state_broadcaster is ready
    delay_moveit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[move_group_node, rviz_node],
        )
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Vision Node (YOLOv8 object detection)
    # ──────────────────────────────────────────────────────────────────────────
    vision_node = Node(
        package='simpler_pick_place',
        executable='vision_node.py',
        name='vision_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Delay vision node until camera is ready
    delay_vision = TimerAction(
        period=8.0,
        actions=[vision_node],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Scene Manager Node (receives detected cube poses from vision)
    # ──────────────────────────────────────────────────────────────────────────
    scene_manager_node = Node(
        package='simpler_pick_place',
        executable='scene_manager.py',
        name='scene_manager',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Delay scene manager until vision is ready
    delay_scene_manager = TimerAction(
        period=12.0,
        actions=[scene_manager_node],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Pose Logger Node (logs actual cube positions from Gazebo for debugging)
    # ──────────────────────────────────────────────────────────────────────────
    pose_logger_node = Node(
        package='simpler_pick_place',
        executable='pose_logger.py',
        name='pose_logger',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Delay pose logger until simulation is stable
    delay_pose_logger = TimerAction(
        period=10.0,
        actions=[pose_logger_node],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Grasp Detector Node (V3 - Contact-based grasp detection)
    # ──────────────────────────────────────────────────────────────────────────
    grasp_detector_node = Node(
        package='simpler_pick_place',
        executable='grasp_detector.py',
        name='grasp_detector',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Delay grasp detector until controllers are ready
    delay_grasp_detector = TimerAction(
        period=8.0,
        actions=[grasp_detector_node],
    )

    return LaunchDescription(
        declared_arguments
        + [
            gz_sim_launch,
            gz_clock_bridge,
            gz_attach_bridge,
            gz_pose_bridge,
            gz_image_bridge,
            gz_camera_info_bridge,
            robot_state_publisher,
            gz_spawn_robot,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
            delay_initial_detach,  # CRITICAL: Detach cubes right after spawn
            delay_moveit,
            delay_vision,
            delay_scene_manager,
            delay_pose_logger,
            delay_grasp_detector,  # V3: Contact-based grasp detection
        ]
    )
