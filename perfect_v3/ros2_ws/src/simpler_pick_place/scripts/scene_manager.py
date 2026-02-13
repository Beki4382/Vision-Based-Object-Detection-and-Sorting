#!/usr/bin/env python3
"""
Scene Manager Node for Perfect V2

This node is responsible for:
1. Publishing the poses of objects (cubes) in the scene
2. Providing a service for the pick-place controller to query object locations
3. Monitoring object states (attached/detached)

The scene is set up by the launch file (Gazebo world + robot spawn).
This node reads the actual cube positions from Gazebo and publishes them.

Author: Perfect V2 Architecture
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from std_srvs.srv import Trigger

import json
import time


class CubeInfo:
    """Information about a cube in the scene."""
    def __init__(self, name: str, color: str, pick_x: float, pick_y: float, 
                 place_x: float, place_y: float):
        self.name = name
        self.color = color
        self.pick_x = pick_x
        self.pick_y = pick_y
        self.place_x = place_x
        self.place_y = place_y
        self.is_placed = False


class SceneManager(Node):
    """
    Scene Manager node that publishes object locations for the pick-place controller.
    
    In a real vision-based system, this would be replaced by a camera/perception node.
    For simulation, we know the spawn positions from the world file.
    """
    
    def __init__(self):
        super().__init__('scene_manager')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # ══════════════════════════════════════════════════════════════════════
        # Scene Configuration
        # Robot is spawned at world position (0, -0.5, 0.74)
        # Cube positions in world frame need to be converted to robot base frame
        # ══════════════════════════════════════════════════════════════════════
        
        # Robot base position in world frame
        self.robot_base_world_y = -0.5
        
        # Cube configurations (world coordinates from pick_place.sdf)
        # Red cube: World (0.4, -0.35, 0.77) → Robot base frame: (0.4, 0.15)
        # Green cube: World (0.4, -0.60, 0.77) → Robot base frame: (0.4, -0.10)
        self.cubes = {
            'red': CubeInfo(
                name='red_cube',
                color='red',
                pick_x=0.4,
                pick_y=0.15,  # -0.35 - (-0.5) = 0.15
                place_x=0.35,
                place_y=-0.30
            ),
            'green': CubeInfo(
                name='green_cube', 
                color='green',
                pick_x=0.4,
                pick_y=-0.10,  # -0.60 - (-0.5) = -0.10
                place_x=0.70,  # Moved from 0.55 to avoid overlap with red marker
                place_y=-0.30
            )
        }
        
        # Heights (relative to robot base)
        self.approach_height = 0.28
        self.pick_height = 0.16
        self.place_height = 0.18  # Slightly higher than pick for release
        self.safe_height = 0.35
        
        # ══════════════════════════════════════════════════════════════════════
        # Publishers
        # ══════════════════════════════════════════════════════════════════════
        
        # Publish cube poses as JSON for easy parsing
        # Uses latching QoS so late subscribers get the last message
        latching_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self._cube_poses_pub = self.create_publisher(
            String,
            '/scene/cube_poses',
            latching_qos
        )
        
        # Publish individual cube poses
        self._red_pose_pub = self.create_publisher(
            PoseStamped,
            '/scene/red_cube/pose',
            latching_qos
        )
        
        self._green_pose_pub = self.create_publisher(
            PoseStamped,
            '/scene/green_cube/pose',
            latching_qos
        )
        
        # Scene ready signal
        self._scene_ready_pub = self.create_publisher(
            String,
            '/scene/status',
            latching_qos
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Services
        # ══════════════════════════════════════════════════════════════════════
        
        # Service to get all cube information
        self._get_cubes_srv = self.create_service(
            Trigger,
            '/scene/get_cubes',
            self._get_cubes_callback,
            callback_group=self.callback_group
        )
        
        # Service to signal scene is ready
        self._scene_ready_srv = self.create_service(
            Trigger,
            '/scene/is_ready',
            self._is_ready_callback,
            callback_group=self.callback_group
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Initialization
        # ══════════════════════════════════════════════════════════════════════
        
        self._scene_ready = False
        
        # Timer to publish cube poses periodically
        self._publish_timer = self.create_timer(
            1.0,  # 1 Hz
            self._publish_cube_poses,
            callback_group=self.callback_group
        )
        
        # Initial delay to let Gazebo settle
        self.get_logger().info('Scene Manager starting...')
        self.get_logger().info('Waiting for Gazebo to settle...')
        
        # Create a one-shot timer to mark scene as ready after delay
        self._ready_timer = self.create_timer(
            5.0,  # Wait 5 seconds for Gazebo to settle
            self._mark_scene_ready,
            callback_group=self.callback_group
        )
    
    def _mark_scene_ready(self):
        """Mark the scene as ready after initial delay."""
        self._ready_timer.cancel()  # One-shot timer
        self._scene_ready = True
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('   Scene Manager Ready')
        self.get_logger().info('═' * 60)
        self.get_logger().info(f'   Cubes in scene: {list(self.cubes.keys())}')
        for color, cube in self.cubes.items():
            self.get_logger().info(
                f'   • {color.upper()}: pick=({cube.pick_x}, {cube.pick_y}) → '
                f'place=({cube.place_x}, {cube.place_y})'
            )
        self.get_logger().info('═' * 60)
        
        # Publish ready status
        status_msg = String()
        status_msg.data = 'ready'
        self._scene_ready_pub.publish(status_msg)
        
        # Publish initial poses
        self._publish_cube_poses()
    
    def _publish_cube_poses(self):
        """Publish current cube poses."""
        if not self._scene_ready:
            return
        
        # Build JSON with all cube information
        cube_data = {
            'cubes': [],
            'heights': {
                'approach': self.approach_height,
                'pick': self.pick_height,
                'place': self.place_height,
                'safe': self.safe_height
            }
        }
        
        for color, cube in self.cubes.items():
            cube_info = {
                'name': cube.name,
                'color': cube.color,
                'pick': {'x': cube.pick_x, 'y': cube.pick_y},
                'place': {'x': cube.place_x, 'y': cube.place_y},
                'is_placed': cube.is_placed
            }
            cube_data['cubes'].append(cube_info)
            
            # Publish individual pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose.position.x = cube.pick_x
            pose_msg.pose.position.y = cube.pick_y
            pose_msg.pose.position.z = self.pick_height
            pose_msg.pose.orientation.w = 1.0
            
            if color == 'red':
                self._red_pose_pub.publish(pose_msg)
            else:
                self._green_pose_pub.publish(pose_msg)
        
        # Publish combined JSON
        json_msg = String()
        json_msg.data = json.dumps(cube_data)
        self._cube_poses_pub.publish(json_msg)
    
    def _get_cubes_callback(self, request, response):
        """Service callback to return cube information."""
        cube_data = {
            'cubes': [],
            'heights': {
                'approach': self.approach_height,
                'pick': self.pick_height,
                'place': self.place_height,
                'safe': self.safe_height
            }
        }
        
        for color, cube in self.cubes.items():
            cube_info = {
                'name': cube.name,
                'color': cube.color,
                'pick': {'x': cube.pick_x, 'y': cube.pick_y},
                'place': {'x': cube.place_x, 'y': cube.place_y},
                'is_placed': cube.is_placed
            }
            cube_data['cubes'].append(cube_info)
        
        response.success = self._scene_ready
        response.message = json.dumps(cube_data)
        return response
    
    def _is_ready_callback(self, request, response):
        """Service callback to check if scene is ready."""
        response.success = self._scene_ready
        response.message = 'Scene is ready' if self._scene_ready else 'Scene not ready yet'
        return response


def main(args=None):
    rclpy.init(args=args)
    
    node = SceneManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
