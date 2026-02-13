#!/usr/bin/env python3
"""
Scene Manager Node for Perfect YOLO V1

This node is responsible for:
1. Subscribing to detected cube poses from the vision node
2. Publishing the poses of objects (cubes) in the scene
3. Providing a service for the pick-place controller to query object locations
4. Matching detected cubes with their target place locations

This is the vision-based version that receives cube positions from 
the vision_node.py instead of using hardcoded positions.

Author: Perfect YOLO V1 Architecture
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
    Scene Manager node that receives cube locations from vision and 
    publishes them for the pick-place controller.
    
    This is the vision-based version that subscribes to /detected_cubes
    from the vision_node instead of using hardcoded positions.
    """
    
    def __init__(self):
        super().__init__('scene_manager')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # ══════════════════════════════════════════════════════════════════════
        # Place Positions (fixed target locations for each color)
        # These are where cubes should be placed, not where they are detected
        # ══════════════════════════════════════════════════════════════════════
        self.place_positions = {
            'red': {'x': 0.35, 'y': -0.30},
            'green': {'x': 0.70, 'y': -0.30}
        }
        
        # Heights (relative to robot base)
        self.approach_height = 0.28
        self.pick_height = 0.16
        self.place_height = 0.18  # Slightly higher than pick for release
        self.safe_height = 0.35
        
        # ══════════════════════════════════════════════════════════════════════
        # Cube Storage (updated from vision detections)
        # ══════════════════════════════════════════════════════════════════════
        self.cubes = {}  # Will be populated from vision detections
        self.detections_received = False
        self.last_detection_time = None
        
        # ══════════════════════════════════════════════════════════════════════
        # Subscribers
        # ══════════════════════════════════════════════════════════════════════
        
        # Subscribe to vision detections
        self._detection_sub = self.create_subscription(
            String,
            '/detected_cubes',
            self._detection_callback,
            10,
            callback_group=self.callback_group
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Publishers
        # ══════════════════════════════════════════════════════════════════════
        
        latching_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self._cube_poses_pub = self.create_publisher(
            String,
            '/scene/cube_poses',
            latching_qos
        )
        
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
        
        self._scene_ready_pub = self.create_publisher(
            String,
            '/scene/status',
            latching_qos
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Services
        # ══════════════════════════════════════════════════════════════════════
        
        self._get_cubes_srv = self.create_service(
            Trigger,
            '/scene/get_cubes',
            self._get_cubes_callback,
            callback_group=self.callback_group
        )
        
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
        
        # Timer to check for vision detections
        self._check_timer = self.create_timer(
            2.0,  # Check every 2 seconds
            self._check_vision_ready,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Scene Manager starting...')
        self.get_logger().info('Waiting for vision detections on /detected_cubes...')
    
    def _detection_callback(self, msg: String):
        """Handle detection updates from vision node."""
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            
            # Update cubes from detections
            new_cubes = {}
            for det in detections:
                color = det.get('color', 'unknown')
                if color in ['red', 'green']:
                    # Get place position for this color
                    place_pos = self.place_positions.get(color, {'x': 0.5, 'y': -0.3})
                    
                    new_cubes[color] = CubeInfo(
                        name=f'{color}_cube',
                        color=color,
                        pick_x=det['x'],
                        pick_y=det['y'],
                        place_x=place_pos['x'],
                        place_y=place_pos['y']
                    )
                    
                    self.get_logger().debug(
                        f'Vision detected {color} cube at ({det["x"]:.3f}, {det["y"]:.3f})'
                    )
            
            if new_cubes:
                self.cubes = new_cubes
                self.detections_received = True
                self.last_detection_time = self.get_clock().now()
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse detection message: {e}')
    
    def _check_vision_ready(self):
        """Check if vision has provided detections and mark scene ready."""
        if self._scene_ready:
            return
        
        if self.detections_received and len(self.cubes) > 0:
            self._scene_ready = True
            
            self.get_logger().info('═' * 60)
            self.get_logger().info('   Scene Manager Ready (Vision-Based)')
            self.get_logger().info('═' * 60)
            self.get_logger().info(f'   Detected cubes: {list(self.cubes.keys())}')
            for color, cube in self.cubes.items():
                self.get_logger().info(
                    f'   • {color.upper()}: detected at ({cube.pick_x:.3f}, {cube.pick_y:.3f}) → '
                    f'place at ({cube.place_x:.3f}, {cube.place_y:.3f})'
                )
            self.get_logger().info('═' * 60)
            
            # Publish ready status
            status_msg = String()
            status_msg.data = 'ready'
            self._scene_ready_pub.publish(status_msg)
            
            # Publish initial poses
            self._publish_cube_poses()
        else:
            self.get_logger().info('Still waiting for vision detections...')
    
    def _publish_cube_poses(self):
        """Publish current cube poses."""
        if not self._scene_ready or not self.cubes:
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
            elif color == 'green':
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
        
        response.success = self._scene_ready and len(self.cubes) > 0
        response.message = json.dumps(cube_data)
        return response
    
    def _is_ready_callback(self, request, response):
        """Service callback to check if scene is ready."""
        response.success = self._scene_ready and len(self.cubes) > 0
        if response.success:
            response.message = f'Scene is ready with {len(self.cubes)} cube(s) detected'
        else:
            response.message = 'Scene not ready - waiting for vision detections'
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
