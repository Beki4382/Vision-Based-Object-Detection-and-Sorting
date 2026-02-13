#!/usr/bin/env python3
"""
Scene Manager Node for Perfect Size V1

This node is responsible for:
1. Subscribing to detected cube poses from the vision node (with size info)
2. Publishing the poses of objects (cubes) in the scene
3. Providing a service for the pick-place controller to query object locations
4. Matching detected cubes with their target place locations
5. Sorting cubes by priority: big_green, small_green, big_red, small_red

This is the size-based sorting version that handles 4 cubes of 2 colors and 2 sizes.

Author: Perfect Size V1 Architecture
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
    def __init__(self, name: str, color: str, size: str, pick_x: float, pick_y: float, 
                 place_x: float, place_y: float, priority: int):
        self.name = name
        self.color = color
        self.size = size  # 'big' or 'small'
        self.pick_x = pick_x
        self.pick_y = pick_y
        self.place_x = place_x
        self.place_y = place_y
        self.priority = priority  # Lower = higher priority
        self.is_placed = False


class SceneManager(Node):
    """
    Scene Manager node that receives cube locations from vision and 
    publishes them for the pick-place controller.
    
    This is the size-based sorting version that handles 4 cubes.
    Priority order: big_green (1), small_green (2), big_red (3), small_red (4)
    """
    
    def __init__(self):
        super().__init__('scene_manager')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # ══════════════════════════════════════════════════════════════════════
        # Place Positions (fixed target locations for each size+color combination)
        # Two cubes go on each color marker, side by side
        # ══════════════════════════════════════════════════════════════════════
        self.place_positions = {
            'big_green':   {'x': 0.70, 'y': -0.40},  # Green marker left side
            'small_green': {'x': 0.85, 'y': -0.40},  # Green marker right side
            'big_red':     {'x': 0.35, 'y': -0.40},  # Red marker left side
            'small_red':   {'x': 0.20, 'y': -0.40},  # Red marker right side
        }
        
        # Priority order (lower = picked first)
        self.priority_order = {
            'big_green': 1,
            'small_green': 2,
            'big_red': 3,
            'small_red': 4,
        }
        
        # Heights (relative to robot base)
        # Different heights for big vs small cubes
        self.heights = {
            'small': {
                'approach': 0.28,
                'pick': 0.16,      # 6cm cube, half height = 3cm above table
                'place': 0.18,
            },
            'big': {
                'approach': 0.32,
                'pick': 0.19,      # 12cm cube, half height = 6cm above table
                'place': 0.21,
            },
        }
        self.safe_height = 0.38
        
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
        
        self.get_logger().info('Scene Manager (Size Sorting V1) starting...')
        self.get_logger().info('Waiting for vision detections on /detected_cubes...')
        self.get_logger().info('Priority order: big_green -> small_green -> big_red -> small_red')
    
    def _detection_callback(self, msg: String):
        """Handle detection updates from vision node."""
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            
            # Update cubes from detections
            new_cubes = {}
            for det in detections:
                color = det.get('color', 'unknown')
                size = det.get('size', 'small')
                
                if color in ['red', 'green']:
                    # Create unique key for this cube
                    cube_key = f'{size}_{color}'
                    
                    # Get place position for this size+color combination
                    place_pos = self.place_positions.get(cube_key, {'x': 0.5, 'y': -0.3})
                    priority = self.priority_order.get(cube_key, 99)
                    
                    new_cubes[cube_key] = CubeInfo(
                        name=f'{cube_key}_cube',
                        color=color,
                        size=size,
                        pick_x=det['x'],
                        pick_y=det['y'],
                        place_x=place_pos['x'],
                        place_y=place_pos['y'],
                        priority=priority
                    )
                    
                    self.get_logger().debug(
                        f'Vision detected {size} {color} cube at ({det["x"]:.3f}, {det["y"]:.3f})'
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
            
            # Sort cubes by priority for display
            sorted_cubes = sorted(self.cubes.items(), key=lambda x: x[1].priority)
            
            self.get_logger().info('═' * 60)
            self.get_logger().info('   Scene Manager Ready (Size-Based Sorting)')
            self.get_logger().info('═' * 60)
            self.get_logger().info(f'   Detected {len(self.cubes)} cube(s)')
            self.get_logger().info('   Priority Order:')
            for i, (key, cube) in enumerate(sorted_cubes, 1):
                self.get_logger().info(
                    f'   {i}. {cube.size.upper()} {cube.color.upper()}: '
                    f'pick ({cube.pick_x:.3f}, {cube.pick_y:.3f}) → '
                    f'place ({cube.place_x:.3f}, {cube.place_y:.3f})'
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
        
        # Sort cubes by priority
        sorted_cubes = sorted(self.cubes.values(), key=lambda x: x.priority)
        
        # Build JSON with all cube information
        cube_data = {
            'cubes': [],
            'heights': self.heights,
            'safe_height': self.safe_height
        }
        
        for cube in sorted_cubes:
            cube_info = {
                'name': cube.name,
                'color': cube.color,
                'size': cube.size,
                'priority': cube.priority,
                'pick': {'x': cube.pick_x, 'y': cube.pick_y},
                'place': {'x': cube.place_x, 'y': cube.place_y},
                'is_placed': cube.is_placed
            }
            cube_data['cubes'].append(cube_info)
        
        # Publish combined JSON
        json_msg = String()
        json_msg.data = json.dumps(cube_data)
        self._cube_poses_pub.publish(json_msg)
    
    def _get_cubes_callback(self, request, response):
        """Service callback to return cube information."""
        # Sort cubes by priority
        sorted_cubes = sorted(self.cubes.values(), key=lambda x: x.priority)
        
        cube_data = {
            'cubes': [],
            'heights': self.heights,
            'safe_height': self.safe_height
        }
        
        for cube in sorted_cubes:
            cube_info = {
                'name': cube.name,
                'color': cube.color,
                'size': cube.size,
                'priority': cube.priority,
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
