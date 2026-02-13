#!/usr/bin/env python3
"""
Pose Logger - Logs actual cube positions from Gazebo every second.

This node queries Gazebo's dynamic_pose topic directly to get the 
actual world poses of the red and green cubes.

This helps debug issues where cubes may be moving unexpectedly.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import subprocess
import re
import threading


class PoseLogger(Node):
    """
    Node that logs the actual poses of cubes from Gazebo every second.
    Uses subprocess to query Gazebo directly.
    """
    
    def __init__(self):
        super().__init__('pose_logger')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Robot base position (for converting to robot frame)
        self.robot_base_x = 0.0
        self.robot_base_y = -0.5
        
        # Create timer to log poses every second
        self._log_timer = self.create_timer(
            1.0,  # Every 1 second
            self._log_poses,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('  Pose Logger Started')
        self.get_logger().info('  Querying Gazebo for cube positions every 1 second...')
        self.get_logger().info('═' * 60)
    
    def _query_gazebo_poses(self):
        """Query Gazebo for all dynamic poses using gz topic."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t', '/world/pick_place_world/dynamic_pose/info', '-n', '1'],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            
            if result.returncode == 0:
                return result.stdout
            return None
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            self.get_logger().warn(f'Error querying Gazebo: {e}')
            return None
    
    def _parse_pose(self, output: str, model_name: str) -> tuple:
        """
        Parse the pose of a specific model from Gazebo output.
        
        Returns:
            (x, y, z) tuple or None if not found
        """
        if output is None:
            return None
        
        # Split by 'pose {' to get individual pose blocks
        pose_blocks = output.split('pose {')
        
        for block in pose_blocks:
            # Check if this block is for our model
            name_match = re.search(r'name:\s*"([^"]+)"', block)
            if name_match and name_match.group(1) == model_name:
                # Extract position
                pos_match = re.search(
                    r'position\s*\{[^}]*x:\s*([-\d.e]+)[^}]*y:\s*([-\d.e]+)[^}]*z:\s*([-\d.e]+)',
                    block,
                    re.DOTALL
                )
                if pos_match:
                    x = float(pos_match.group(1))
                    y = float(pos_match.group(2))
                    z = float(pos_match.group(3))
                    return (x, y, z)
        
        return None
    
    def _log_poses(self):
        """Log the current poses of all cubes."""
        timestamp = self.get_clock().now().to_msg()
        sec = timestamp.sec % 1000  # Just last 3 digits for readability
        
        # Query Gazebo for all poses
        output = self._query_gazebo_poses()
        
        log_lines = [f'[T={sec:03d}s] Actual Cube Positions (Gazebo World → Robot Frame):']
        
        for cube_name in ['red_cube', 'green_cube']:
            pose = self._parse_pose(output, cube_name)
            
            if pose is not None:
                wx, wy, wz = pose
                
                # Convert to robot base frame
                rx = wx - self.robot_base_x
                ry = wy - self.robot_base_y
                
                log_lines.append(
                    f'  {cube_name:12s}: world=({wx:7.3f}, {wy:7.3f}, {wz:7.3f}) → robot=({rx:7.3f}, {ry:7.3f})'
                )
            else:
                log_lines.append(f'  {cube_name:12s}: POSITION UNKNOWN')
        
        # Log as single block
        self.get_logger().info('\n'.join(log_lines))


def main(args=None):
    rclpy.init(args=args)
    
    node = PoseLogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
