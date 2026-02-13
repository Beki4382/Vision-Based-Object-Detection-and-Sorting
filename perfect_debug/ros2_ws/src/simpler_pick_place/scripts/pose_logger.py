#!/usr/bin/env python3
"""
Pose Logger - Logs actual cube positions from Gazebo every second.
Queries Gazebo's dynamic_pose topic directly to get real positions.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import subprocess
import re


class PoseLogger(Node):
    def __init__(self):
        super().__init__('pose_logger')
        self.callback_group = ReentrantCallbackGroup()
        
        # Robot base position (for converting to robot frame)
        self.robot_base_x = 0.0
        self.robot_base_y = -0.5
        
        # Create timer to log poses every second
        self._log_timer = self.create_timer(
            1.0,
            self._log_poses,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('  POSE LOGGER - Tracking Gazebo cube positions')
        self.get_logger().info('=' * 60)

    def _query_gazebo_poses(self):
        """Query Gazebo for all dynamic poses."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t', '/world/pick_place_world/dynamic_pose/info', '-n', '1'],
                capture_output=True, text=True, timeout=2.0
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except:
            return None

    def _parse_pose(self, output: str, model_name: str):
        """Parse the pose of a specific model from Gazebo output."""
        if output is None:
            return None
        
        pose_blocks = output.split('pose {')
        for block in pose_blocks:
            name_match = re.search(r'name:\s*"([^"]+)"', block)
            if name_match and name_match.group(1) == model_name:
                pos_match = re.search(
                    r'position\s*\{[^}]*x:\s*([-\d.e]+)[^}]*y:\s*([-\d.e]+)[^}]*z:\s*([-\d.e]+)',
                    block, re.DOTALL
                )
                if pos_match:
                    return (float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3)))
        return None

    def _log_poses(self):
        """Log current cube positions."""
        output = self._query_gazebo_poses()
        
        timestamp = self.get_clock().now().to_msg()
        sec = timestamp.sec % 1000
        
        log_lines = [f'[T={sec:03d}s] GAZEBO CUBE POSITIONS:']
        
        for cube_name in ['red_cube', 'green_cube']:
            pose = self._parse_pose(output, cube_name)
            if pose:
                wx, wy, wz = pose
                rx = wx - self.robot_base_x
                ry = wy - self.robot_base_y
                
                # Check if cube is on table (z should be ~0.77) or fallen (z < 0.5)
                status = "ON TABLE" if wz > 0.5 else "FALLEN!"
                
                log_lines.append(
                    f'  {cube_name:12s}: world=({wx:.3f}, {wy:.3f}, {wz:.3f}) robot=({rx:.3f}, {ry:.3f}) [{status}]'
                )
            else:
                log_lines.append(f'  {cube_name:12s}: NOT FOUND')
        
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
