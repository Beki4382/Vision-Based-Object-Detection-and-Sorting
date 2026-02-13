#!/usr/bin/env python3
"""
Pose Logger - Logs actual cube positions from Gazebo.

This node queries Gazebo's dynamic_pose topic directly to get the 
actual world poses of all 4 cubes (big_red, big_green, red, green).

This helps debug issues where cubes may be moving unexpectedly.
It also tracks if cubes have fallen off the table.

Updated for Size V1: Tracks 4 cubes with different sizes.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import subprocess
import re
import threading


class PoseLogger(Node):
    """
    Node that logs the actual poses of cubes from Gazebo.
    Uses subprocess to query Gazebo directly.
    """
    
    def __init__(self):
        super().__init__('pose_logger')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Robot base position (for converting to robot frame)
        self.robot_base_x = 0.0
        self.robot_base_y = -0.5
        
        # Expected cube positions (from SDF file)
        # Pick area moved forward to separate from place area
        # Big cubes (12cm) have half-height of 0.06, so z = 0.74 + 0.06 = 0.80
        # Small cubes (6cm) have half-height of 0.03, so z = 0.74 + 0.03 = 0.77
        self.expected_positions = {
            'big_green_cube': {'x': 0.25, 'y': -0.25, 'z': 0.80, 'size': 'big'},
            'green_cube': {'x': 0.55, 'y': -0.25, 'z': 0.77, 'size': 'small'},
            'big_red_cube': {'x': 0.25, 'y': -0.45, 'z': 0.80, 'size': 'big'},
            'red_cube': {'x': 0.55, 'y': -0.45, 'z': 0.77, 'size': 'small'},
        }
        
        # Table surface height
        self.table_height = 0.74
        
        # Track previous positions to detect movement
        self.prev_positions = {}
        
        # Create timer to log poses every 0.5 seconds
        self._log_timer = self.create_timer(
            0.5,  # Every 0.5 seconds for better tracking
            self._log_poses,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('═' * 70)
        self.get_logger().info('  POSE LOGGER - Tracking 4 Cube Positions (Size V1)')
        self.get_logger().info('  Expected positions (world coordinates):')
        self.get_logger().info(f'    big_green_cube:   ({0.25:.2f}, {-0.35:.2f}, {0.80:.2f}) - Priority 1')
        self.get_logger().info(f'    green_cube:       ({0.55:.2f}, {-0.35:.2f}, {0.77:.2f}) - Priority 2')
        self.get_logger().info(f'    big_red_cube:     ({0.25:.2f}, {-0.65:.2f}, {0.80:.2f}) - Priority 3')
        self.get_logger().info(f'    red_cube:         ({0.55:.2f}, {-0.65:.2f}, {0.77:.2f}) - Priority 4')
        self.get_logger().info('  Table surface at z=0.74')
        self.get_logger().info('═' * 70)
    
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
        """Log the current poses of all cubes with movement detection."""
        timestamp = self.get_clock().now().to_msg()
        sec = timestamp.sec % 1000  # Just last 3 digits for readability
        nanosec = timestamp.nanosec // 100000000  # First digit of nanoseconds
        
        # Query Gazebo for all poses
        output = self._query_gazebo_poses()
        
        log_lines = [f'[T={sec:03d}.{nanosec}s] CUBE POSITIONS:']
        
        # Process cubes in priority order
        cube_names = ['big_green_cube', 'green_cube', 'big_red_cube', 'red_cube']
        
        for cube_name in cube_names:
            pose = self._parse_pose(output, cube_name)
            expected = self.expected_positions.get(cube_name, {})
            size_label = expected.get('size', 'small').upper()
            
            if pose is not None:
                wx, wy, wz = pose
                
                # Convert to robot base frame
                rx = wx - self.robot_base_x
                ry = wy - self.robot_base_y
                
                # Check for issues
                issues = []
                
                # Check if cube has fallen off table
                if wz < self.table_height - 0.1:
                    issues.append(f'FALLEN! (z={wz:.2f} < table)')
                
                # Check if cube has moved from expected position
                if expected:
                    dx = abs(wx - expected.get('x', wx))
                    dy = abs(wy - expected.get('y', wy))
                    if dx > 0.05 or dy > 0.05:
                        issues.append(f'MOVED ({dx:.2f}, {dy:.2f})')
                
                # Check if cube moved since last check
                if cube_name in self.prev_positions:
                    prev = self.prev_positions[cube_name]
                    move_dist = ((wx - prev[0])**2 + (wy - prev[1])**2 + (wz - prev[2])**2)**0.5
                    if move_dist > 0.01:  # More than 1cm movement
                        issues.append(f'MOVING! delta={move_dist:.3f}m')
                
                # Store current position
                self.prev_positions[cube_name] = (wx, wy, wz)
                
                # Build log line
                status = ' '.join(issues) if issues else 'OK'
                # Create display name
                display_name = cube_name.replace('_cube', '').upper()
                log_lines.append(
                    f'  {display_name:10s}: world=({wx:6.3f}, {wy:6.3f}, {wz:6.3f}) robot=({rx:6.3f}, {ry:6.3f}) [{status}]'
                )
            else:
                display_name = cube_name.replace('_cube', '').upper()
                log_lines.append(f'  {display_name:10s}: POSITION UNKNOWN')
        
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
