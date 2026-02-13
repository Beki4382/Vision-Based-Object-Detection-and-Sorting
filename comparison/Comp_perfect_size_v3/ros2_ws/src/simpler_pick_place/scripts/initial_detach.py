#!/usr/bin/env python3
"""
Initial Detach Node - Detaches all cubes from the gripper at startup.

The DetachableJoint plugin in Gazebo attaches models by default when they
are configured. This causes cubes to move with the robot arm even before
the gripper physically touches them.

This node runs once at startup to send detach commands for all 4 cubes,
ensuring they are free to sit on the table independently.

Updated for Size V1: Handles 4 cubes (big_red, big_green, red, green)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import time


class InitialDetach(Node):
    """
    One-shot node that detaches all cubes from the gripper.
    """
    
    def __init__(self):
        super().__init__('initial_detach')
        
        # Create publishers for detach commands for all 4 cubes
        # Small cubes (6cm)
        self._detach_red_pub = self.create_publisher(Empty, '/gripper/detach_red', 10)
        self._detach_green_pub = self.create_publisher(Empty, '/gripper/detach_green', 10)
        # Big cubes (12cm)
        self._detach_big_red_pub = self.create_publisher(Empty, '/gripper/detach_big_red', 10)
        self._detach_big_green_pub = self.create_publisher(Empty, '/gripper/detach_big_green', 10)
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('  INITIAL DETACH - Releasing all 4 cubes from gripper')
        self.get_logger().info('═' * 60)
        
        # Wait for publishers to be ready
        time.sleep(0.5)
        
        # Send detach commands multiple times to ensure they're received
        for i in range(5):
            self.get_logger().info(f'  Sending detach commands (attempt {i+1}/5)...')
            # Small cubes
            self._detach_red_pub.publish(Empty())
            self._detach_green_pub.publish(Empty())
            # Big cubes
            self._detach_big_red_pub.publish(Empty())
            self._detach_big_green_pub.publish(Empty())
            time.sleep(0.2)
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('  ✓ Detach commands sent for all 4 cubes')
        self.get_logger().info('    - big_green_cube')
        self.get_logger().info('    - green_cube (small)')
        self.get_logger().info('    - big_red_cube')
        self.get_logger().info('    - red_cube (small)')
        self.get_logger().info('═' * 60)


def main(args=None):
    rclpy.init(args=args)
    
    node = InitialDetach()
    
    # Just run once and exit
    # The detach commands have already been sent in __init__
    
    # Keep node alive briefly to ensure messages are sent
    time.sleep(1.0)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
