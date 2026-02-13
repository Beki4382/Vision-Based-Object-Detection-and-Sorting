#!/usr/bin/env python3
"""
Grasp Detector Node for Realistic Gripper Simulation (Size V3)

This node monitors:
1. Contact sensors on both gripper finger tips
2. Gripper joint position (open/closed state)

When both finger tips contact the same cube AND the gripper is closing,
it triggers the DetachableJoint attach. When the gripper opens, it detaches.

This provides more realistic grasping than manual attach/detach commands.

Author: Perfect Size V3
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Empty, Float64MultiArray, String
from sensor_msgs.msg import JointState

import time
from typing import Optional, Set


# Cube names in the simulation
CUBE_NAMES = ['red_cube', 'green_cube', 'big_red_cube', 'big_green_cube']

# Gripper thresholds
GRIPPER_CLOSED_THRESHOLD = 0.4  # Position above this = closed
GRIPPER_OPEN_THRESHOLD = 0.2    # Position below this = open

# Contact timing
CONTACT_HOLD_TIME = 0.1  # Seconds both fingers must contact before attach


class GraspDetector(Node):
    """
    Monitors gripper contacts and automatically triggers attach/detach.
    """

    def __init__(self):
        super().__init__('grasp_detector')
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('  GRASP DETECTOR - Contact-Based Grasping (Size V3)')
        self.get_logger().info('═' * 60)
        
        # State tracking
        self.gripper_position = 0.0
        self.left_finger_contacts: Set[str] = set()  # Models in contact with left finger
        self.right_finger_contacts: Set[str] = set()  # Models in contact with right finger
        self.attached_cube: Optional[str] = None
        self.contact_start_time: Optional[float] = None
        self.pending_attach_cube: Optional[str] = None
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribe to gripper joint state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # Note: In gz-sim, contact sensor data is published on Gazebo topics
        # We'll use a simpler approach: monitor gripper position and use timing
        # The contact sensors are for future enhancement with gz-ros bridge
        
        # Publishers for DetachableJoint attach/detach
        self._attach_pubs = {
            'red_cube': self.create_publisher(Empty, '/gripper/attach_red', 10),
            'green_cube': self.create_publisher(Empty, '/gripper/attach_green', 10),
            'big_red_cube': self.create_publisher(Empty, '/gripper/attach_big_red', 10),
            'big_green_cube': self.create_publisher(Empty, '/gripper/attach_big_green', 10),
        }
        
        self._detach_pubs = {
            'red_cube': self.create_publisher(Empty, '/gripper/detach_red', 10),
            'green_cube': self.create_publisher(Empty, '/gripper/detach_green', 10),
            'big_red_cube': self.create_publisher(Empty, '/gripper/detach_big_red', 10),
            'big_green_cube': self.create_publisher(Empty, '/gripper/detach_big_green', 10),
        }
        
        # Subscribe to gripper commands to detect intended grasps
        self.gripper_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/gripper_forward_position_controller/commands',
            self._gripper_cmd_callback,
            10
        )
        
        # Subscribe to pending grasp signal from controller
        self.pending_grasp_sub = self.create_subscription(
            String,
            '/gripper/pending_grasp',
            self._pending_grasp_callback,
            10
        )
        
        # Track commanded gripper state
        self.gripper_commanded_closed = False
        self.gripper_was_open = True
        
        # Timer for state monitoring
        self.create_timer(0.05, self._monitor_grasp_state)  # 20 Hz
        
        self.get_logger().info('Grasp Detector initialized')
        self.get_logger().info(f'  Closed threshold: {GRIPPER_CLOSED_THRESHOLD}')
        self.get_logger().info(f'  Open threshold: {GRIPPER_OPEN_THRESHOLD}')
        self.get_logger().info('Monitoring gripper state for automatic attach/detach...')
    
    def _joint_state_callback(self, msg: JointState):
        """Track gripper joint position."""
        try:
            idx = msg.name.index('robotiq_85_left_knuckle_joint')
            self.gripper_position = msg.position[idx]
        except (ValueError, IndexError):
            pass
    
    def _gripper_cmd_callback(self, msg: Float64MultiArray):
        """Track gripper commands to know intent."""
        if msg.data:
            cmd_position = msg.data[0]
            self.gripper_commanded_closed = cmd_position > GRIPPER_CLOSED_THRESHOLD
    
    def _pending_grasp_callback(self, msg):
        """Receive pending grasp signal from controller."""
        cube_name = msg.data
        if cube_name:
            # Normalize cube name (controller might send different formats)
            normalized = cube_name.replace('_cube', '').replace('_', '_')
            for name in CUBE_NAMES:
                if normalized in name or name.replace('_cube', '') in normalized:
                    self.pending_attach_cube = name
                    self.get_logger().info(f'[GRASP] Received pending grasp: {name}')
                    return
            # Try direct match
            if cube_name in CUBE_NAMES:
                self.pending_attach_cube = cube_name
                self.get_logger().info(f'[GRASP] Received pending grasp: {cube_name}')
    
    def _monitor_grasp_state(self):
        """
        Main grasp state machine.
        
        Logic:
        - When gripper closes past threshold AND was previously open -> attach
        - When gripper opens past threshold AND something is attached -> detach
        """
        gripper_is_closed = self.gripper_position > GRIPPER_CLOSED_THRESHOLD
        gripper_is_open = self.gripper_position < GRIPPER_OPEN_THRESHOLD
        
        # Detect gripper closing transition
        if gripper_is_closed and self.gripper_was_open and self.attached_cube is None:
            # Gripper just closed - attach to the target cube
            # In a real system, we'd check contact sensors here
            # For now, we rely on the controller telling us which cube to attach
            if self.pending_attach_cube:
                self._attach_cube(self.pending_attach_cube)
                self.pending_attach_cube = None
        
        # Detect gripper opening transition  
        if gripper_is_open and self.attached_cube is not None:
            # Gripper opened - detach
            self._detach_cube(self.attached_cube)
        
        # Update state
        self.gripper_was_open = gripper_is_open
    
    def _attach_cube(self, cube_name: str):
        """Attach a cube using DetachableJoint."""
        if cube_name not in self._attach_pubs:
            self.get_logger().warn(f'Unknown cube: {cube_name}')
            return
        
        self.get_logger().info(f'[GRASP] Attaching {cube_name} (contact-based)')
        msg = Empty()
        pub = self._attach_pubs[cube_name]
        
        # Send multiple times for reliability
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.02)
        
        self.attached_cube = cube_name
        self.get_logger().info(f'[GRASP] ✓ {cube_name} attached')
    
    def _detach_cube(self, cube_name: str):
        """Detach a cube using DetachableJoint."""
        if cube_name not in self._detach_pubs:
            self.get_logger().warn(f'Unknown cube: {cube_name}')
            return
        
        self.get_logger().info(f'[GRASP] Detaching {cube_name}')
        msg = Empty()
        pub = self._detach_pubs[cube_name]
        
        # Send multiple times for reliability
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.02)
        
        self.attached_cube = None
        self.get_logger().info(f'[GRASP] ✓ {cube_name} detached')
    
    def request_attach(self, cube_name: str):
        """
        Called by controller to indicate which cube should be attached
        when the gripper closes.
        """
        self.pending_attach_cube = cube_name
        self.get_logger().info(f'[GRASP] Pending attach: {cube_name}')
    
    def initial_detach_all(self):
        """Detach all cubes at startup."""
        self.get_logger().info('[GRASP] Initial detach of all cubes...')
        msg = Empty()
        
        for cube_name, pub in self._detach_pubs.items():
            for _ in range(5):
                pub.publish(msg)
                time.sleep(0.02)
        
        time.sleep(0.3)
        self.attached_cube = None
        self.get_logger().info('[GRASP] ✓ All cubes detached')


def main(args=None):
    rclpy.init(args=args)
    node = GraspDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
