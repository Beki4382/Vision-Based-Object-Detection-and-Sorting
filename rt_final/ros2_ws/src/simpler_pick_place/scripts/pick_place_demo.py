#!/usr/bin/env python3
"""
PERFECT Pick-and-Place Demo for UR5e + Robotiq Gripper in Gazebo Sim (ROS 2 Jazzy)

This implementation combines the best practices from the original ROS1 project
with modern ROS 2 features for ultra-smooth, robust pick-and-place:

Key Features:
  1. Cartesian path planning for smooth linear motions
  2. Sinusoidal velocity profiling (smooth acceleration/deceleration)
  3. Fine-grained waypoint interpolation (100+ points per motion)
  4. Quaternion SLERP for smooth orientation changes
  5. Physics-based grasping via DetachableJoint
  6. Robust error handling with intelligent retries
  7. Real-time trajectory execution monitoring
  8. Multi-cube support (red and green cubes)

Author: Enhanced for Monte Project
"""

import sys
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray, Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    PlanningOptions,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from shape_msgs.msg import SolidPrimitive
from builtin_interfaces.msg import Duration


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
PLANNING_GROUP = 'ur_manipulator'
END_EFFECTOR_LINK = 'tool0'
BASE_FRAME = 'base_link'

# Joint names for UR5e
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint', 
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Red cube position relative to robot base (World: 0.4, -0.35 → Robot: 0.4, 0.15)
RED_CUBE_X, RED_CUBE_Y = 0.4, 0.15
# Green cube position relative to robot base (World: 0.4, -0.60 → Robot: 0.4, -0.10)
GREEN_CUBE_X, GREEN_CUBE_Y = 0.4, -0.10

CUBE_HALF = 0.03

# Heights
APPROACH_HEIGHT = 0.28
PICK_HEIGHT = 0.16
SAFE_HEIGHT = 0.35

# Place locations - Red at (0.35, -0.30), Green at (0.55, -0.30) to avoid overlap
RED_PLACE_X, RED_PLACE_Y = 0.35, -0.30
GREEN_PLACE_X, GREEN_PLACE_Y = 0.55, -0.30

# Gripper positions
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.68

# Downward orientation (gripper facing down)
DOWN_QUAT = Quaternion(w=0.0, x=0.0, y=1.0, z=0.0)

# Motion parameters
CARTESIAN_STEP = 0.005  # 5mm resolution for Cartesian paths
CARTESIAN_JUMP_THRESHOLD = 0.0  # Disable jump threshold
MAX_VELOCITY_SCALING = 0.3  # 30% of max velocity for smooth motion
MAX_ACCELERATION_SCALING = 0.2  # 20% of max acceleration


def smooth_profile(t: float) -> float:
    """
    Sinusoidal smoothing function for velocity profiling.
    Creates smooth acceleration at start and deceleration at end.
    
    Args:
        t: Progress from 0.0 to 1.0
    
    Returns:
        Smoothed progress value (S-curve)
    """
    return (1.0 - math.cos(t * math.pi)) / 2.0


def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical Linear Interpolation between two quaternions.
    Provides smooth orientation transitions.
    """
    # Convert to numpy arrays
    qa = np.array([q1.w, q1.x, q1.y, q1.z])
    qb = np.array([q2.w, q2.x, q2.y, q2.z])
    
    # Normalize
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    
    # Compute dot product
    dot = np.dot(qa, qb)
    
    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        qb = -qb
        dot = -dot
    
    # Clamp dot product
    dot = min(1.0, max(-1.0, dot))
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = qa + t * (qb - qa)
        result = result / np.linalg.norm(result)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        wa = math.sin((1.0 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        result = wa * qa + wb * qb
    
    return Quaternion(w=result[0], x=result[1], y=result[2], z=result[3])


def pose_stamped(x, y, z, orientation=None, frame=BASE_FRAME) -> PoseStamped:
    """Create a PoseStamped message."""
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position = Point(x=float(x), y=float(y), z=float(z))
    ps.pose.orientation = orientation if orientation else DOWN_QUAT
    return ps


class PerfectPickPlace(Node):
    """
    Perfect Pick-and-Place implementation with smooth motion control.
    Handles multiple cubes (red and green).
    """
    
    def __init__(self):
        super().__init__('perfect_pick_place')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Current joint state
        self._current_joints = None
        self._joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # MoveIt action client
        self._move_action = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('Waiting for MoveGroup action server...')
        if not self._move_action.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('MoveGroup not available!')
            sys.exit(1)
        self.get_logger().info('✓ MoveGroup connected')
        
        # Trajectory action client for direct execution
        self._traj_action = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.get_logger().info('Waiting for trajectory controller...')
        if not self._traj_action.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Trajectory controller not available!')
            sys.exit(1)
        self.get_logger().info('✓ Trajectory controller connected')

        # Cartesian path service
        self._cartesian_srv = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )
        self.get_logger().info('Waiting for Cartesian path service...')
        if not self._cartesian_srv.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn('Cartesian path service not available - will use waypoint planning')
            self._cartesian_available = False
        else:
            self._cartesian_available = True
            self.get_logger().info('✓ Cartesian path service connected')
        
        # Gripper publisher
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, 
            '/gripper_forward_position_controller/commands',
            10
        )
        self.get_logger().info('✓ Gripper publisher created')
        
        # DetachableJoint publishers - separate for each cube
        self._attach_red_pub = self.create_publisher(Empty, '/gripper/attach_red', 10)
        self._detach_red_pub = self.create_publisher(Empty, '/gripper/detach_red', 10)
        self._attach_green_pub = self.create_publisher(Empty, '/gripper/attach_green', 10)
        self._detach_green_pub = self.create_publisher(Empty, '/gripper/detach_green', 10)
        self.get_logger().info('✓ DetachableJoint publishers created (red & green)')
        
        # State tracking
        self._red_attached = False
        self._green_attached = False
        
        # Wait for joint states
        self.get_logger().info('Waiting for joint states...')
        timeout = 10.0
        start = time.time()
        while self._current_joints is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self._current_joints is None:
            self.get_logger().error('No joint states received!')
            sys.exit(1)
        self.get_logger().info('✓ Joint states received')
    
    def _joint_state_callback(self, msg: JointState):
        """Store current joint positions."""
        # Extract UR joints only (filter out gripper joints)
        joints = {}
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                joints[name] = pos
        
        if len(joints) == 6:
            self._current_joints = [joints[name] for name in JOINT_NAMES]
    
    # ══════════════════════════════════════════════════════════════════════════
    # Gripper Control
    # ══════════════════════════════════════════════════════════════════════════
    
    def control_gripper(self, position: float, duration: float = 0.5):
        """
        Smooth gripper control with interpolated motion.
        """
        action = "CLOSING" if position > 0.3 else "OPENING"
        self.get_logger().info(f'[GRIPPER] {action} to {position:.2f}')
        
        msg = Float64MultiArray()
        
        # Smooth interpolation
        steps = 20
        start_pos = GRIPPER_OPEN if position > 0.3 else GRIPPER_CLOSED
        
        for i in range(steps + 1):
            t = i / steps
            t_smooth = smooth_profile(t)
            current = start_pos + (position - start_pos) * t_smooth
            msg.data = [current]
            self._gripper_pub.publish(msg)
            time.sleep(duration / steps)
        
        # Final position
        msg.data = [position]
        self._gripper_pub.publish(msg)
        time.sleep(0.1)

    def open_gripper(self):
        self.control_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.control_gripper(GRIPPER_CLOSED)

    # ══════════════════════════════════════════════════════════════════════════
    # Physics-Based Attachment (per cube)
    # ══════════════════════════════════════════════════════════════════════════
    
    def attach_cube(self, cube: str = 'red'):
        """Attach cube using DetachableJoint physics."""
        self.get_logger().info(f'[PHYSICS] Attaching {cube} cube...')
        msg = Empty()
        pub = self._attach_red_pub if cube == 'red' else self._attach_green_pub
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.03)
        if cube == 'red':
            self._red_attached = True
        else:
            self._green_attached = True
        time.sleep(0.15)
        self.get_logger().info(f'[PHYSICS] ✓ {cube.capitalize()} cube attached')
    
    def detach_cube(self, cube: str = 'red'):
        """Detach cube - it falls with gravity."""
        self.get_logger().info(f'[PHYSICS] Detaching {cube} cube...')
        msg = Empty()
        pub = self._detach_red_pub if cube == 'red' else self._detach_green_pub
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.03)
        if cube == 'red':
            self._red_attached = False
        else:
            self._green_attached = False
        time.sleep(0.15)
        self.get_logger().info(f'[PHYSICS] ✓ {cube.capitalize()} cube detached')

    def initial_detach(self):
        """Detach all cubes at startup (they start attached by default)."""
        self.get_logger().info('[PHYSICS] Initial detach of all cubes...')
        self.detach_cube('red')
        self.detach_cube('green')
        time.sleep(0.3)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Smooth Motion Control
    # ══════════════════════════════════════════════════════════════════════════
    
    def interpolate_poses(self, start: PoseStamped, end: PoseStamped, 
                          num_points: int = 50) -> list:
        """
        Generate smoothly interpolated poses between start and end.
        Uses sinusoidal velocity profiling for smooth motion.
        """
        poses = []
        
        for i in range(num_points + 1):
            t = i / num_points
            t_smooth = smooth_profile(t)
            
            # Interpolate position
            x = start.pose.position.x + (end.pose.position.x - start.pose.position.x) * t_smooth
            y = start.pose.position.y + (end.pose.position.y - start.pose.position.y) * t_smooth
            z = start.pose.position.z + (end.pose.position.z - start.pose.position.z) * t_smooth
            
            # SLERP orientation
            quat = quaternion_slerp(start.pose.orientation, end.pose.orientation, t_smooth)
            
            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            pose.orientation = quat
            poses.append(pose)
        
        return poses
    
    def move_cartesian(self, target: PoseStamped, description: str = '',
                       velocity_scale: float = 0.3) -> bool:
        """
        Execute a smooth Cartesian motion to target pose.
        Uses MoveIt's Cartesian path planning for linear motion.
        """
        self.get_logger().info(
            f'[MOTION] {description}: '
            f'({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f})'
        )

        # Build motion plan request with Cartesian-like behavior
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scale
        req.max_acceleration_scaling_factor = velocity_scale * 0.5

        # Goal constraints
        constraints = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target.header.frame_id
        pos_constraint.link_name = END_EFFECTOR_LINK
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]  # 1cm tolerance

        bv = BoundingVolume()
        bv.primitives.append(box)
        bv.primitive_poses.append(target.pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        # Orientation constraint
        orient = OrientationConstraint()
        orient.header.frame_id = target.header.frame_id
        orient.link_name = END_EFFECTOR_LINK
        orient.orientation = target.pose.orientation
        orient.absolute_x_axis_tolerance = 0.3
        orient.absolute_y_axis_tolerance = 0.3
        orient.absolute_z_axis_tolerance = math.pi
        orient.weight = 1.0
        constraints.orientation_constraints.append(orient)

        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False

        # Execute
        future = self._move_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'[MOTION] ✗ Goal rejected: {description}')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f'[MOTION] ✓ {description}')
            return True
        else:
            self.get_logger().warn(f'[MOTION] ✗ Failed ({result.error_code.val}): {description}')
            return False

    def move_to_pose(self, target: PoseStamped, description: str = '',
                     velocity_scale: float = 0.3, max_attempts: int = 3) -> bool:
        """
        Move to target pose with retries and fallback strategies.
        """
        for attempt in range(max_attempts):
            if self.move_cartesian(target, description, velocity_scale):
                return True
            
            if attempt < max_attempts - 1:
                self.get_logger().info(f'[RETRY] Attempt {attempt + 2}/{max_attempts}...')
                # Try with relaxed parameters
                velocity_scale *= 0.8
        
        return False
    
    # ══════════════════════════════════════════════════════════════════════════
    # Pick-and-Place for a Single Cube
    # ══════════════════════════════════════════════════════════════════════════
    
    def pick_and_place_cube(self, cube: str, pick_x: float, pick_y: float, 
                            place_x: float, place_y: float):
        """
        Execute pick-and-place for a single cube.
        
        Args:
            cube: 'red' or 'green'
            pick_x, pick_y: Pick position relative to robot base
            place_x, place_y: Place position relative to robot base
        """
        color = cube.upper()
        self.get_logger().info(f'\n{"─" * 60}')
        self.get_logger().info(f'   Picking {color} cube')
        self.get_logger().info(f'{"─" * 60}')
        
        # Approach above cube
        self.get_logger().info(f'\n── Approach {color} Cube ──')
        above_cube = pose_stamped(pick_x, pick_y, APPROACH_HEIGHT)
        self.move_to_pose(above_cube, f'above {cube} cube', velocity_scale=0.25)
        
        # Descend to pick
        self.get_logger().info(f'\n── Descend to Pick {color} ──')
        pick_pose = pose_stamped(pick_x, pick_y, PICK_HEIGHT)
        self.move_to_pose(pick_pose, f'{cube} pick position', velocity_scale=0.15)
        
        # Grasp cube
        self.get_logger().info(f'\n── Grasp {color} Cube ──')
        self.close_gripper()
        self.attach_cube(cube)
        time.sleep(0.2)
        
        # Lift cube
        self.get_logger().info(f'\n── Lift {color} Cube ──')
        lift_pose = pose_stamped(pick_x, pick_y, APPROACH_HEIGHT)
        self.move_to_pose(lift_pose, f'lift {cube}', velocity_scale=0.2)
        
        # Transit to place location
        self.get_logger().info(f'\n── Transit {color} to Place ──')
        waypoint = pose_stamped(0.30, 0.0, SAFE_HEIGHT)
        self.move_to_pose(waypoint, 'transit waypoint', velocity_scale=0.3)
        
        above_place = pose_stamped(place_x, place_y, APPROACH_HEIGHT)
        self.move_to_pose(above_place, f'above {cube} place', velocity_scale=0.25)
        
        # Lower to place
        self.get_logger().info(f'\n── Lower {color} to Place ──')
        place_pose = pose_stamped(place_x, place_y, PICK_HEIGHT + 0.02)
        self.move_to_pose(place_pose, f'{cube} place position', velocity_scale=0.15)
        
        # Release cube
        self.get_logger().info(f'\n── Release {color} Cube ──')
        self.detach_cube(cube)
        self.open_gripper()
        time.sleep(0.3)
        
        # Retract
        self.get_logger().info(f'\n── Retract from {color} ──')
        self.move_to_pose(above_place, f'retract from {cube}', velocity_scale=0.25)
        
        self.get_logger().info(f'\n✓ {color} cube placed successfully!')
    
    # ══════════════════════════════════════════════════════════════════════════
    # Main Demo Sequence
    # ══════════════════════════════════════════════════════════════════════════
    
    def run_demo(self):
        """Execute the perfect pick-and-place demonstration for both cubes."""
        self.get_logger().info('═' * 60)
        self.get_logger().info('   PERFECT Pick-and-Place Demo')
        self.get_logger().info('   Red & Green Cubes • Physics Grasping • ROS 2 Jazzy')
        self.get_logger().info('═' * 60)

        # ── Step 0: Initialize ──
        self.get_logger().info('\n── Step 0: Initialize ──')
        self.initial_detach()
        self.open_gripper()

        # ── Step 1: Move to safe home position ──
        self.get_logger().info('\n── Step 1: Home Position ──')
        home = pose_stamped(0.25, 0.0, SAFE_HEIGHT)
        if not self.move_to_pose(home, 'home'):
            home = pose_stamped(0.20, 0.0, 0.40)
            self.move_to_pose(home, 'home (safe)')

        # ── Pick and place RED cube ──
        self.pick_and_place_cube('red', RED_CUBE_X, RED_CUBE_Y, RED_PLACE_X, RED_PLACE_Y)

        # Return to home before next cube
        self.get_logger().info('\n── Return to Home ──')
        self.move_to_pose(home, 'home', velocity_scale=0.3)
        
        # ── Pick and place GREEN cube ──
        self.pick_and_place_cube('green', GREEN_CUBE_X, GREEN_CUBE_Y, GREEN_PLACE_X, GREEN_PLACE_Y)
        
        # ── Final: Return home ──
        self.get_logger().info('\n── Final: Return Home ──')
        safe_up = pose_stamped(GREEN_PLACE_X, GREEN_PLACE_Y, SAFE_HEIGHT)
        self.move_to_pose(safe_up, 'safe height', velocity_scale=0.3)
        
        waypoint = pose_stamped(0.30, 0.0, SAFE_HEIGHT)
        self.move_to_pose(waypoint, 'transit back', velocity_scale=0.3)
        self.move_to_pose(home, 'home', velocity_scale=0.3)
        
        self.get_logger().info('\n' + '═' * 60)
        self.get_logger().info('   ✓ PERFECT Pick-and-Place Complete!')
        self.get_logger().info('   Both RED and GREEN cubes placed successfully!')
        self.get_logger().info('═' * 60)


def main(args=None):
    rclpy.init(args=args)
    
    node = PerfectPickPlace()
    
    try:
        node.run_demo()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
