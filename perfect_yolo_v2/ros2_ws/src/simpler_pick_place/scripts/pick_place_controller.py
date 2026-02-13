#!/usr/bin/env python3
"""
Pick-and-Place Controller for Perfect V2

This controller subscribes to cube poses from the Scene Manager and performs
pick-and-place operations based on the received positions.

Key differences from V1:
  - Does NOT have hardcoded cube positions
  - Waits for scene_manager to publish cube locations
  - Uses service call or topic subscription to get cube data
  - Can handle dynamically spawned objects

Author: Perfect V2 Architecture
"""

import sys
import time
import math
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray, Empty, String
from std_srvs.srv import Trigger
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
)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
PLANNING_GROUP = 'ur_manipulator'
END_EFFECTOR_LINK = 'tool0'
BASE_FRAME = 'base_link'

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint', 
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Gripper positions
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.68

# Downward orientation (gripper facing down)
DOWN_QUAT = Quaternion(w=0.0, x=0.0, y=1.0, z=0.0)


def smooth_profile(t: float) -> float:
    """Sinusoidal smoothing function for velocity profiling."""
    return (1.0 - math.cos(t * math.pi)) / 2.0


def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Spherical Linear Interpolation between two quaternions."""
    qa = np.array([q1.w, q1.x, q1.y, q1.z])
    qb = np.array([q2.w, q2.x, q2.y, q2.z])
    
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    
    dot = np.dot(qa, qb)
    if dot < 0:
        qb = -qb
        dot = -dot
    
    dot = min(1.0, max(-1.0, dot))
    
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


class CubeTarget:
    """Represents a cube to pick and place."""
    def __init__(self, color: str, pick_x: float, pick_y: float, 
                 place_x: float, place_y: float):
        self.color = color
        self.pick_x = pick_x
        self.pick_y = pick_y
        self.place_x = place_x
        self.place_y = place_y


class PickPlaceController(Node):
    """
    Pick-and-Place Controller that gets cube positions from scene_manager.
    """
    
    def __init__(self):
        super().__init__('pick_place_controller')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # ══════════════════════════════════════════════════════════════════════
        # Scene data from scene_manager
        # ══════════════════════════════════════════════════════════════════════
        self._cubes = []  # List of CubeTarget
        self._heights = {
            'approach': 0.28,
            'pick': 0.16,
            'place': 0.18,
            'safe': 0.35
        }
        self._scene_ready = False
        
        # Subscribe to scene status
        self._scene_status_sub = self.create_subscription(
            String,
            '/scene/status',
            self._scene_status_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to cube poses
        self._cube_poses_sub = self.create_subscription(
            String,
            '/scene/cube_poses',
            self._cube_poses_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Service client to get cubes
        self._get_cubes_client = self.create_client(
            Trigger,
            '/scene/get_cubes',
            callback_group=self.callback_group
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Robot state
        # ══════════════════════════════════════════════════════════════════════
        self._current_joints = None
        self._joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # MoveIt and control clients
        # ══════════════════════════════════════════════════════════════════════
        self._move_action = ActionClient(self, MoveGroup, 'move_action')
        self._traj_action = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self._cartesian_srv = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        
        # Gripper publisher
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, 
            '/gripper_forward_position_controller/commands',
            10
        )
        
        # DetachableJoint publishers
        self._attach_red_pub = self.create_publisher(Empty, '/gripper/attach_red', 10)
        self._detach_red_pub = self.create_publisher(Empty, '/gripper/detach_red', 10)
        self._attach_green_pub = self.create_publisher(Empty, '/gripper/attach_green', 10)
        self._detach_green_pub = self.create_publisher(Empty, '/gripper/detach_green', 10)
        
        self.get_logger().info('Pick-Place Controller initializing...')
    
    def _scene_status_callback(self, msg: String):
        """Handle scene status updates."""
        if msg.data == 'ready':
            self._scene_ready = True
            self.get_logger().info('[SCENE] Scene manager reports ready')
    
    def _cube_poses_callback(self, msg: String):
        """Handle cube pose updates from scene manager."""
        try:
            data = json.loads(msg.data)
            
            # Update heights
            if 'heights' in data:
                self._heights.update(data['heights'])
            
            # Update cube list
            self._cubes = []
            for cube_info in data.get('cubes', []):
                if not cube_info.get('is_placed', False):
                    cube = CubeTarget(
                        color=cube_info['color'],
                        pick_x=cube_info['pick']['x'],
                        pick_y=cube_info['pick']['y'],
                        place_x=cube_info['place']['x'],
                        place_y=cube_info['place']['y']
                    )
                    self._cubes.append(cube)
            
            self.get_logger().debug(f'[SCENE] Received {len(self._cubes)} cube(s)')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse cube poses: {e}')
    
    def _joint_state_callback(self, msg: JointState):
        """Store current joint positions."""
        joints = {}
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                joints[name] = pos
        
        if len(joints) == 6:
            self._current_joints = [joints[name] for name in JOINT_NAMES]
    
    def wait_for_scene(self, timeout: float = 30.0) -> bool:
        """Wait for scene manager to be ready."""
        self.get_logger().info('Waiting for scene manager...')
        
        start = time.time()
        while not self._scene_ready and (time.time() - start) < timeout:
            # Try service call
            if self._get_cubes_client.wait_for_service(timeout_sec=1.0):
                future = self._get_cubes_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                if future.result() and future.result().success:
                    self._scene_ready = True
                    # Parse the cube data
                    self._cube_poses_callback(String(data=future.result().message))
                    break
            rclpy.spin_once(self, timeout_sec=0.5)
        
        if self._scene_ready:
            self.get_logger().info('✓ Scene manager ready')
            self.get_logger().info(f'  Found {len(self._cubes)} cube(s) to pick')
            for cube in self._cubes:
                self.get_logger().info(
                    f'  • {cube.color.upper()}: pick=({cube.pick_x}, {cube.pick_y}) → '
                    f'place=({cube.place_x}, {cube.place_y})'
                )
            return True
        else:
            self.get_logger().error('Scene manager not available!')
            return False
    
    def wait_for_controllers(self, timeout: float = 30.0) -> bool:
        """Wait for all controllers to be ready."""
        self.get_logger().info('Waiting for controllers...')
        
        # MoveIt
        if not self._move_action.wait_for_server(timeout_sec=timeout):
            self.get_logger().error('MoveGroup not available!')
            return False
        self.get_logger().info('✓ MoveGroup connected')
        
        # Trajectory controller
        if not self._traj_action.wait_for_server(timeout_sec=timeout):
            self.get_logger().error('Trajectory controller not available!')
            return False
        self.get_logger().info('✓ Trajectory controller connected')
        
        # Wait for joint states
        start = time.time()
        while self._current_joints is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self._current_joints is None:
            self.get_logger().error('No joint states received!')
            return False
        self.get_logger().info('✓ Joint states received')
        
        return True
    
    # ══════════════════════════════════════════════════════════════════════════
    # Gripper Control
    # ══════════════════════════════════════════════════════════════════════════
    
    def control_gripper(self, position: float, duration: float = 0.5):
        """Smooth gripper control."""
        action = "CLOSING" if position > 0.3 else "OPENING"
        self.get_logger().info(f'[GRIPPER] {action} to {position:.2f}')
        
        msg = Float64MultiArray()
        steps = 20
        start_pos = GRIPPER_OPEN if position > 0.3 else GRIPPER_CLOSED
        
        for i in range(steps + 1):
            t = i / steps
            t_smooth = smooth_profile(t)
            current = start_pos + (position - start_pos) * t_smooth
            msg.data = [current]
            self._gripper_pub.publish(msg)
            time.sleep(duration / steps)
        
        msg.data = [position]
        self._gripper_pub.publish(msg)
        time.sleep(0.1)

    def open_gripper(self):
        self.control_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.control_gripper(GRIPPER_CLOSED)

    # ══════════════════════════════════════════════════════════════════════════
    # Physics Attachment
    # ══════════════════════════════════════════════════════════════════════════
    
    def attach_cube(self, color: str):
        """Attach cube using DetachableJoint."""
        self.get_logger().info(f'[PHYSICS] Attaching {color} cube...')
        msg = Empty()
        pub = self._attach_red_pub if color == 'red' else self._attach_green_pub
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.03)
        time.sleep(0.15)
        self.get_logger().info(f'[PHYSICS] ✓ {color.capitalize()} cube attached')
    
    def detach_cube(self, color: str):
        """Detach cube."""
        self.get_logger().info(f'[PHYSICS] Detaching {color} cube...')
        msg = Empty()
        pub = self._detach_red_pub if color == 'red' else self._detach_green_pub
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.03)
        time.sleep(0.15)
        self.get_logger().info(f'[PHYSICS] ✓ {color.capitalize()} cube detached')

    def initial_detach(self):
        """Detach all cubes at startup."""
        self.get_logger().info('[PHYSICS] Initial detach of all cubes...')
        self.detach_cube('red')
        self.detach_cube('green')
        time.sleep(0.3)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Motion Control
    # ══════════════════════════════════════════════════════════════════════════
    
    def move_to_pose(self, target: PoseStamped, description: str = '',
                     velocity_scale: float = 0.3, max_attempts: int = 3) -> bool:
        """Move to target pose with retries."""
        for attempt in range(max_attempts):
            if self._execute_move(target, description, velocity_scale):
                return True
            
            if attempt < max_attempts - 1:
                self.get_logger().info(f'[RETRY] Attempt {attempt + 2}/{max_attempts}...')
                velocity_scale *= 0.8
                time.sleep(0.5)
        
        return False
    
    def _execute_move(self, target: PoseStamped, description: str,
                      velocity_scale: float) -> bool:
        """Execute a single move attempt."""
        self.get_logger().info(
            f'[MOTION] {description}: '
            f'({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f})'
        )

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scale
        req.max_acceleration_scaling_factor = velocity_scale * 0.5

        constraints = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target.header.frame_id
        pos_constraint.link_name = END_EFFECTOR_LINK
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]

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
    
    # ══════════════════════════════════════════════════════════════════════════
    # Pick-and-Place Logic
    # ══════════════════════════════════════════════════════════════════════════
    
    def pick_and_place_cube(self, cube: CubeTarget):
        """Execute pick-and-place for a single cube using data from scene manager."""
        color = cube.color.upper()
        
        self.get_logger().info(f'\n{"─" * 60}')
        self.get_logger().info(f'   Picking {color} cube')
        self.get_logger().info(f'   From: ({cube.pick_x}, {cube.pick_y}) → To: ({cube.place_x}, {cube.place_y})')
        self.get_logger().info(f'{"─" * 60}')
        
        approach_h = self._heights['approach']
        pick_h = self._heights['pick']
        place_h = self._heights['place']
        safe_h = self._heights['safe']
        
        # Approach above cube
        self.get_logger().info(f'\n── Approach {color} Cube ──')
        above_cube = pose_stamped(cube.pick_x, cube.pick_y, approach_h)
        self.move_to_pose(above_cube, f'above {cube.color} cube', velocity_scale=0.25)
        
        # Descend to pick
        self.get_logger().info(f'\n── Descend to Pick {color} ──')
        pick_pose = pose_stamped(cube.pick_x, cube.pick_y, pick_h)
        self.move_to_pose(pick_pose, f'{cube.color} pick position', velocity_scale=0.15)
        
        # Grasp cube
        self.get_logger().info(f'\n── Grasp {color} Cube ──')
        self.close_gripper()
        self.attach_cube(cube.color)
        time.sleep(0.2)
        
        # Lift cube
        self.get_logger().info(f'\n── Lift {color} Cube ──')
        lift_pose = pose_stamped(cube.pick_x, cube.pick_y, approach_h)
        self.move_to_pose(lift_pose, f'lift {cube.color}', velocity_scale=0.2)
        
        # Transit to place location
        self.get_logger().info(f'\n── Transit {color} to Place ──')
        waypoint = pose_stamped(0.30, 0.0, safe_h)
        self.move_to_pose(waypoint, 'transit waypoint', velocity_scale=0.3)
        
        above_place = pose_stamped(cube.place_x, cube.place_y, approach_h)
        self.move_to_pose(above_place, f'above {cube.color} place', velocity_scale=0.25)
        
        # Lower to place
        self.get_logger().info(f'\n── Lower {color} to Place ──')
        place_pose = pose_stamped(cube.place_x, cube.place_y, place_h)
        self.move_to_pose(place_pose, f'{cube.color} place position', velocity_scale=0.15)
        
        # Release cube
        self.get_logger().info(f'\n── Release {color} Cube ──')
        self.detach_cube(cube.color)
        self.open_gripper()
        time.sleep(0.3)
        
        # Retract
        self.get_logger().info(f'\n── Retract from {color} ──')
        self.move_to_pose(above_place, f'retract from {cube.color}', velocity_scale=0.25)
        
        self.get_logger().info(f'\n✓ {color} cube placed successfully!')
    
    # ══════════════════════════════════════════════════════════════════════════
    # Main Execution
    # ══════════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Main execution loop."""
        # Wait for scene manager
        if not self.wait_for_scene():
            return False
        
        # Wait for controllers
        if not self.wait_for_controllers():
            return False
        
        # Check if we have cubes to pick
        if not self._cubes:
            self.get_logger().warn('No cubes to pick!')
            return False
        
        self.get_logger().info('═' * 60)
        self.get_logger().info('   Pick-and-Place Controller V2')
        self.get_logger().info('   Dynamic Cube Detection • ROS 2 Jazzy')
        self.get_logger().info('═' * 60)
        
        # Initialize
        self.get_logger().info('\n── Initialize ──')
        # Note: Removed initial_detach() - cubes are not attached at startup
        # and sending detach commands was causing physics instability
        self.open_gripper()
        
        # Move to home
        self.get_logger().info('\n── Home Position ──')
        home = pose_stamped(0.25, 0.0, self._heights['safe'])
        self.move_to_pose(home, 'home')
        
        # Pick and place each cube
        for cube in self._cubes:
            self.pick_and_place_cube(cube)
            
            # Return to home between cubes
            self.get_logger().info('\n── Return to Home ──')
            self.move_to_pose(home, 'home', velocity_scale=0.3)
        
        # Final home
        self.get_logger().info('\n── Final Position ──')
        self.move_to_pose(home, 'final home', velocity_scale=0.3)
        
        self.get_logger().info('\n' + '═' * 60)
        self.get_logger().info('   ✓ Pick-and-Place Complete!')
        self.get_logger().info(f'   Processed {len(self._cubes)} cube(s)')
        self.get_logger().info('═' * 60)
        
        return True


def main(args=None):
    rclpy.init(args=args)
    
    node = PickPlaceController()
    
    try:
        success = node.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
