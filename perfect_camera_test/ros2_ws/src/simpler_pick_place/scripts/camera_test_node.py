#!/usr/bin/env python3
"""
Camera Test Node - Simple color-based detection to verify camera calibration.

This node:
1. Uses ONLY color segmentation (no YOLO) to detect red and green cubes
2. Logs detailed debug info for coordinate transformation verification
3. Compares detected positions with known Gazebo positions

Known cube positions in Gazebo (from pick_place.sdf):
- Red cube:   World=(0.4, -0.35, 0.77)  → Robot frame=(0.4, 0.15)
- Green cube: World=(0.4, -0.60, 0.77)  → Robot frame=(0.4, -0.10)

Camera position: (0.4, -0.5, 1.5) with pitch=90 degrees (looking down)
Robot base:      (0.0, -0.5, 0.74)
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import json
import subprocess
import re


class CameraTestNode(Node):
    """
    Simple camera test node using only color segmentation.
    """
    
    def __init__(self):
        super().__init__('camera_test_node')
        
        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        
        # ══════════════════════════════════════════════════════════════════════
        # Camera Configuration (from pick_place.sdf)
        # Camera pose: <pose>0.4 -0.5 1.5 0 1.5708 0</pose>
        # This means: position=(0.4, -0.5, 1.5), orientation=(roll=0, pitch=90deg, yaw=0)
        # ══════════════════════════════════════════════════════════════════════
        self.camera_position = np.array([0.4, -0.5, 1.5])
        self.camera_pitch = 1.5708  # 90 degrees in radians
        
        # Robot base position in world frame
        self.robot_base_world = np.array([0.0, -0.5, 0.74])
        
        # Known cube positions (for comparison)
        self.known_cubes = {
            'red': {'world': (0.4, -0.35, 0.77), 'robot': (0.4, 0.15)},
            'green': {'world': (0.4, -0.60, 0.77), 'robot': (0.4, -0.10)},
        }
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.0
        self.cy = 240.0
        self.camera_info_received = False
        
        # Image dimensions
        self.img_width = 640
        self.img_height = 480
        
        # ══════════════════════════════════════════════════════════════════════
        # HSV Color Ranges
        # ══════════════════════════════════════════════════════════════════════
        # Red color (wraps around in HSV)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Green color
        self.green_lower = np.array([35, 100, 100])
        self.green_upper = np.array([85, 255, 255])
        
        # ══════════════════════════════════════════════════════════════════════
        # Image Storage
        # ══════════════════════════════════════════════════════════════════════
        self.latest_rgb = None
        self.latest_depth = None
        
        # ══════════════════════════════════════════════════════════════════════
        # Subscribers
        # ══════════════════════════════════════════════════════════════════════
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self._rgb_sub = self.create_subscription(
            Image, '/camera/image', self._rgb_callback, sensor_qos,
            callback_group=self.callback_group
        )
        
        self._depth_sub = self.create_subscription(
            Image, '/camera/depth_image', self._depth_callback, sensor_qos,
            callback_group=self.callback_group
        )
        
        self._camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._camera_info_callback, 10,
            callback_group=self.callback_group
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Publishers
        # ══════════════════════════════════════════════════════════════════════
        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._detection_pub = self.create_publisher(String, '/detected_cubes', latching_qos)
        
        # ══════════════════════════════════════════════════════════════════════
        # Detection Timer
        # ══════════════════════════════════════════════════════════════════════
        self._detection_timer = self.create_timer(
            2.0,  # Every 2 seconds for detailed logging
            self._run_detection,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('═' * 70)
        self.get_logger().info('  Camera Test Node - Color-Only Detection')
        self.get_logger().info('═' * 70)
        self.get_logger().info(f'  Camera position: {self.camera_position}')
        self.get_logger().info(f'  Camera pitch: {np.degrees(self.camera_pitch):.1f} degrees')
        self.get_logger().info(f'  Robot base: {self.robot_base_world}')
        self.get_logger().info('═' * 70)
        self.get_logger().info('Waiting for camera images...')
    
    def _rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
    
    def _depth_callback(self, msg: Image):
        try:
            if msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.latest_depth = depth_mm.astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')
    
    def _camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.img_width = msg.width
            self.img_height = msg.height
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, cy={self.cy:.1f}, size={self.img_width}x{self.img_height}'
            )
    
    def _pixel_to_world_v1(self, px: float, py: float, depth: float) -> tuple:
        """
        Original transformation (from vision_node.py).
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Convert pixel to camera frame
        cam_x = (px - self.cx) / self.fx * depth
        cam_y = (py - self.cy) / self.fy * depth
        cam_z = depth
        
        # Camera is rotated 90 degrees down (pitch = 90 deg)
        # Original transformation:
        world_x = self.camera_position[0] + cam_y
        world_y = self.camera_position[1] - cam_x
        world_z = self.camera_position[2] - cam_z
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _pixel_to_world_v2(self, px: float, py: float, depth: float) -> tuple:
        """
        Alternative transformation - trying different axis mapping.
        
        Camera looking down (pitch=90):
        - Camera +Z points towards table (world -Z)
        - Camera +X points to image right (world +X? or +Y?)
        - Camera +Y points to image down (world +Y? or +X?)
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Convert pixel to camera frame (3D point in camera coordinates)
        # In camera frame: X=right, Y=down, Z=forward (depth)
        cam_x = (px - self.cx) / self.fx * depth
        cam_y = (py - self.cy) / self.fy * depth
        cam_z = depth
        
        # Camera at (0.4, -0.5, 1.5) looking down (pitch=90)
        # When pitch=90, the camera's Z axis points to world -Z
        # Camera X axis -> World X (but may be negated)
        # Camera Y axis -> World Y (but need to account for pitch)
        
        # Alternative transformation:
        # The object position in world frame relative to camera is:
        # - Along world X: camera X offset (cam_x stays in X)
        # - Along world Y: camera Y offset (cam_y maps to Y)
        # - Along world Z: camera height minus depth
        
        world_x = self.camera_position[0] + cam_x  # cam_x -> world X offset
        world_y = self.camera_position[1] + cam_y  # cam_y -> world Y offset
        world_z = self.camera_position[2] - cam_z
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _pixel_to_world_v3(self, px: float, py: float, depth: float) -> tuple:
        """
        Using proper rotation matrix for pitch=90 degrees.
        
        For a camera with pitch=90 (looking down):
        R_pitch = [[1, 0, 0], [0, cos(90), -sin(90)], [0, sin(90), cos(90)]]
               = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        
        So: world_point = camera_pos + R_pitch @ camera_point
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Convert pixel to camera frame
        cam_x = (px - self.cx) / self.fx * depth
        cam_y = (py - self.cy) / self.fy * depth
        cam_z = depth
        
        # Rotation matrix for pitch = 90 degrees (pi/2)
        # R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        # For theta = 90 deg: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        # This rotates camera Z to point down (world -Z)
        
        # Apply rotation: world = R @ camera
        world_dx = cam_z   # camera Z -> world X offset
        world_dy = cam_y   # camera Y -> world Y offset  
        world_dz = -cam_x  # camera X -> world -Z offset
        
        world_x = self.camera_position[0] + world_dx
        world_y = self.camera_position[1] + world_dy
        world_z = self.camera_position[2] + world_dz
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _pixel_to_world_v4(self, px: float, py: float, depth: float) -> tuple:
        """
        Empirically derived transformation based on test data.
        
        From test results:
        - Red:   pixel(535, 276) → robot(0.400, 0.150), depth=0.758
        - Green: pixel(535, 65)  → robot(0.400, -0.100), depth=0.758
        
        Camera is at (0.4, -0.5, 1.5) looking down.
        Image center is (320, 240).
        
        Observations:
        - Both cubes have same px=535, same robot_x=0.4
        - px > cx means object is to the RIGHT of camera center
        - Camera is looking straight down, so px offset should affect world Y
        
        Let's work backwards:
        - For camera at (0.4, -0.5, 1.5), robot base at (0, -0.5, 0.74)
        - World X of cube = 0.4 → robot X = 0.4
        - World Y of red = -0.35 → robot Y = -0.35 - (-0.5) = 0.15
        - World Y of green = -0.60 → robot Y = -0.60 - (-0.5) = -0.10
        
        The camera looks down with pitch=90. In Gazebo's convention:
        - Camera +X points in the direction of world +Y (after pitch rotation)
        - Camera +Y points in the direction of world -X (after pitch rotation)
        - Camera +Z (depth) points in the direction of world -Z
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Convert pixel to camera frame offsets
        cam_x = (px - self.cx) / self.fx * depth  # Offset in camera X
        cam_y = (py - self.cy) / self.fy * depth  # Offset in camera Y
        
        # For a camera pitched 90 degrees down:
        # Camera frame has: X=right, Y=down, Z=forward (towards ground)
        # After 90 degree pitch around Y axis:
        # - Camera Z (forward) -> World -Z (down)
        # - Camera X (right) -> World +Y (for camera at 0 yaw)
        # - Camera Y (down in image) -> World -X
        
        # But wait - we need to check the Gazebo convention for the RGBD camera sensor
        # In Gazebo, sensor frame typically has Z forward, X right, Y down (optical frame)
        
        # Looking at actual data:
        # px=535 > cx=320, so cam_x > 0 (point is to the right)
        # Red has py=276 > cy=240, so cam_y > 0 (point is below center)
        # Green has py=65 < cy=240, so cam_y < 0 (point is above center)
        
        # Red is at robot_y=0.15 (more positive Y)
        # Green is at robot_y=-0.10 (more negative Y)
        # So higher py (more "down" in image) → more positive robot Y
        
        # This means: cam_y maps to robot Y (with sign change accounted for)
        # cam_y > 0 → more positive robot Y
        
        # For world coordinates:
        # When camera looks down, image "down" (positive cam_y) points to... 
        # which direction in world? Let's trace it:
        # - Camera at (0.4, -0.5, 1.5) with pitch=90
        # - Image center (320, 240) points at camera position's XY = (0.4, -0.5)
        # - Image "down" direction (increasing py) points towards positive world Y
        
        # So the transform should be:
        world_x = self.camera_position[0] - cam_y  # cam_y (image down) -> world -X? Let's test
        world_y = self.camera_position[1] + cam_x  # cam_x (image right) -> world +Y
        world_z = self.camera_position[2] - depth
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _pixel_to_world_v5(self, px: float, py: float, depth: float) -> tuple:
        """
        Corrected empirical fit.
        
        From data:
        - Red:   pixel(535, 276) → robot(0.400, 0.150)
        - Green: pixel(535, 65)  → robot(0.400, -0.100)
        
        Key insight: both cubes have px=535 but robot_x=0.4 (same).
        This means px does NOT map to robot_x significantly.
        
        py varies (276 vs 65), and robot_y varies (0.15 vs -0.10).
        Higher py → higher robot_y (positive correlation).
        
        Camera at (0.4, -0.5) looking straight down.
        Robot base at (0, -0.5).
        
        So image "down" (increasing py) points towards world +Y.
        And image "right" (increasing px) points towards world +X.
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Scale factor from pixel to world
        scale = depth / self.fx
        
        # Compute offsets from image center
        dx_pixel = px - self.cx  # Horizontal offset (positive = right in image)
        dy_pixel = py - self.cy  # Vertical offset (positive = down in image)
        
        # Convert to world offsets:
        # Image right (positive dx) → World +X
        # Image down (positive dy) → World +Y
        world_x = self.camera_position[0] + dx_pixel * scale
        world_y = self.camera_position[1] + dy_pixel * scale
        world_z = self.camera_position[2] - depth
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _pixel_to_world_v6(self, px: float, py: float, depth: float) -> tuple:
        """
        Empirically calibrated transformation.
        
        From data:
        - Red:   pixel(535, 276) → robot(0.400, 0.150), depth=0.758
        - Green: pixel(535, 65)  → robot(0.400, -0.100), depth=0.758
        
        Both cubes have px=535 and robot_x=0.4, so we can derive:
        - When px=535, robot_x=0.4
        - Using same scale as Y: scale = 0.001185
        - robot_x = camera_x + (px - cx_adjusted) * scale
        - 0.4 = 0.4 + (535 - cx_adjusted) * 0.001185
        - 0 = (535 - cx_adjusted) * 0.001185
        - cx_adjusted = 535 (the effective center for X is at px=535!)
        
        Wait, this suggests the camera's optical center is offset.
        Actually, if both cubes at world_x=0.4 appear at px=535,
        it means the optical center (in world X) is at px=535.
        
        Let's reconsider: maybe the camera pose in SDF has rotation we missed.
        The pose is: <pose>0.4 -0.5 1.5 0 1.5708 0</pose>
        This is: x=0.4, y=-0.5, z=1.5, roll=0, pitch=1.5708, yaw=0
        
        With pitch=90 degrees (1.5708 rad), camera looks down.
        But with yaw=0, the camera's "right" direction (positive px) 
        should point to world +X.
        
        The fact that objects at world_x=0.4 appear at px=535 suggests
        either the camera intrinsics are different, or there's a yaw.
        
        For now, let's use a fully empirical approach:
        - Use the observed pixel coordinates to derive the mapping
        - Both cubes at px=535 have robot_x=0.4
        - This establishes: robot_x = f(px) where f(535) = 0.4
        
        For Y (already working):
        - robot_y = 0.001185 * py - 0.177
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Empirically derived linear mapping for Y
        # From: py=276 → robot_y=0.15 and py=65 → robot_y=-0.10
        robot_y = 0.001185 * py - 0.177
        
        # For X: We only have one data point (px=535 → robot_x=0.4)
        # We'll assume the same scale as Y and derive the offset
        # robot_x = scale * px + offset
        # 0.4 = 0.001185 * 535 + offset
        # 0.4 = 0.634 + offset
        # offset = 0.4 - 0.634 = -0.234
        # So: robot_x = 0.001185 * px - 0.234
        
        robot_x = 0.001185 * px - 0.234
        
        world_z = self.camera_position[2] - depth
        
        return (robot_x, robot_y, world_z)
    
    def _get_gazebo_cube_poses(self):
        """Query Gazebo for actual cube positions."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t', '/world/pick_place_world/dynamic_pose/info', '-n', '1'],
                capture_output=True, text=True, timeout=2.0
            )
            
            if result.returncode != 0:
                return {}
            
            poses = {}
            output = result.stdout
            pose_blocks = output.split('pose {')
            
            for block in pose_blocks:
                name_match = re.search(r'name:\s*"([^"]+)"', block)
                if name_match:
                    name = name_match.group(1)
                    if name in ['red_cube', 'green_cube']:
                        pos_match = re.search(
                            r'position\s*\{[^}]*x:\s*([-\d.e]+)[^}]*y:\s*([-\d.e]+)[^}]*z:\s*([-\d.e]+)',
                            block, re.DOTALL
                        )
                        if pos_match:
                            x, y, z = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                            color = 'red' if 'red' in name else 'green'
                            poses[color] = {'world': (x, y, z), 'robot': (x, y + 0.5)}
            
            return poses
        except Exception as e:
            return {}
    
    def _detect_cubes_by_color(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> list:
        """Detect cubes using only color segmentation."""
        detections = []
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Detect red
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Detect green  
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        for color, mask in [('red', red_mask), ('green', green_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 50000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
                
                # Get center pixel
                cx = x + w // 2
                cy = y + h // 2
                
                # Get depth at center
                depth = 0.76  # Default
                if depth_image is not None:
                    region_size = 5
                    y_start = max(0, cy - region_size)
                    y_end = min(depth_image.shape[0], cy + region_size)
                    x_start = max(0, cx - region_size)
                    x_end = min(depth_image.shape[1], cx + region_size)
                    
                    depth_region = depth_image[y_start:y_end, x_start:x_end]
                    valid_depths = depth_region[~np.isnan(depth_region) & (depth_region > 0)]
                    
                    if len(valid_depths) > 0:
                        depth = np.median(valid_depths)
                
                detections.append({
                    'color': color,
                    'pixel': (cx, cy),
                    'depth': depth,
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        return detections
    
    def _run_detection(self):
        """Run detection and log detailed results."""
        if self.latest_rgb is None:
            return
        
        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy() if self.latest_depth is not None else None
        
        # Detect cubes
        detections = self._detect_cubes_by_color(rgb, depth)
        
        # Get actual Gazebo poses
        gazebo_poses = self._get_gazebo_cube_poses()
        
        self.get_logger().info('')
        self.get_logger().info('═' * 70)
        self.get_logger().info('  CAMERA CALIBRATION TEST')
        self.get_logger().info('═' * 70)
        
        if not detections:
            self.get_logger().warn('No cubes detected by color segmentation!')
            return
        
        # Keep best detection per color
        best_per_color = {}
        for det in detections:
            color = det['color']
            if color not in best_per_color or det['area'] > best_per_color[color]['area']:
                best_per_color[color] = det
        
        for color, det in best_per_color.items():
            px, py = det['pixel']
            depth_val = det['depth']
            
            self.get_logger().info(f'')
            self.get_logger().info(f'  {color.upper()} CUBE:')
            self.get_logger().info(f'    Pixel coordinates: ({px}, {py})')
            self.get_logger().info(f'    Depth: {depth_val:.3f} m')
            
            # Test all transformation versions
            v1 = self._pixel_to_world_v1(px, py, depth_val)
            v5 = self._pixel_to_world_v5(px, py, depth_val)
            v6 = self._pixel_to_world_v6(px, py, depth_val)
            
            self.get_logger().info(f'    Transform V1 (original): robot=({v1[0]:.3f}, {v1[1]:.3f})' if v1 else '    Transform V1: Failed')
            self.get_logger().info(f'    Transform V5 (corrected): robot=({v5[0]:.3f}, {v5[1]:.3f})' if v5 else '    Transform V5: Failed')
            self.get_logger().info(f'    Transform V6 (empirical): robot=({v6[0]:.3f}, {v6[1]:.3f})' if v6 else '    Transform V6: Failed')
            
            # Compare with known/actual position
            if color in gazebo_poses:
                actual = gazebo_poses[color]
                ax, ay = actual['robot']
                self.get_logger().info(f'    ACTUAL Gazebo position: robot=({ax:.3f}, {ay:.3f})')
                
                errors = []
                for name, v in [('V1', v1), ('V5', v5), ('V6', v6)]:
                    if v:
                        err = np.sqrt((v[0]-ax)**2 + (v[1]-ay)**2)
                        errors.append((name, err))
                
                # Find best
                if errors:
                    best = min(errors, key=lambda x: x[1])
                    for name, err in errors:
                        marker = ' *** BEST ***' if name == best[0] else ''
                        self.get_logger().info(f'    {name} error: {err:.3f} m{marker}')
            else:
                known = self.known_cubes.get(color, {})
                if 'robot' in known:
                    kx, ky = known['robot']
                    self.get_logger().info(f'    EXPECTED position: robot=({kx:.3f}, {ky:.3f})')
        
        self.get_logger().info('')
        self.get_logger().info('═' * 70)
        
        # Publish detections (using V1 for now)
        output_detections = []
        for color, det in best_per_color.items():
            pos = self._pixel_to_world_v1(det['pixel'][0], det['pixel'][1], det['depth'])
            if pos:
                output_detections.append({
                    'color': color,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'confidence': 0.9
                })
        
        if output_detections:
            msg = String()
            msg.data = json.dumps({'detections': output_detections})
            self._detection_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
