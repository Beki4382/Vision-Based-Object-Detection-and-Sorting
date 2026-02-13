#!/usr/bin/env python3
"""
Vision Node for YOLO-based Object Detection

This node:
1. Subscribes to RGB and Depth images from the overhead camera
2. Runs YOLOv8 inference to detect objects (cubes)
3. Classifies detected objects by color (red/green) using HSV
4. Calculates 3D world coordinates using depth and camera intrinsics
5. Publishes detected cube poses to /detected_cubes

For the simulation, we use a pretrained YOLOv8 model and rely on
color classification to distinguish red vs green cubes.

Author: Perfect YOLO V1
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
import time

# Try to import ultralytics, provide helpful error if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Install with: pip install ultralytics")


class VisionNode(Node):
    """
    Vision node that detects cubes using YOLOv8 and color classification.
    """
    
    def __init__(self):
        super().__init__('vision_node')
        
        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        
        # ══════════════════════════════════════════════════════════════════════
        # Camera Configuration
        # Camera is positioned at (0.4, -0.5, 1.5) looking down at table
        # Table surface is at z=0.74, so camera height above table is ~0.76m
        # ══════════════════════════════════════════════════════════════════════
        self.camera_position = np.array([0.4, -0.5, 1.5])
        self.table_height = 0.74
        self.cube_half_height = 0.03  # Half of cube size (6cm cube)
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.fx = 554.25  # Approximate focal length for 640x480 with 60 deg FOV
        self.fy = 554.25
        self.cx = 320.0   # Principal point (image center)
        self.cy = 240.0
        self.camera_info_received = False
        
        # Robot base position in world frame
        self.robot_base_world = np.array([0.0, -0.5, 0.74])
        
        # ══════════════════════════════════════════════════════════════════════
        # YOLO Model
        # ══════════════════════════════════════════════════════════════════════
        self.model = None
        if YOLO_AVAILABLE:
            try:
                # Use YOLOv8 nano model (fastest, good for simulation)
                self.get_logger().info('Loading YOLOv8 model...')
                self.model = YOLO('yolov8n.pt')
                self.get_logger().info('YOLOv8 model loaded successfully')
            except Exception as e:
                self.get_logger().error(f'Failed to load YOLO model: {e}')
                self.model = None
        
        # ══════════════════════════════════════════════════════════════════════
        # HSV Color Ranges for Cube Classification
        # ══════════════════════════════════════════════════════════════════════
        # Red color (wraps around in HSV, so we need two ranges)
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
        self.rgb_stamp = None
        self.depth_stamp = None
        
        # ══════════════════════════════════════════════════════════════════════
        # Subscribers
        # ══════════════════════════════════════════════════════════════════════
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self._rgb_sub = self.create_subscription(
            Image,
            '/camera/image',
            self._rgb_callback,
            sensor_qos,
            callback_group=self.callback_group
        )
        
        self._depth_sub = self.create_subscription(
            Image,
            '/camera/depth_image',
            self._depth_callback,
            sensor_qos,
            callback_group=self.callback_group
        )
        
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self._camera_info_callback,
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
        
        self._detection_pub = self.create_publisher(
            String,
            '/detected_cubes',
            latching_qos
        )
        
        # Debug image publisher (optional)
        self._debug_image_pub = self.create_publisher(
            Image,
            '/vision/debug_image',
            10
        )
        
        # ══════════════════════════════════════════════════════════════════════
        # Detection Timer
        # ══════════════════════════════════════════════════════════════════════
        self._detection_timer = self.create_timer(
            1.0,  # Run detection every 1 second
            self._run_detection,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Vision Node initialized')
        self.get_logger().info('Waiting for camera images...')
    
    def _rgb_callback(self, msg: Image):
        """Store latest RGB image."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.rgb_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {e}')
    
    def _depth_callback(self, msg: Image):
        """Store latest depth image."""
        try:
            # Depth image is typically 32FC1 (float meters) or 16UC1 (uint16 mm)
            if msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.latest_depth = depth_mm.astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
    
    def _camera_info_callback(self, msg: CameraInfo):
        """Update camera intrinsics from CameraInfo."""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, '
                f'cx={self.cx:.1f}, cy={self.cy:.1f}'
            )
    
    def _classify_color(self, rgb_image: np.ndarray, bbox) -> str:
        """
        Classify the color of an object within a bounding box.
        
        Args:
            rgb_image: BGR image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            'red', 'green', or 'unknown'
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounds are valid
        h, w = rgb_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        
        # Extract region of interest
        roi = rgb_image[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Count red pixels (two ranges because red wraps around)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_count = cv2.countNonZero(red_mask)
        
        # Count green pixels
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_count = cv2.countNonZero(green_mask)
        
        # Determine color based on pixel counts
        total_pixels = (x2 - x1) * (y2 - y1)
        min_ratio = 0.1  # At least 10% of pixels should be the color
        
        if red_count > green_count and red_count > total_pixels * min_ratio:
            return 'red'
        elif green_count > red_count and green_count > total_pixels * min_ratio:
            return 'green'
        else:
            return 'unknown'
    
    def _pixel_to_world(self, px: float, py: float, depth: float) -> tuple:
        """
        Convert pixel coordinates and depth to world coordinates.
        
        The camera is looking down (pitch = 90 degrees), so:
        - Camera X axis -> World -Y axis
        - Camera Y axis -> World X axis
        - Camera Z axis (depth) -> World -Z axis
        
        Args:
            px, py: Pixel coordinates
            depth: Depth value in meters
        
        Returns:
            (x, y, z) in robot base frame
        """
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None
        
        # Convert pixel to camera frame (normalized coordinates)
        cam_x = (px - self.cx) / self.fx * depth
        cam_y = (py - self.cy) / self.fy * depth
        cam_z = depth
        
        # Camera is rotated 90 degrees down (pitch = 90 deg)
        # Camera frame to world frame transformation:
        # World X = Camera Y (with offset)
        # World Y = -Camera X (with offset)
        # World Z = Camera height - depth
        world_x = self.camera_position[0] + cam_y
        world_y = self.camera_position[1] - cam_x
        world_z = self.camera_position[2] - cam_z
        
        # Convert to robot base frame
        robot_x = world_x - self.robot_base_world[0]
        robot_y = world_y - self.robot_base_world[1]
        
        return (robot_x, robot_y, world_z)
    
    def _detect_cubes_yolo(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> list:
        """
        Detect cubes using YOLO and classify by color.
        
        Returns:
            List of detected cubes with positions
        """
        detections = []
        
        if self.model is None:
            return detections
        
        # Run YOLO detection
        results = self.model(rgb_image, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Filter low confidence detections
                if confidence < 0.3:
                    continue
                
                # Classify color
                color = self._classify_color(rgb_image, [x1, y1, x2, y2])
                
                if color == 'unknown':
                    continue
                
                # Get center of bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Get depth at center
                if depth_image is not None:
                    # Use a small region around center for more stable depth
                    region_size = 5
                    y_start = max(0, cy - region_size)
                    y_end = min(depth_image.shape[0], cy + region_size)
                    x_start = max(0, cx - region_size)
                    x_end = min(depth_image.shape[1], cx + region_size)
                    
                    depth_region = depth_image[y_start:y_end, x_start:x_end]
                    valid_depths = depth_region[~np.isnan(depth_region) & (depth_region > 0)]
                    
                    if len(valid_depths) > 0:
                        depth = np.median(valid_depths)
                    else:
                        depth = 0.76  # Default: camera height above table
                else:
                    depth = 0.76
                
                # Convert to world coordinates
                world_pos = self._pixel_to_world(cx, cy, depth)
                
                if world_pos is not None:
                    detections.append({
                        'color': color,
                        'x': float(world_pos[0]),
                        'y': float(world_pos[1]),
                        'z': float(world_pos[2]),
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return detections
    
    def _detect_cubes_color_only(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> list:
        """
        Fallback detection using only color segmentation (no YOLO).
        
        Returns:
            List of detected cubes with positions
        """
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Detect red cubes
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Detect green cubes
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        for color, mask in [('red', red_mask), ('green', green_mask)]:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (cube should be reasonably sized)
                if area < 100 or area > 50000:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (cube should be roughly square from above)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Get center
                cx = x + w // 2
                cy = y + h // 2
                
                # Get depth
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
                    else:
                        depth = 0.76
                else:
                    depth = 0.76
                
                # Convert to world coordinates
                world_pos = self._pixel_to_world(cx, cy, depth)
                
                if world_pos is not None:
                    detections.append({
                        'color': color,
                        'x': float(world_pos[0]),
                        'y': float(world_pos[1]),
                        'z': float(world_pos[2]),
                        'confidence': 0.8,  # Color detection confidence
                        'bbox': [float(x), float(y), float(x + w), float(y + h)]
                    })
        
        return detections
    
    def _run_detection(self):
        """Run object detection on current images."""
        if self.latest_rgb is None:
            return
        
        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy() if self.latest_depth is not None else None
        
        # Run detection
        if self.model is not None:
            detections = self._detect_cubes_yolo(rgb, depth)
        else:
            # Fallback to color-only detection
            detections = self._detect_cubes_color_only(rgb, depth)
        
        # Filter duplicates (keep highest confidence per color)
        unique_detections = {}
        for det in detections:
            color = det['color']
            if color not in unique_detections or det['confidence'] > unique_detections[color]['confidence']:
                unique_detections[color] = det
        
        detections = list(unique_detections.values())
        
        # Publish detections
        if detections:
            msg = String()
            msg.data = json.dumps({'detections': detections})
            self._detection_pub.publish(msg)
            
            self.get_logger().info(
                f'Detected {len(detections)} cube(s): ' + 
                ', '.join([f"{d['color']} at ({d['x']:.2f}, {d['y']:.2f})" for d in detections])
            )
        
        # Publish debug image
        debug_image = rgb.copy()
        for det in detections:
            bbox = det['bbox']
            color = (0, 0, 255) if det['color'] == 'red' else (0, 255, 0)
            cv2.rectangle(debug_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            cv2.putText(debug_image, 
                       f"{det['color']} ({det['x']:.2f}, {det['y']:.2f})",
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            self._debug_image_pub.publish(debug_msg)
        except Exception as e:
            pass  # Ignore debug image errors


def main(args=None):
    rclpy.init(args=args)
    
    node = VisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
