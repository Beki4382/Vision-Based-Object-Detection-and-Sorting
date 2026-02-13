#!/usr/bin/env python3
"""
Camera Calibration Script for Size V1

This script logs the pixel coordinates from YOLO detection alongside
the actual Gazebo positions to derive the correct pixel-to-robot transformation.

Run this after the scene is launched to collect calibration data.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import subprocess
import re
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        
        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        
        # YOLO model
        self.model = None
        CUSTOM_MODEL_PATH = '/home/beki/Vision-Based-Object-Detection-and-Sorting/yolo/cube_detector_best.pt'
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(CUSTOM_MODEL_PATH)
                self.get_logger().info('YOLO model loaded')
            except Exception as e:
                self.get_logger().error(f'Failed to load YOLO: {e}')
        
        # Camera position
        self.camera_position = np.array([0.4, -0.5, 1.5])
        
        # Robot base in world frame
        self.robot_base_x = 0.0
        self.robot_base_y = -0.5
        
        # Expected cube positions in world frame (from SDF)
        # Pick area moved forward to separate from place area
        self.expected_world = {
            'big_green_cube': (0.25, -0.25, 0.80),
            'green_cube': (0.55, -0.25, 0.77),
            'big_red_cube': (0.25, -0.45, 0.80),
            'red_cube': (0.55, -0.45, 0.77),
        }
        
        # Convert to robot frame
        self.expected_robot = {}
        for name, (wx, wy, wz) in self.expected_world.items():
            rx = wx - self.robot_base_x
            ry = wy - self.robot_base_y
            self.expected_robot[name] = (rx, ry)
        
        # Storage for detected pixels
        self.latest_rgb = None
        
        # Subscribe to camera
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
        
        # Run calibration after a delay
        self._timer = self.create_timer(5.0, self._run_calibration, callback_group=self.callback_group)
        
        self.get_logger().info('Camera Calibrator starting...')
        self.get_logger().info('Waiting for camera images...')
    
    def _rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            pass
    
    def _run_calibration(self):
        self._timer.cancel()
        
        if self.latest_rgb is None:
            self.get_logger().error('No camera image received!')
            return
        
        if self.model is None:
            self.get_logger().error('YOLO model not available!')
            return
        
        self.get_logger().info('\n' + '=' * 70)
        self.get_logger().info('  CAMERA CALIBRATION')
        self.get_logger().info('=' * 70)
        
        # Run YOLO detection
        results = self.model(self.latest_rgb, verbose=False)
        
        # Map class names to colors
        CLASS_TO_COLOR = {
            'red cube': 'red', 'red_cube': 'red', 'redcube': 'red',
            'green cube': 'green', 'green_cube': 'green', 'greencube': 'green',
        }
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower()
                
                if confidence < 0.5:
                    continue
                
                color = CLASS_TO_COLOR.get(class_name)
                if color is None:
                    continue
                
                # Get center pixel
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Get bounding box area for size classification
                area = (x2 - x1) * (y2 - y1)
                size = 'big' if area > 5000 else 'small'
                
                detections.append({
                    'color': color,
                    'size': size,
                    'px': cx,
                    'py': cy,
                    'area': area,
                    'bbox': (x1, y1, x2, y2)
                })
        
        self.get_logger().info(f'\nDetected {len(detections)} cube(s):\n')
        
        # Match detections to expected positions
        calibration_points = []
        
        for det in detections:
            # Find matching expected cube
            size_prefix = 'big_' if det['size'] == 'big' else ''
            cube_name = f"{size_prefix}{det['color']}_cube"
            
            if cube_name in self.expected_robot:
                rx, ry = self.expected_robot[cube_name]
                px, py = det['px'], det['py']
                
                calibration_points.append({
                    'name': cube_name,
                    'px': px, 'py': py,
                    'rx': rx, 'ry': ry
                })
                
                self.get_logger().info(
                    f'  {cube_name}:'
                )
                self.get_logger().info(
                    f'    Pixel: ({px}, {py})  Area: {det["area"]:.0f}'
                )
                self.get_logger().info(
                    f'    Expected Robot: ({rx:.3f}, {ry:.3f})'
                )
        
        if len(calibration_points) >= 2:
            self.get_logger().info('\n' + '-' * 70)
            self.get_logger().info('  DERIVING TRANSFORMATION')
            self.get_logger().info('-' * 70)
            
            # Use least squares to find best fit
            # robot_x = scale_x * px + offset_x
            # robot_y = scale_y * py + offset_y
            
            px_vals = np.array([p['px'] for p in calibration_points])
            py_vals = np.array([p['py'] for p in calibration_points])
            rx_vals = np.array([p['rx'] for p in calibration_points])
            ry_vals = np.array([p['ry'] for p in calibration_points])
            
            # Linear regression for X
            A_x = np.column_stack([px_vals, np.ones(len(px_vals))])
            scale_x, offset_x = np.linalg.lstsq(A_x, rx_vals, rcond=None)[0]
            
            # Linear regression for Y
            A_y = np.column_stack([py_vals, np.ones(len(py_vals))])
            scale_y, offset_y = np.linalg.lstsq(A_y, ry_vals, rcond=None)[0]
            
            self.get_logger().info(f'\n  Derived transformation:')
            self.get_logger().info(f'    robot_x = {scale_x:.6f} * px + ({offset_x:.6f})')
            self.get_logger().info(f'    robot_y = {scale_y:.6f} * py + ({offset_y:.6f})')
            
            # Test the transformation
            self.get_logger().info(f'\n  Verification:')
            for p in calibration_points:
                pred_rx = scale_x * p['px'] + offset_x
                pred_ry = scale_y * p['py'] + offset_y
                err_x = abs(pred_rx - p['rx'])
                err_y = abs(pred_ry - p['ry'])
                self.get_logger().info(
                    f'    {p["name"]}: predicted ({pred_rx:.3f}, {pred_ry:.3f}) '
                    f'actual ({p["rx"]:.3f}, {p["ry"]:.3f}) '
                    f'error ({err_x:.4f}, {err_y:.4f})'
                )
            
            self.get_logger().info('\n' + '=' * 70)
            self.get_logger().info('  UPDATE vision_node.py with these values:')
            self.get_logger().info('=' * 70)
            self.get_logger().info(f'    SCALE_X = {scale_x:.6f}')
            self.get_logger().info(f'    OFFSET_X = {offset_x:.6f}')
            self.get_logger().info(f'    SCALE_Y = {scale_y:.6f}')
            self.get_logger().info(f'    OFFSET_Y = {offset_y:.6f}')
            self.get_logger().info('=' * 70 + '\n')
        
        # Shutdown after calibration
        self.get_logger().info('Calibration complete. Shutting down...')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
