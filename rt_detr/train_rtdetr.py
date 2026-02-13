#!/usr/bin/env python3
"""
RT-DETR Training Script for Cube Detection

This script:
1. Downloads the cube detection dataset from Roboflow
2. Fine-tunes RT-DETR on the dataset
3. Saves the trained model for use in the vision node

RT-DETR (Real-Time Detection Transformer) is Baidu's Vision Transformer-based 
real-time object detector. It offers:
- NMS-free detection (no non-maximum suppression needed)
- Efficient hybrid encoder for multiscale features
- Anchor-free detection

Usage:
    python train_rtdetr.py
"""

import os
import sys

# Install required packages if not present
def install_packages():
    """Install required packages."""
    import subprocess
    packages = ['roboflow', 'ultralytics']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', pkg, 
                '--user', '--break-system-packages'
            ])

print("=" * 60)
print("  RT-DETR Cube Detection Training")
print("=" * 60)

# Install packages
print("\n[1/4] Checking dependencies...")
install_packages()

from roboflow import Roboflow
from ultralytics import RTDETR

# Configuration
ROBOFLOW_API_KEY = "3aPxFehoIx0YShcDI3FG"
WORKSPACE = "beky"
PROJECT = "red-green-blue-cube-detection-tidtc"
VERSION = 2

# Training parameters
EPOCHS = 100
IMAGE_SIZE = 416  # Reduced from 640 to save GPU memory
BATCH_SIZE = 2    # RT-DETR is more memory-intensive than YOLO
MODEL_NAME = "rtdetr-l.pt"  # RT-DETR-L model (can use rtdetr-x.pt for better accuracy)

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs")

print(f"\n[2/4] Downloading dataset from Roboflow...")
print(f"  Workspace: {WORKSPACE}")
print(f"  Project: {PROJECT}")
print(f"  Version: {VERSION}")

# Download dataset (RT-DETR uses same format as YOLO)
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download("yolov11", location=os.path.join(SCRIPT_DIR, "dataset"))

print(f"\n  Dataset downloaded to: {dataset.location}")

# Get the data.yaml path
data_yaml = os.path.join(dataset.location, "data.yaml")
print(f"  data.yaml: {data_yaml}")

print(f"\n[3/4] Training RT-DETR...")
print(f"  Base model: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Image size: {IMAGE_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print("=" * 60)

# Load pretrained RT-DETR model
model = RTDETR(MODEL_NAME)

# Train the model
results = model.train(
    data=data_yaml,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    name="cube_detector_rtdetr",
    project=OUTPUT_DIR,
    patience=20,  # Early stopping
    save=True,
    plots=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("[4/4] Training Complete!")
print("=" * 60)

# Find the best model
best_model_path = os.path.join(OUTPUT_DIR, "cube_detector_rtdetr", "weights", "best.pt")
if os.path.exists(best_model_path):
    print(f"\n  Best model saved at: {best_model_path}")
    
    # Copy to a convenient location
    final_model_path = os.path.join(SCRIPT_DIR, "cube_detector_rtdetr_best.pt")
    import shutil
    shutil.copy(best_model_path, final_model_path)
    print(f"  Copied to: {final_model_path}")
    
    print(f"\n  To use in vision_node.py, update the model path:")
    print(f"    from ultralytics import RTDETR")
    print(f"    self.model = RTDETR('{final_model_path}')")
else:
    print(f"\n  Warning: Best model not found at {best_model_path}")
    print(f"  Check the runs directory for training output.")

print("\n" + "=" * 60)
