## Low-light robustness (scene-only) — metrics summary

These metrics come from running `scene.launch.py` only (vision + scene manager) in a dim-light Gazebo world for ~90 seconds.

- **YOLOv11** (`yolo_low_light_metrics.csv`):
  - Frames: 58
  - Avg inference: 119.96 ms (min 64.45, max 2359.84, std 299.44)
  - Avg detections/frame: 6.00
  - Frames with ≥1 detection: 100.0%

- **RT-DETR** (`rtdetr_low_light_metrics.csv`):
  - Frames: 40
  - Avg inference: 229.45 ms (min 64.78, max 5959.50, std 929.31)
  - Avg detections/frame: 6.00
  - Frames with ≥1 detection: 100.0%
