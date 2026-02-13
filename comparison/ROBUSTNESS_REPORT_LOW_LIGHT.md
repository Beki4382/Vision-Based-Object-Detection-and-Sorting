# Low-light robustness check (YOLOv11 vs RT-DETR)

## Goal
Evaluate **robustness under low illumination** by running both detectors in the *same Gazebo world* but with **reduced ambient + directional light intensity**, then observing:

- **Detection stability**: detections per frame, % of frames with ≥1 detection
- **System readiness**: whether `scene_manager` becomes “Ready” and sees all 4 cubes

This test isolates perception by running **scene only** (vision + scene manager) for a fixed time window.

## What changed (comparison folder only)

- **Dim world variant added**:
  - `comparison/Comp_perfect_size_v3/ros2_ws/src/simpler_description/worlds/pick_place_dim.sdf`
  - `comparison/Comp_perfect_RT_v1/ros2_ws/src/simpler_description/worlds/pick_place_dim.sdf`
  - Lighting changes vs `pick_place.sdf`:
    - `<scene><ambient>`: `0.4 0.4 0.4 1` → `0.05 0.05 0.05 1`
    - `<light><diffuse>`: `0.8 0.8 0.8 1` → `0.12 0.12 0.12 1`
    - `<light><specular>`: `0.2 0.2 0.2 1` → `0.03 0.03 0.03 1`

- **Metrics output made configurable (so we don’t overwrite baseline CSVs)**:
  - `comparison/Comp_perfect_size_v3/ros2_ws/src/simpler_pick_place/scripts/vision_node.py`
  - `comparison/Comp_perfect_RT_v1/ros2_ws/src/simpler_pick_place/scripts/vision_node.py`
  - Both now respect `METRICS_CSV_PATH` (defaults remain the original benchmark paths).

## How to run (recommended)

Run the automated scene-only low-light test (runs YOLO then RT-DETR, headless, ~90s each):

```bash
chmod +x "/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison/robustness_test/run_low_light_scene_only.sh"
"/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison/robustness_test/run_low_light_scene_only.sh"
python3 "/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison/robustness_test/analyze_low_light_metrics.py"
```

Artifacts are written to:

- `comparison/robustness_test/runs_low_light/yolo_low_light_scene.log`
- `comparison/robustness_test/runs_low_light/rtdetr_low_light_scene.log`
- `comparison/robustness_test/runs_low_light/yolo_low_light_metrics.csv`
- `comparison/robustness_test/runs_low_light/rtdetr_low_light_metrics.csv`
- `comparison/robustness_test/runs_low_light/LOW_LIGHT_METRICS_SUMMARY.md`

## Results (this run)

From `comparison/robustness_test/runs_low_light/LOW_LIGHT_METRICS_SUMMARY.md`:

- **YOLOv11**:
  - Frames: **58**
  - Avg inference: **119.96 ms**
  - Avg detections/frame: **6.00**
  - Frames with ≥1 detection: **100.0%**

- **RT-DETR**:
  - Frames: **40**
  - Avg inference: **229.45 ms**
  - Avg detections/frame: **6.00**
  - Frames with ≥1 detection: **100.0%**

### Scene manager readiness (qualitative)

Both pipelines eventually reached “Scene Manager Ready” and reported **4 cubes detected** in low light:

- **YOLO**: `scene_manager` prints:
  - `Scene Manager Ready (Size-Based Sorting)`
  - `Detected 4 cube(s)`
  in `comparison/robustness_test/runs_low_light/yolo_low_light_scene.log`

- **RT-DETR**: same readiness messages appear in:
  - `comparison/robustness_test/runs_low_light/rtdetr_low_light_scene.log`

## Interpretation

- **Robustness to low illumination (in this synthetic test)**: both models maintained detections on **every recorded frame**.
- **Speed under low light**: YOLO remained faster on average; RT-DETR showed higher latency and higher variance (large max inference time spikes).

## Notes / limitations

- This is a **Gazebo lighting** robustness test, not a real camera low-light test. Real sensors add noise, motion blur, and auto-exposure behavior.
- The “Avg detections/frame” counts **all raw detections** from the model output; it can exceed the number of physical cubes if multiple boxes are produced.
- For end-to-end robustness (including grasp + placement), repeat with `pick_place.launch.py` after the scene is ready and score success rate.

