#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison"

YOLO_WS="$BASE_DIR/Comp_perfect_size_v3/ros2_ws"
RTDETR_WS="$BASE_DIR/Comp_perfect_RT_v1/ros2_ws"

YOLO_WORLD="$YOLO_WS/src/simpler_description/worlds/pick_place_dim.sdf"
RTDETR_WORLD="$RTDETR_WS/src/simpler_description/worlds/pick_place_dim.sdf"

OUT_DIR="$BASE_DIR/robustness_test/runs_low_light"
mkdir -p "$OUT_DIR"

kill_all() {
  pkill -9 -f gz 2>/dev/null || true
  pkill -9 -f ros 2>/dev/null || true
  pkill -9 -f ruby 2>/dev/null || true
  pkill -9 -f gzserver 2>/dev/null || true
  pkill -9 -f gzclient 2>/dev/null || true
  sleep 2
}

run_scene_only() {
  local model="$1"
  local ws="$2"
  local world="$3"
  local metrics_csv="$4"
  local log_file="$5"

  echo "=================================================================="
  echo "LOW-LIGHT SCENE-ONLY TEST: $model"
  echo "  ws: $ws"
  echo "  world: $world"
  echo "  metrics: $metrics_csv"
  echo "  log: $log_file"
  echo "=================================================================="

  kill_all

  cd "$ws"
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash

  export METRICS_CSV_PATH="$metrics_csv"
  mkdir -p "$(dirname "$METRICS_CSV_PATH")"
  rm -f "$METRICS_CSV_PATH"

  # Run only the scene (vision + scene_manager) for a fixed window.
  # This isolates detection robustness from controller/grasp issues.
  timeout 90s ros2 launch simpler_bringup scene.launch.py \
    gazebo_gui:=false launch_rviz:=false \
    world_file:="$world" \
    2>&1 | tee "$log_file" || true

  kill_all
}

run_scene_only \
  "yolo" \
  "$YOLO_WS" \
  "$YOLO_WORLD" \
  "$OUT_DIR/yolo_low_light_metrics.csv" \
  "$OUT_DIR/yolo_low_light_scene.log"

run_scene_only \
  "rtdetr" \
  "$RTDETR_WS" \
  "$RTDETR_WORLD" \
  "$OUT_DIR/rtdetr_low_light_metrics.csv" \
  "$OUT_DIR/rtdetr_low_light_scene.log"

echo ""
echo "Done."
echo "Metrics:"
echo "  $OUT_DIR/yolo_low_light_metrics.csv"
echo "  $OUT_DIR/rtdetr_low_light_metrics.csv"
