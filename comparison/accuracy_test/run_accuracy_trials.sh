#!/usr/bin/env bash
# Note: do NOT enable `set -u` because ROS setup scripts may reference
# unset variables (e.g., AMENT_TRACE_SETUP_FILES) depending on environment.
set -eo pipefail

# Runs headless pick-and-place trials and saves logs under comparison/accuracy_test/runs/.
#
# Usage:
#   ./accuracy_test/run_accuracy_trials.sh yolo 3
#   ./accuracy_test/run_accuracy_trials.sh rtdetr 3
#
# Notes:
# - Each trial starts a fresh Gazebo+ROS graph (kills processes first)
# - Scene is launched headless (gazebo_gui:=false, launch_rviz:=false)
# - Controller is launched and allowed to run up to a timeout

ROOT="/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison"
MODEL="${1:-}"
TRIALS="${2:-3}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 {yolo|rtdetr} [trials]"
  exit 1
fi

if [[ "${MODEL}" == "yolo" ]]; then
  WS="${ROOT}/Comp_perfect_size_v3/ros2_ws"
elif [[ "${MODEL}" == "rtdetr" ]]; then
  WS="${ROOT}/Comp_perfect_RT_v1/ros2_ws"
else
  echo "Unknown model: ${MODEL} (expected yolo or rtdetr)"
  exit 1
fi

OUTDIR="${ROOT}/accuracy_test/runs/${MODEL}"
mkdir -p "${OUTDIR}"

kill_all() {
  pkill -9 -f gz 2>/dev/null || true
  pkill -9 -f ros 2>/dev/null || true
  pkill -9 -f ruby 2>/dev/null || true
  pkill -9 -f gzserver 2>/dev/null || true
  pkill -9 -f gzclient 2>/dev/null || true
}

echo "[INFO] Model=${MODEL} trials=${TRIALS}"
echo "[INFO] Workspace=${WS}"
echo "[INFO] Output=${OUTDIR}"

for i in $(seq 1 "${TRIALS}"); do
  RUN="${OUTDIR}/trial_${i}"
  mkdir -p "${RUN}"

  echo "[INFO] === Trial ${i}/${TRIALS} ==="
  kill_all
  sleep 3

  cd "${WS}"
  # Ensure environment is loaded for ros2 launch
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash

  SCENE_LOG="${RUN}/scene.log"
  CTRL_LOG="${RUN}/controller.log"

  echo "[INFO] Launching scene (headless)..."
  # Start scene in background and capture PID
  (ros2 launch simpler_bringup scene.launch.py gazebo_gui:=false launch_rviz:=false >"${SCENE_LOG}" 2>&1) &
  SCENE_PID=$!

  # Give Gazebo + bridges + MoveIt + vision + scene_manager time to come up.
  # (scene.launch.py itself already has timers, but we add a buffer)
  echo "[INFO] Waiting for scene to stabilize..."
  sleep 40

  echo "[INFO] Launching controller (timeout 12 min)..."
  # The controller itself waits ~15s at start. Allow enough time for 4 cubes.
  timeout 720 ros2 launch simpler_bringup pick_place.launch.py >"${CTRL_LOG}" 2>&1 || true

  echo "[INFO] Stopping scene..."
  kill "${SCENE_PID}" 2>/dev/null || true
  sleep 2
  kill_all

  echo "[INFO] Trial ${i} done. Logs:"
  echo "  - ${SCENE_LOG}"
  echo "  - ${CTRL_LOG}"
done

