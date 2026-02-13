#!/bin/bash
# ============================================================================
# Vision-Based Object Detection Comparison Benchmark Script
# ============================================================================
# This script helps run benchmarks for YOLOv11 vs RT-DETR comparison.
# 
# Usage:
#   ./run_benchmark.sh yolo    - Run YOLO benchmark
#   ./run_benchmark.sh rtdetr  - Run RT-DETR benchmark
#   ./run_benchmark.sh analyze - Analyze collected metrics
# ============================================================================

COMPARISON_DIR="/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison"
YOLO_WS="$COMPARISON_DIR/Comp_perfect_size_v3/ros2_ws"
RTDETR_WS="$COMPARISON_DIR/Comp_perfect_RT_v1/ros2_ws"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

kill_all() {
    echo -e "${YELLOW}Killing existing Gazebo and ROS processes...${NC}"
    pkill -9 -f gz 2>/dev/null
    pkill -9 -f ros 2>/dev/null
    pkill -9 -f ruby 2>/dev/null
    pkill -9 -f gzserver 2>/dev/null
    pkill -9 -f gzclient 2>/dev/null
    sleep 2
    echo -e "${GREEN}Done.${NC}"
}

run_yolo_benchmark() {
    print_header "Running YOLOv11 Benchmark"
    
    echo -e "${YELLOW}Step 1: Kill existing processes${NC}"
    kill_all
    
    echo -e "${YELLOW}Step 2: Build workspace${NC}"
    cd "$YOLO_WS"
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install 2>/dev/null
    source install/setup.bash
    
    echo -e "${GREEN}Workspace built successfully.${NC}"
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}MANUAL STEPS REQUIRED:${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "Terminal 1 - Launch scene:"
    echo -e "${GREEN}cd $YOLO_WS${NC}"
    echo -e "${GREEN}source /opt/ros/jazzy/setup.bash${NC}"
    echo -e "${GREEN}source install/setup.bash${NC}"
    echo -e "${GREEN}ros2 launch simpler_bringup scene.launch.py${NC}"
    echo ""
    echo -e "Terminal 2 - Run pick and place (after scene is ready):"
    echo -e "${GREEN}cd $YOLO_WS${NC}"
    echo -e "${GREEN}source /opt/ros/jazzy/setup.bash${NC}"
    echo -e "${GREEN}source install/setup.bash${NC}"
    echo -e "${GREEN}ros2 launch simpler_bringup pick_place.launch.py${NC}"
    echo ""
    echo -e "${YELLOW}Let it run for at least 60 seconds to collect metrics.${NC}"
    echo -e "${YELLOW}Metrics will be saved to: $COMPARISON_DIR/yolo_metrics.csv${NC}"
}

run_rtdetr_benchmark() {
    print_header "Running RT-DETR Benchmark"
    
    echo -e "${YELLOW}Step 1: Kill existing processes${NC}"
    kill_all
    
    echo -e "${YELLOW}Step 2: Build workspace${NC}"
    cd "$RTDETR_WS"
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install 2>/dev/null
    source install/setup.bash
    
    echo -e "${GREEN}Workspace built successfully.${NC}"
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}MANUAL STEPS REQUIRED:${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "Terminal 1 - Launch scene:"
    echo -e "${GREEN}cd $RTDETR_WS${NC}"
    echo -e "${GREEN}source /opt/ros/jazzy/setup.bash${NC}"
    echo -e "${GREEN}source install/setup.bash${NC}"
    echo -e "${GREEN}ros2 launch simpler_bringup scene.launch.py${NC}"
    echo ""
    echo -e "Terminal 2 - Run pick and place (after scene is ready):"
    echo -e "${GREEN}cd $RTDETR_WS${NC}"
    echo -e "${GREEN}source /opt/ros/jazzy/setup.bash${NC}"
    echo -e "${GREEN}source install/setup.bash${NC}"
    echo -e "${GREEN}ros2 launch simpler_bringup pick_place.launch.py${NC}"
    echo ""
    echo -e "${YELLOW}Let it run for at least 60 seconds to collect metrics.${NC}"
    echo -e "${YELLOW}Metrics will be saved to: $COMPARISON_DIR/rtdetr_metrics.csv${NC}"
}

analyze_metrics() {
    print_header "Analyzing Benchmark Results"
    
    YOLO_CSV="$COMPARISON_DIR/yolo_metrics.csv"
    RTDETR_CSV="$COMPARISON_DIR/rtdetr_metrics.csv"
    
    echo ""
    if [ -f "$YOLO_CSV" ]; then
        echo -e "${GREEN}=== YOLOv11 Results ===${NC}"
        # Skip header and calculate stats
        tail -n +2 "$YOLO_CSV" | awk -F',' '
        BEGIN { sum=0; count=0; min=999999; max=0; det_sum=0 }
        {
            sum += $2
            det_sum += $3
            count++
            if ($2 < min) min = $2
            if ($2 > max) max = $2
        }
        END {
            if (count > 0) {
                avg = sum / count
                printf "  Frames processed: %d\n", count
                printf "  Average inference time: %.2f ms\n", avg
                printf "  Min inference time: %.2f ms\n", min
                printf "  Max inference time: %.2f ms\n", max
                printf "  Total detections: %d\n", det_sum
                printf "  Avg detections/frame: %.2f\n", det_sum/count
            }
        }'
    else
        echo -e "${RED}YOLO metrics file not found. Run YOLO benchmark first.${NC}"
    fi
    
    echo ""
    if [ -f "$RTDETR_CSV" ]; then
        echo -e "${GREEN}=== RT-DETR Results ===${NC}"
        tail -n +2 "$RTDETR_CSV" | awk -F',' '
        BEGIN { sum=0; count=0; min=999999; max=0; det_sum=0 }
        {
            sum += $2
            det_sum += $3
            count++
            if ($2 < min) min = $2
            if ($2 > max) max = $2
        }
        END {
            if (count > 0) {
                avg = sum / count
                printf "  Frames processed: %d\n", count
                printf "  Average inference time: %.2f ms\n", avg
                printf "  Min inference time: %.2f ms\n", min
                printf "  Max inference time: %.2f ms\n", max
                printf "  Total detections: %d\n", det_sum
                printf "  Avg detections/frame: %.2f\n", det_sum/count
            }
        }'
    else
        echo -e "${RED}RT-DETR metrics file not found. Run RT-DETR benchmark first.${NC}"
    fi
    
    echo ""
    print_header "Comparison Summary"
    
    if [ -f "$YOLO_CSV" ] && [ -f "$RTDETR_CSV" ]; then
        YOLO_AVG=$(tail -n +2 "$YOLO_CSV" | awk -F',' '{sum+=$2; count++} END {print sum/count}')
        RTDETR_AVG=$(tail -n +2 "$RTDETR_CSV" | awk -F',' '{sum+=$2; count++} END {print sum/count}')
        
        echo -e "YOLOv11 Average: ${GREEN}${YOLO_AVG} ms${NC}"
        echo -e "RT-DETR Average: ${GREEN}${RTDETR_AVG} ms${NC}"
        
        # Compare
        if (( $(echo "$YOLO_AVG < $RTDETR_AVG" | bc -l) )); then
            SPEEDUP=$(echo "scale=2; $RTDETR_AVG / $YOLO_AVG" | bc)
            echo -e "${GREEN}YOLOv11 is ${SPEEDUP}x faster than RT-DETR${NC}"
        else
            SPEEDUP=$(echo "scale=2; $YOLO_AVG / $RTDETR_AVG" | bc)
            echo -e "${GREEN}RT-DETR is ${SPEEDUP}x faster than YOLOv11${NC}"
        fi
    fi
}

# Main script
case "$1" in
    yolo)
        run_yolo_benchmark
        ;;
    rtdetr)
        run_rtdetr_benchmark
        ;;
    analyze)
        analyze_metrics
        ;;
    kill)
        kill_all
        ;;
    *)
        echo "Usage: $0 {yolo|rtdetr|analyze|kill}"
        echo ""
        echo "Commands:"
        echo "  yolo    - Prepare and show commands for YOLO benchmark"
        echo "  rtdetr  - Prepare and show commands for RT-DETR benchmark"
        echo "  analyze - Analyze collected metrics from both benchmarks"
        echo "  kill    - Kill all Gazebo and ROS processes"
        exit 1
        ;;
esac
