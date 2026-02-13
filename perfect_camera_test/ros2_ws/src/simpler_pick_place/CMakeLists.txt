cmake_minimum_required(VERSION 3.8)
project(simpler_pick_place)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(scripts)

# Install all Python scripts
install(PROGRAMS
  scripts/pick_place_demo.py
  scripts/scene_manager.py
  scripts/pick_place_controller.py
  scripts/vision_node.py
  scripts/pose_logger.py
  scripts/camera_test_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
