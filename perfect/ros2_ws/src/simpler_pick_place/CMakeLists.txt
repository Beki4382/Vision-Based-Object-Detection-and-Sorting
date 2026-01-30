cmake_minimum_required(VERSION 3.8)
project(simpler_pick_place)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(scripts)

install(PROGRAMS
  scripts/pick_place_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
