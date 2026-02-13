# Vision-Based-Object-Detection-and-Sorting

### How to run (3 commands per version)

- **Important**: Run **Command 2** and **Command 3** in **two separate terminals** (start the scene first, then start pick & place).
- **Assumes**: the workspace is already built and `install/setup.bash` exists (run `colcon build --symlink-install` once if needed).

---

### `perfect_size_v2`

```bash
pkill -9 -f gz 2>/dev/null; pkill -9 -f ros 2>/dev/null; pkill -9 -f ruby 2>/dev/null; pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null; sleep 2
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_size_v2/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup scene.launch.py
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_size_v2/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup pick_place.launch.py
```green pick position
[pick_place_controller.py-1] [INFO] [1770207229.744006640] [pick_place_controller]: 
[pick_place_controller.py-1] ── Grasp BIG GREEN Cube ──
[pick_place_controller.py-1] [INFO] [1770207229.745061080] [pick_place_controller]: [GRIPPER] CLOSING to 0.68
[pick_place_controller.py-1

---

### `perfect_size_v3`

```bash
pkill -9 -f gz 2>/dev/null; pkill -9 -f ros 2>/dev/null; pkill -9 -f ruby 2>/dev/null; pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null; sleep 2
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_size_v3/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup scene.launch.py
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_size_v3/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup pick_place.launch.py
```

---

### `perfect_RT_v1`

```bash
pkill -9 -f gz 2>/dev/null; pkill -9 -f ros 2>/dev/null; pkill -9 -f ruby 2>/dev/null; pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null; sleep 2
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_RT_v1/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup scene.launch.py
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/perfect_RT_v1/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup pick_place.launch.py
```

---

### `rt_final`

```bash
pkill -9 -f gz 2>/dev/null; pkill -9 -f ros 2>/dev/null; pkill -9 -f ruby 2>/dev/null; pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null; sleep 2
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/rt_final/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup scene.launch.py
```

```bash
cd "/home/beki/Vision-Based-Object-Detection-and-Sorting/rt_final/ros2_ws" && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch simpler_bringup pick_place.launch.py
```