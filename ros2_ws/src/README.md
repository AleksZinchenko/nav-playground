# ros2_ws/src

ROS2-пакети (модуль 8). Збірка:

```bash
cd ros2_ws && colcon build && source install/setup.bash
```

План: `navpg_fusion` — нода, що підписується на `/imu` (`sensor_msgs/Imu`) і `/gnss` (`sensor_msgs/NavSatFix`)
та публікує `/odometry/filtered` (`nav_msgs/Odometry`).
