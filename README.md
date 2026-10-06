# nav-playground 🚁

Навчальний проєкт до курсу «Інженерія навігаційних систем»:
від IMU та GNSS до системи, яка продовжує орієнтуватися без стабільного сигналу.

**Фінальна ціль:** навігаційна система, що об'єднує кілька сенсорів і
продовжує оцінювати положення під час втрати GNSS, з валідацією
через ground truth та RMSE.

## Структура

```
src/navpg/
  coordinates.py     # WGS-84, LLA <-> ECEF <-> ENU, обертання
  sensors.py         # моделі похибок IMU, симуляція GNSS з провалами сигналу
  metrics.py         # RMSE, похибки позиції
  filters/
    kf.py            # лінійний Kalman Filter
    ekf.py           # Extended KF
    ukf.py           # Unscented KF (сигма-точки Merwe)
examples/            # скрипти, що запускаються
notebooks/           # Jupyter для експериментів
tests/               # pytest
cpp/                 # C++ реалізації (порт з Python)
ros2_ws/             # ROS2 workspace (colcon)
data/                # датасети (не в git)
docs/                # конспекти
```

## Старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/01_kf_gnss_outage.py
```

## Roadmap (за модулями курсу)

- [ ] **1. Системи координат**: WGS-84, ECEF, ENU/NED, кватерніони, кути Ейлера
- [ ] **2. Похибки сенсорів**: bias, білий шум, random walk, Allan variance
- [ ] **3. Kalman Filter**: лінійний KF, налаштування Q/R, NEES/NIS
- [ ] **4. EKF / UKF**: нелінійні моделі, порівняння на одній задачі
- [ ] **5. INS mechanization**: інтегрування IMU, накопичення дрейфу
- [ ] **6. GNSS/INS**: loosely coupled fusion, поведінка під час втрати GNSS
- [ ] **7. SLAM / VIO**: запуск на EuRoC, порівняння з ground truth
- [ ] **8. ROS2**: ноди для фільтра, топіки IMU/GNSS, rosbag replay
- [ ] **9. Валідація**: RMSE, ATE/RPE, звіт
- [ ] **10. Фінал**: multi-sensor fusion + GNSS outage
