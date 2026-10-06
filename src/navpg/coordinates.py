"""Системи координат: WGS-84 геодезичні (LLA), ECEF, локальна ENU."""

import numpy as np

# WGS-84
A = 6378137.0  # велика піввісь, м
F = 1 / 298.257223563  # стиснення
B = A * (1 - F)
E2 = F * (2 - F)  # квадрат першого ексцентриситету


def lla_to_ecef(lat_deg, lon_deg, h):
    """Геодезичні (град, град, м) -> ECEF (м)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + h) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_lla(x, y, z, iterations=5):
    """ECEF (м) -> геодезичні (град, град, м). Ітеративний метод."""
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    h = 0.0
    for _ in range(iterations):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


def ecef_to_enu_matrix(lat_deg, lon_deg):
    """Матриця обертання ECEF -> ENU у точці (lat, lon)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def lla_to_enu(lat_deg, lon_deg, h, ref_lat_deg, ref_lon_deg, ref_h):
    """Геодезичні -> ENU (м) відносно опорної точки."""
    d = lla_to_ecef(lat_deg, lon_deg, h) - lla_to_ecef(ref_lat_deg, ref_lon_deg, ref_h)
    return ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg) @ d


def rot_z(yaw):
    """Обертання навколо осі Z (рад)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
