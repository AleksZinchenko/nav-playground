import numpy as np

from navpg.coordinates import ecef_to_lla, lla_to_ecef, lla_to_enu


def test_ecef_roundtrip():
    lat, lon, h = 50.4501, 30.5234, 179.0  # Київ
    lat2, lon2, h2 = ecef_to_lla(*lla_to_ecef(lat, lon, h))
    assert np.allclose([lat2, lon2], [lat, lon], atol=1e-9)
    assert abs(h2 - h) < 1e-3


def test_enu_up_and_north():
    up = lla_to_enu(50.0, 30.0, 100.0, 50.0, 30.0, 0.0)
    assert np.allclose(up, [0, 0, 100], atol=1e-6)
    north = lla_to_enu(50.001, 30.0, 0.0, 50.0, 30.0, 0.0)
    assert north[1] > 100 and abs(north[0]) < 1e-6
