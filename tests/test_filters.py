import numpy as np

from navpg.filters import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter
from navpg.metrics import rmse


def _const_velocity(steps=200, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.column_stack([np.arange(steps) * dt * 2.0, np.full(steps, 2.0)])
    z = truth[:, 0] + rng.normal(0, 1.0, steps)
    F = np.array([[1, dt], [0, 1]])
    return truth, z, F


def test_kf_beats_raw_measurements():
    truth, z, F = _const_velocity()
    kf = KalmanFilter(F, [[1, 0]], Q=1e-4 * np.eye(2), R=[[1.0]], x0=[0, 0], P0=10 * np.eye(2))
    est = []
    for zi in z:
        kf.predict()
        est.append(kf.update(zi)[0])
    assert rmse(est[50:], truth[50:, 0]) < rmse(z[50:], truth[50:, 0])


def test_ekf_and_ukf_match_kf_on_linear_problem():
    truth, z, F = _const_velocity()
    args = dict(Q=1e-4 * np.eye(2), R=[[1.0]], x0=[0, 0], P0=10 * np.eye(2))
    kf = KalmanFilter(F, [[1, 0]], **args)
    ekf = ExtendedKalmanFilter(
        f=lambda x, u: F @ x, h=lambda x: x[:1],
        F_jac=lambda x, u: F, H_jac=lambda x: np.array([[1.0, 0.0]]), **args)
    ukf = UnscentedKalmanFilter(f=lambda x, u: F @ x, h=lambda x: x[:1], **args)
    for zi in z:
        for flt in (kf, ekf, ukf):
            flt.predict()
            flt.update(zi)
    assert np.allclose(kf.x, ekf.x)
    assert np.allclose(kf.x, ukf.x, atol=1e-6)
