"""Лінійний Kalman Filter."""

import numpy as np


class KalmanFilter:
    """x_k = F x_{k-1} + B u + w,  w ~ N(0, Q);  z = H x + v,  v ~ N(0, R)."""

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F, self.H, self.Q, self.R, self.B = (np.atleast_2d(m) for m in (F, H, Q, R, B if B is not None else np.zeros((len(x0), 1))))
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.atleast_2d(P0).astype(float).copy()

    def predict(self, u=None):
        self.x = self.F @ self.x
        if u is not None:
            self.x = self.x + self.B @ np.atleast_1d(u)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.atleast_1d(z)
        y = z - self.H @ self.x  # інновація
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        # Форма Джозефа — чисельно стабільніша
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x
