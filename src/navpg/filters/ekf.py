"""Extended Kalman Filter з довільними нелінійними моделями."""

import numpy as np


class ExtendedKalmanFilter:
    """f(x, u) — модель руху, h(x) — модель виміру; F_jac, H_jac — їх якобіани."""

    def __init__(self, f, h, F_jac, H_jac, Q, R, x0, P0):
        self.f, self.h, self.F_jac, self.H_jac = f, h, F_jac, H_jac
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.atleast_2d(P0).astype(float).copy()

    def predict(self, u=None):
        F = self.F_jac(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z):
        H = self.H_jac(self.x)
        y = np.atleast_1d(z) - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x
