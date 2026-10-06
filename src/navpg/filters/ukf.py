"""Unscented Kalman Filter (сигма-точки Van der Merwe)."""

import numpy as np


class UnscentedKalmanFilter:
    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-3, beta=2.0, kappa=0.0):
        self.f, self.h = f, h
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.atleast_2d(P0).astype(float).copy()
        n = len(self.x)
        self.lam = alpha ** 2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)

    def _sigma_points(self):
        n = len(self.x)
        L = np.linalg.cholesky((n + self.lam) * self.P)
        return np.vstack([self.x, self.x + L.T, self.x - L.T])

    def predict(self, u=None):
        self._X = np.array([self.f(s, u) for s in self._sigma_points()])
        self.x = self.Wm @ self._X
        d = self._X - self.x
        self.P = d.T @ np.diag(self.Wc) @ d + self.Q
        return self.x

    def update(self, z):
        X = self._sigma_points()
        Z = np.array([np.atleast_1d(self.h(s)) for s in X])
        z_pred = self.Wm @ Z
        dz, dx = Z - z_pred, X - self.x
        S = dz.T @ np.diag(self.Wc) @ dz + self.R
        Pxz = dx.T @ np.diag(self.Wc) @ dz
        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (np.atleast_1d(z) - z_pred)
        self.P = self.P - K @ S @ K.T
        return self.x
