"""Метрики якості навігації."""

import numpy as np


def rmse(estimate, truth, axis=0):
    """Root Mean Square Error по осі часу."""
    err = np.asarray(estimate) - np.asarray(truth)
    return np.sqrt(np.mean(err ** 2, axis=axis))


def position_error_norm(estimate, truth):
    """Евклідова похибка позиції на кожному кроці; масиви (T, D)."""
    return np.linalg.norm(np.asarray(estimate) - np.asarray(truth), axis=1)
