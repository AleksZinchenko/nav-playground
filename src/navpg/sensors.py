"""Прості моделі сенсорів для симуляції."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ImuErrorModel:
    """Похибки одного каналу IMU: постійний bias, білий шум, random walk bias."""

    bias: float = 0.0
    noise_std: float = 0.0  # стандартне відхилення білого шуму на крок
    bias_rw_std: float = 0.0  # СКВ приросту bias на крок

    def corrupt(self, truth, rng):
        """Додає похибки до істинного сигналу (масив по часу)."""
        truth = np.asarray(truth, dtype=float)
        bias = self.bias + np.cumsum(rng.normal(0.0, self.bias_rw_std, truth.shape))
        return truth + bias + rng.normal(0.0, self.noise_std, truth.shape)


def simulate_gnss(true_pos, noise_std, rng, outages=()):
    """Зашумлені GNSS-виміри позиції; NaN під час провалів сигналу.

    true_pos: (T, D) масив; outages: список (start, stop) індексів.
    """
    meas = np.asarray(true_pos, dtype=float) + rng.normal(0.0, noise_std, np.shape(true_pos))
    for start, stop in outages:
        meas[start:stop] = np.nan
    return meas
