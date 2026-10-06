"""2D рух з постійною швидкістю: KF на GNSS-вимірах з провалом сигналу.

Під час провалу фільтр лише прогнозує — видно, як росте невизначеність
і похибка. Це мотивація для GNSS/INS fusion у наступних модулях.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navpg.filters import KalmanFilter
from navpg.metrics import position_error_norm, rmse
from navpg.sensors import simulate_gnss

rng = np.random.default_rng(42)
dt, T = 0.1, 600
outage = (250, 400)

# Істинна траєкторія: x = [px, py, vx, vy], повільний поворот
t = np.arange(T) * dt
truth = np.column_stack([10 * np.sin(0.05 * t) * 20, 3 * t, 10 * np.cos(0.05 * t), 3 * np.ones(T)])
gnss = simulate_gnss(truth[:, :2], noise_std=3.0, rng=rng, outages=[outage])

F = np.eye(4)
F[0, 2] = F[1, 3] = dt
H = np.hstack([np.eye(2), np.zeros((2, 2))])
q = 0.5
G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
kf = KalmanFilter(F, H, Q=q**2 * G @ G.T, R=9.0 * np.eye(2), x0=np.r_[gnss[0], 0, 0], P0=100 * np.eye(4))

est, sigma = [], []
for z in gnss:
    kf.predict()
    if not np.isnan(z).any():
        kf.update(z)
    est.append(kf.x.copy())
    sigma.append(np.sqrt(kf.P[0, 0] + kf.P[1, 1]))
est, sigma = np.array(est), np.array(sigma)

err = position_error_norm(est[:, :2], truth[:, :2])
print(f"RMSE позиції (весь час):     {rmse(est[:, :2], truth[:, :2]).round(2)} м")
print(f"Макс. похибка під час outage: {err[outage[0]:outage[1]].max():.2f} м")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(truth[:, 0], truth[:, 1], "k-", label="truth")
a1.plot(gnss[:, 0], gnss[:, 1], ".", ms=2, alpha=0.4, label="GNSS")
a1.plot(est[:, 0], est[:, 1], "r-", label="KF")
a1.set_aspect("equal"), a1.legend(), a1.set_title("Траєкторія")
a2.plot(t, err, label="|похибка|")
a2.plot(t, sigma, "--", label="σ з P")
a2.axvspan(t[outage[0]], t[outage[1]], color="gray", alpha=0.2, label="GNSS outage")
a2.set_xlabel("t, с"), a2.set_ylabel("м"), a2.legend(), a2.set_title("Похибка позиції")
fig.tight_layout()
fig.savefig("01_kf_gnss_outage.png", dpi=120)
print("Збережено 01_kf_gnss_outage.png")
