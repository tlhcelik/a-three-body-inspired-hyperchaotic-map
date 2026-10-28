"""Transient chaos at a = 5.2: settling times and finite-time exponent of the transient."""
import json, numpy as np
from rgmap.core import Params, step, REF_STATE
rng = np.random.default_rng(8); M = 4000
Q = np.repeat(REF_STATE[None], M, 0).copy(); Q[:, 0, :2] = rng.uniform(-1, 1, (M, 2))
P = Params(np.full(M, 5.2))
hist = [Q]
for _ in range(3002):
    Q = step(Q, P); hist.append(Q)
H = np.array(hist)                                  # (T, M, 3, 3)
d2 = np.linalg.norm((H[2:] - H[:-2]).reshape(len(H) - 2, M, -1), axis=2)   # period-2 residual
settled = d2 < 1e-6
# first time after which it stays settled
T = np.full(M, -1)
for m in range(M):
    idx = np.where(~settled[:, m])[0]
    T[m] = 0 if len(idx) == 0 else idx[-1] + 1
# finite-time exponent during the transient (two trajectories, delta0 = 1e-10)
X0 = np.repeat(REF_STATE[None], 500, 0).copy(); X0[:, 0, :2] = rng.uniform(-1, 1, (500, 2))
v = rng.normal(size=X0.shape); v *= 1e-10 / np.linalg.norm(v.reshape(500, -1), axis=1)[:, None, None]
A, B = X0, X0 + v; P5 = Params(np.full(500, 5.2)); sep = []
for _ in range(40):
    A = step(A, P5); B = step(B, P5); sep.append(np.median(np.linalg.norm((B - A).reshape(500, -1), axis=1)))
sep = np.array(sep); n_sat = int(np.argmax(sep > 1e-2)) + 1
lam_tr = float(np.log(sep[n_sat - 1] / 1e-10) / n_sat)
out = dict(T=T.tolist(), T_median=float(np.median(T)), T_q=[float(np.percentile(T, q)) for q in (10, 25, 75, 90)],
           T_max=int(T.max()), frac_unsettled=float(np.mean(T >= 3000)), n_sat=n_sat, lam_transient=lam_tr)
print({k: v for k, v in out.items() if k != "T"}); json.dump(out, open("results/transient.json", "w"), indent=1)
