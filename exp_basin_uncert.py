"""Uncertainty exponent of the basin boundaries at a = 5.2 (McDonald et al. 1985).
For random initial states x and perturbed states x + h e (e: unit vector along x1), the fraction
f(h) of pairs converging to different period-2 attractors scales as h^alpha.
Outputs results/basin_uncert.json"""
import json, numpy as np
from rgmap.core import Params, iterate, step, REF_STATE

rng = np.random.default_rng(21)
M = 4000
hs = np.logspace(-10, -1, 10)

def label(Q, P):
    Q = iterate(Q, P, 3000); Q2 = step(Q, P)
    k1 = np.round(Q.reshape(len(Q), -1), 4); k2 = np.round(Q2.reshape(len(Q), -1), 4)
    return [min(tuple(u), tuple(v)) for u, v in zip(k1, k2)]

base = np.repeat(REF_STATE[None], M, 0).copy()
base[:, 0, 0] = rng.uniform(-1, 1, M); base[:, 0, 1] = rng.uniform(-1, 1, M)
P = Params(np.full(M, 5.2))
L0 = label(base, P)
frac = []
for h in hs:
    X = base.copy(); X[:, 0, 0] += h
    L1 = label(X, P)
    frac.append(float(np.mean([u != v for u, v in zip(L0, L1)])))
    print(h, frac[-1], flush=True)
frac = np.array(frac)
m = (frac > 0) & (frac < 0.9)
alpha = float(np.polyfit(np.log(hs[m]), np.log(frac[m]), 1)[0]) if m.sum() >= 2 else float("nan")
print("alpha", alpha, "n attractors", len(set(L0)))
json.dump(dict(h=hs.tolist(), frac=frac.tolist(), alpha=alpha, n_attr=len(set(L0)), M=M),
          open("results/basin_uncert.json", "w"), indent=1)
