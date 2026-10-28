"""Quantitative comparison with well-known maps under one protocol.
Outputs results/compare.json"""
import json, math, time, numpy as np
from rgmap.core import Params, ensemble, lyap_max, spectrum, iterate

THR = 0.057
rng = np.random.default_rng(5)


def qr_spectrum(f, jac, x0, n_tr=1000, n=20000):
    x = np.array(x0, float)
    for _ in range(n_tr):
        x = f(x)
    d = len(x); Bm = np.eye(d); s = np.zeros(d); xs = []
    for _ in range(n):
        J = jac(x); x = f(x); xs.append(x[0])
        Bm, R = np.linalg.qr(J @ Bm); s += np.log(np.abs(np.diag(R)) + 1e-300)
        if not np.all(np.isfinite(x)):
            return None, None
    return np.sort(s / n)[::-1], np.array(xs)


def perm_entropy(x, m=6):
    idx = np.lib.stride_tricks.sliding_window_view(x, m)
    pats = np.argsort(idx, axis=1, kind="stable")
    _, c = np.unique(pats, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


maps = {}
# 1-D maps -----------------------------------------------------------------
maps["Logistic"] = dict(rng=(0.0, 4.0), x0=[0.3141],
    f=lambda r: (lambda x: np.array([r * x[0] * (1 - x[0])])),
    J=lambda r: (lambda x: np.array([[r * (1 - 2 * x[0])]])), tsin=0, dim=1, ref="classical")
maps["Sine"] = dict(rng=(0.0, 1.0), x0=[0.3141],
    f=lambda u: (lambda x: np.array([u * np.sin(np.pi * x[0])])),
    J=lambda u: (lambda x: np.array([[u * np.pi * np.cos(np.pi * x[0])]])), tsin=1, dim=1, ref="classical")
maps["LSS (Zhou 2014)"] = dict(rng=(0.0, 4.0), x0=[0.3141],
    f=lambda r: (lambda x: np.array([(r * x[0] * (1 - x[0]) + (4 - r) * np.sin(np.pi * x[0]) / 4) % 1.0])),
    J=lambda r: (lambda x: np.array([[r * (1 - 2 * x[0]) + (4 - r) * np.pi * np.cos(np.pi * x[0]) / 4]])),
    tsin=1, dim=1, ref="zhou2014new1d")
# 2-D maps -----------------------------------------------------------------
def lasm_f(u):
    def f(v):
        x, y = v
        xn = np.sin(np.pi * u * (y + 3) * x * (1 - x))
        yn = np.sin(np.pi * u * (xn + 3) * y * (1 - y))
        return np.array([xn, yn])
    return f
def lasm_J(u):
    def J(v):
        x, y = v
        A = np.pi * u * (y + 3) * x * (1 - x); xn = np.sin(A)
        dxn_dx = np.cos(A) * np.pi * u * (y + 3) * (1 - 2 * x)
        dxn_dy = np.cos(A) * np.pi * u * x * (1 - x)
        Bv = np.pi * u * (xn + 3) * y * (1 - y)
        dyn_dxn = np.cos(Bv) * np.pi * u * y * (1 - y)
        dyn_dy_direct = np.cos(Bv) * np.pi * u * (xn + 3) * (1 - 2 * y)
        return np.array([[dxn_dx, dxn_dy],
                         [dyn_dxn * dxn_dx, dyn_dxn * dxn_dy + dyn_dy_direct]])
    return J
maps["2D-LASM (Hua 2016)"] = dict(rng=(0.0, 1.0), x0=[0.3141, 0.2718], f=lasm_f, J=lasm_J,
                                  tsin=1, dim=2, ref="hua2016lasm")
maps["Henon"] = dict(rng=(1.0, 1.4), x0=[0.1, 0.1], p_ref=1.4,
    f=lambda a: (lambda v: np.array([1 - a * v[0] ** 2 + v[1], 0.3 * v[0]])),
    J=lambda a: (lambda v: np.array([[-2 * a * v[0], 1.0], [0.3, 0.0]])), tsin=0, dim=2, ref="henon1976")
# 3-D hyperchaotic generalised Henon map (Baier & Klein 1990), b = 0.1
maps["Gen. Henon 3D (Baier-Klein)"] = dict(rng=(1.0, 1.8), x0=[0.1, 0.1, 0.1], p_ref=1.76,
    f=lambda a: (lambda v: np.array([a - v[1] ** 2 - 0.1 * v[2], v[0], v[1]])),
    J=lambda a: (lambda v: np.array([[0.0, -2 * v[1], -0.1], [1.0, 0, 0], [0, 1.0, 0]])),
    tsin=0, dim=3, ref="baier1990maximum")

out = {}
grid_n = 141
for name, M in maps.items():
    lo, hi = M["rng"]; grid = np.linspace(lo, hi, grid_n)
    lam1 = []
    for g in grid:
        sp, _ = qr_spectrum(M["f"](g), M["J"](g), M["x0"], n_tr=500, n=3000)
        lam1.append(np.nan if sp is None else sp[0])
    lam1 = np.array(lam1)
    ratio = float(np.nanmean(np.where(np.isnan(lam1), 0, lam1 > THR)))
    g_star = M.get("p_ref", grid[int(np.nanargmax(lam1))])
    sp, xs = qr_spectrum(M["f"](g_star), M["J"](g_star), M["x0"], n_tr=1000, n=20000)
    t = time.perf_counter(); f = M["f"](g_star); x = np.array(M["x0"], float)
    for _ in range(20000): x = f(x)
    tpo = (time.perf_counter() - t) / 20000 / M["dim"]
    out[name] = dict(ref=M["ref"], dim=M["dim"], param_range=[lo, hi], p_star=float(g_star),
                     lam_max=float(sp[0]), n_pos=int((sp > THR).sum()),
                     ks_bound=float(sp[sp > 0].sum()), chaotic_ratio=ratio,
                     pe6=perm_entropy(xs), sin_per_output=M["tsin"], us_per_output=tpo * 1e6)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)

# proposed map (scalar loop timing for fairness, batched code for exponents)
P1 = Params([0.97]); sp = spectrum(ensemble(5), Params(np.full(5, 0.97)), n_tr=1000, n=10000).mean(0)
Q = iterate(ensemble(1), P1, 1000); xs = []
for _ in range(20000):
    Q = iterate(Q, P1, 1); xs.append(Q[0, 0, 0])
from rgmap.core import step
t = time.perf_counter()
for _ in range(20000): Q = step(Q, P1)
tpo = (time.perf_counter() - t) / 20000 / 9
mech = json.load(open("results/mechanism.json"))
out["Proposed (N=3)"] = dict(ref="this work", dim=9, param_range=[-6, 8], p_star=0.97,
                             lam_max=float(sp[0]), n_pos=int((sp > THR).sum()),
                             ks_bound=float(sp[sp > 0].sum()),
                             chaotic_ratio=mech["newton"]["chaotic_ratio"], pe6=perm_entropy(np.array(xs)),
                             sin_per_output=1, us_per_output=tpo * 1e6)
print("Proposed", out["Proposed (N=3)"])
json.dump(out, open("results/compare.json", "w"), indent=1)
