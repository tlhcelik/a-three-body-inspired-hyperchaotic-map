"""Two-parameter Lyapunov diagrams (a, eps_s) and (a, R^2), and basin slices.
Outputs results/twoparam.npz"""
import numpy as np
from rgmap.core import Params, ensemble, lyap_max, iterate, jacobian, step, REF_STATE

A = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
EPS = np.geomspace(0.005, 0.5, 34)
R2 = np.geomspace(0.25, 32.0, 34)
NIC = 3
ic = ensemble(NIC, seed=99)

def plane(P_of, ys):
    aa, yy = np.meshgrid(A, ys)                     # (ny, na)
    a = np.repeat(aa.ravel(), NIC); y = np.repeat(yy.ravel(), NIC)
    Q0 = np.tile(ic, (aa.size, 1, 1))
    with np.errstate(all="ignore"):
        lam = lyap_max(Q0, P_of(a, y), n_tr=500, n=2500)
    lam = np.where(np.isfinite(lam), lam, np.nan).reshape(aa.size, NIC)
    return np.nanmedian(lam, axis=1).reshape(aa.shape)

L_eps = plane(lambda a, y: Params(a, eps=y), EPS); print("eps plane done", flush=True)

def lyap_R2(a, y):
    # R2 is a scalar in Params; group by R2 value
    out = np.empty(len(a))
    for v in np.unique(y):
        idx = np.where(y == v)[0]
        with np.errstate(all="ignore"):
            out[idx] = lyap_max(np.tile(ic, (len(idx) // NIC, 1, 1)), Params(a[idx], R2=v), n_tr=500, n=2500)
    return out

aa, yy = np.meshgrid(A, R2)
a = np.repeat(aa.ravel(), NIC); y = np.repeat(yy.ravel(), NIC)
lam = lyap_R2(a, y)
lam = np.where(np.isfinite(lam), lam, np.nan).reshape(aa.size, NIC)
L_R2 = np.nanmedian(lam, axis=1).reshape(aa.shape); print("R2 plane done", flush=True)

# ---- basin slices: vary (x1, y1) of body 1, other coordinates at the reference state
n = 150
g = np.linspace(-1, 1, n)
X, Y = np.meshgrid(g, g)
Q0 = np.repeat(REF_STATE[None], n * n, axis=0)
Q0[:, 0, 0] = X.ravel(); Q0[:, 0, 1] = Y.ravel()

# a = 5.2: identify which period-2 cycle is reached
P = Params(np.full(n * n, 5.2))
Q = iterate(Q0, P, 3000)
Q2 = step(Q, P)
key = np.concatenate([np.round(Q.reshape(len(Q), -1), 4), np.round(Q2.reshape(len(Q), -1), 4)], 1)
# canonical: order the two cycle points so that both phases give the same label
k1 = np.round(Q.reshape(len(Q), -1), 4); k2 = np.round(Q2.reshape(len(Q), -1), 4)
lab_str = [min(tuple(u), tuple(v)) for u, v in zip(k1, k2)]
uniq = {}
labels = np.array([uniq.setdefault(s, len(uniq)) for s in lab_str]).reshape(n, n)
print("a=5.2 attractors in slice:", len(uniq), flush=True)

# a = -1: first two exponents (QR with 2 vectors) to separate the coexisting attractors
P = Params(np.full(n * n, -1.0))
Q = iterate(Q0, P, 1000)
V = np.broadcast_to(np.eye(9)[:, :2], (n * n, 9, 2)).copy()
s = np.zeros((n * n, 2)); NIT = 3000
for _ in range(NIT):
    J, _, _ = jacobian(Q, P); Q = step(Q, P)
    V, Rm = np.linalg.qr(J @ V)
    s += np.log(np.abs(np.diagonal(Rm, axis1=1, axis2=2)))
l12 = (s / NIT).reshape(n, n, 2)
print("a=-1 lambda2>thr fraction:", float(np.mean(l12[..., 1] > 0.057)), flush=True)

np.savez("results/twoparam.npz", A=A, EPS=EPS, R2=R2, L_eps=L_eps, L_R2=L_R2,
         basin_grid=g, basin52=labels, basin_m1=l12)
