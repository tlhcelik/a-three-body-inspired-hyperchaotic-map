"""Vectorised implementation of the three-body-inspired bounded hyperchaotic map.

State: array of shape (B, N, 3) -- B independent copies (parameter values or
initial conditions), N bodies, 3 coordinates.  All functions are batched over B.

The map (paper Eqs. map definition):
    r~_ij   = sqrt(|r_i - r_j|^2 + eps^2)
    Phi_i   = -G sum_{j!=i} m_j / r~_ij
    F_ik    =  G sum_{j!=i} m_i m_j (q_jk - q_ik) / r~_ij^p        (p = 3: Newton)
    B_i     = 1 - |r_i|^2 / R2
    D_ik    = a q_ik (1 + Phi_i) + B_i q_ik + F_ik
    q_ik'   = fold(G / D_ik),  fold = sin by default

Options allow the ablations / controls used in the paper:
    use_phi=False         -> Phi_i = 0
    force_scale           -> multiplies F (0 removes it)
    p                     -> distance exponent of the pair force (3 = Newton, 0 = linear diffusive)
    W                     -> if given (3N x 3N), replaces F by force_scale * W @ q (random static coupling)
    offset                -> if given (N,3), replaces F by a constant, state-independent offset (no coupling)
    fold in {"sin", "tri", "saw", "tanh", "none"} (Jacobian available for sin and tri)
"""
import numpy as np

G = 1.0


class Params:
    def __init__(self, a, eps=0.1, R2=2.0, masses=(1.0, 1.0, 1.0), p=3.0,
                 use_phi=True, force_scale=1.0, W=None, fold="sin", offset=None):
        self.a = np.atleast_1d(np.asarray(a, dtype=float))
        self.eps = np.atleast_1d(np.asarray(eps, dtype=float))
        self.R2 = float(R2)
        self.m = np.asarray(masses, dtype=float)
        self.N = len(self.m)
        self.p = float(p)
        self.use_phi = use_phi
        self.force_scale = float(force_scale)
        self.W = W
        self.fold = fold
        self.offset = None if offset is None else np.asarray(offset, dtype=float)


def _pair(Q, P):
    d = Q[:, None, :, :] - Q[:, :, None, :]            # d[b,i,j,:] = q_j - q_i
    r2 = (d ** 2).sum(-1) + (P.eps ** 2)[:, None, None]
    rt = np.sqrt(r2)
    N = Q.shape[1]
    off = ~np.eye(N, dtype=bool)
    return d, rt, off


def terms(Q, P):
    """Return Phi (B,N), F (B,N,3), Bnd (B,N), D (B,N,3)."""
    d, rt, off = _pair(Q, P)
    m = P.m
    inv = np.where(off, 1.0 / rt, 0.0)
    if P.use_phi:
        Phi = -G * (inv * m[None, None, :]).sum(-1)
    else:
        Phi = np.zeros(Q.shape[:2])
    if P.offset is not None:
        F = P.force_scale * np.broadcast_to(P.offset, Q.shape)
    elif P.W is not None:
        Bsz = Q.shape[0]
        F = P.force_scale * (Q.reshape(Bsz, -1) @ P.W.T).reshape(Q.shape)
    else:
        w = np.where(off, rt ** (-P.p), 0.0) * m[None, None, :] * m[None, :, None]
        F = P.force_scale * G * (w[..., None] * d).sum(2)
    Bnd = 1.0 - (Q ** 2).sum(-1) / P.R2
    D = P.a[:, None, None] * Q * (1.0 + Phi)[..., None] + Bnd[..., None] * Q + F
    return Phi, F, Bnd, D


def _fold(x, kind):
    if kind == "sin":
        return np.sin(x)
    if kind == "tri":
        return (2.0 / np.pi) * np.arcsin(np.sin(x))
    if kind == "saw":
        return np.mod(x + np.pi, 2 * np.pi) / np.pi - 1.0
    if kind == "tanh":
        return np.tanh(x)
    if kind == "none":
        return x
    raise ValueError(kind)


def _dfold(x, kind):
    if kind == "sin":
        return np.cos(x)
    if kind == "tri":
        return (2.0 / np.pi) * np.sign(np.cos(x))
    raise ValueError(kind)


def safe(D):
    return np.where(np.abs(D) < 1e-12, np.where(D >= 0, 1e-12, -1e-12), D)


def step(Q, P):
    D = safe(terms(Q, P)[3])
    return _fold(G / D, P.fold)


def dD_dq(Q, P):
    """Jacobian of D w.r.t. the state, shape (B, 3N, 3N) (row (i,k), column (l,c))."""
    Bsz, N, _ = Q.shape
    d, rt, off = _pair(Q, P)
    m = P.m
    Phi, F, Bnd, D = terms(Q, P)
    a = P.a
    I3 = np.eye(3)
    eyeN = np.eye(N)
    J = np.zeros((Bsz, N, 3, N, 3))
    # diagonal term delta_il delta_kc [a(1+Phi_i) + B_i]
    diag = a[:, None] * (1.0 + Phi) + Bnd                     # (B,N)
    J += diag[:, :, None, None, None] * eyeN[None, :, None, :, None] * I3[None, None, :, None, :]
    # dPhi[b,i,l,c] = -G [delta_il sum_j m_j d_ijc / r^3 - m_l d_ilc / r_il^3]
    if P.use_phi:
        inv3 = np.where(off, rt ** -3.0, 0.0)
        S = (m[None, None, :, None] * d * inv3[..., None])      # (B,i,j,c)
        dPhi = -G * (eyeN[None, :, :, None] * S.sum(2)[:, :, None, :] - S)
    else:
        dPhi = np.zeros((Bsz, N, N, 3))
    # dB[b,i,l,c] = -2 delta_il q_ic / R2
    dB = -2.0 * eyeN[None, :, :, None] * Q[:, :, None, :] / P.R2
    J += Q[:, :, :, None, None] * (a[:, None, None, None] * dPhi + dB)[:, :, None, :, :]
    # force
    if P.offset is not None:
        pass
    elif P.W is not None:
        J += P.force_scale * np.broadcast_to(P.W.reshape(N, 3, N, 3), J.shape)
    elif P.force_scale != 0.0:
        p = P.p
        rp = np.where(off, rt ** (-p), 0.0)
        rp2 = np.where(off, rt ** (-p - 2.0), 0.0)
        T = (-I3[None, None, None] * rp[..., None, None]
             + p * d[..., :, None] * d[..., None, :] * rp2[..., None, None])   # (B,i,j,k,c)
        Tm = T * m[None, None, :, None, None]
        dF = G * m[None, :, None, None, None] * (
            eyeN[None, :, None, :, None] * Tm.sum(2)[:, :, :, None, :]
            - np.transpose(Tm, (0, 1, 3, 2, 4)))                               # (B,i,k,l,c)
        J += P.force_scale * dF
    return J.reshape(Bsz, 3 * N, 3 * N), D


def jacobian(Q, P):
    """Full map Jacobian (B,3N,3N) and the folding factor f'(G/D) G/D^2 (B,3N)."""
    JD, D = dD_dq(Q, P)
    D = safe(D).reshape(Q.shape[0], -1)
    g = -_dfold(G / D, P.fold) * G / D ** 2
    return g[:, :, None] * JD, g, JD


def iterate(Q, P, n):
    for _ in range(n):
        Q = step(Q, P)
    return Q


def lyap_max(Q, P, n_tr=1000, n=10000, d0=1e-8, seed=0):
    """Two-trajectory (Benettin) estimate of the largest exponent, batched."""
    rng = np.random.default_rng(seed)
    Q = iterate(Q, P, n_tr)
    v = rng.normal(size=Q.shape)
    v *= d0 / np.linalg.norm(v.reshape(len(v), -1), axis=1)[:, None, None]
    X = Q + v
    s = np.zeros(len(Q))
    for _ in range(n):
        Q = step(Q, P); X = step(X, P)
        dv = X - Q
        dist = np.linalg.norm(dv.reshape(len(dv), -1), axis=1)
        dist = np.where(dist == 0, 1e-300, dist)
        s += np.log(dist / d0)
        X = Q + dv * (d0 / dist)[:, None, None]
    return s / n


def spectrum(Q, P, n_tr=1000, n=10000, return_decomp=False):
    """QR Lyapunov spectrum, batched. Optionally the exact decomposition of the
    exponent sum:  sum_i lambda_i = <sum ln|f'(G/D) G/D^2|> + <ln|det dD/dq|>."""
    Q = iterate(Q, P, n_tr)
    Bsz = len(Q); n9 = 3 * P.N
    Bm = np.broadcast_to(np.eye(n9), (Bsz, n9, n9)).copy()
    s = np.zeros((Bsz, n9))
    fold_sum = np.zeros(Bsz); coup_sum = np.zeros(Bsz)
    for _ in range(n):
        J, g, JD = jacobian(Q, P)
        Q = step(Q, P)
        Bm, R = np.linalg.qr(J @ Bm)
        s += np.log(np.abs(np.diagonal(R, axis1=1, axis2=2)))
        if return_decomp:
            fold_sum += np.log(np.abs(g)).sum(1)
            coup_sum += np.linalg.slogdet(JD)[1]
    lam = -np.sort(-s / n, axis=1)
    if return_decomp:
        return lam, fold_sum / n, coup_sum / n
    return lam


REF_STATE = np.array([[-0.1255, 0.4507, 0.232],
                      [0.0987, -0.3440, -0.3440],
                      [-0.4419, 0.3662, 0.1011]])


def ensemble(n_ic, seed=2026, N=3, spread=0.15, base=None):
    rng = np.random.default_rng(seed)
    if base is None:
        if N == 3:
            base = REF_STATE
        else:
            base = np.random.default_rng(1000 + N).uniform(-0.5, 0.5, (N, 3))
    return base[None] + rng.uniform(-spread, spread, (n_ic, N, 3))
