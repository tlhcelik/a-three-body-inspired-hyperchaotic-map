"""Helpers for the coupling-control experiments: several ways of matching the
magnitude of a replacement coupling to that of the Newtonian pair force.

The original analysis (exp_mechanism.py) equalises the root-mean-square value of the
coupling term F on 4000 states of the reference attractor.  Here the same controls can be
equalised by other criteria:

    "rms"     root-mean-square of F on the reference attractor      (original)
    "medF"    median |F| on the reference attractor
    "medD"    median |D| of the denominator on the control's *own* attractor at a = 0.97
              (the quantity the fold actually sees)
"""
import numpy as np
from .core import Params, ensemble, iterate, terms, lyap_max, spectrum

A_REF = 0.97
A_GRID = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
THR = 0.057
CRITERIA = ("rms", "medF", "medD")


def attractor_sample(n_ic=20, n_tr=1000, n_samp=200):
    """4000 states of the reference (Newtonian, a = 0.97) attractor."""
    P = Params(np.full(n_ic, A_REF))
    Q = iterate(ensemble(n_ic), P, n_tr)
    out = []
    for _ in range(n_samp):
        Q = iterate(Q, P, 1)
        out.append(Q.copy())
    return np.concatenate(out)


def unit_kw(kind, seed=0):
    """Parameters of a control at force_scale = 1 (before magnitude matching)."""
    r = np.random.default_rng(seed)
    if kind == "newton":
        return dict(p=3.0)
    if kind.startswith("p="):
        return dict(p=float(kind[2:]))
    if kind == "linear":
        return dict(p=0.0)
    if kind == "random_linear":
        return dict(W=r.normal(size=(9, 9)))
    if kind == "offset":
        return dict(offset=r.normal(size=(3, 3)))
    raise ValueError(kind)


def coupling_stat(S, kw, how):
    """Magnitude statistic of the coupling term of a unit-scale control on the states S."""
    F = terms(S, Params(np.full(len(S), A_REF), **kw))[1]
    if how == "rms":
        return float(np.sqrt((F ** 2).mean()))
    if how == "medF":
        return float(np.median(np.abs(F)))
    raise ValueError(how)


def own_D_median(kw, n_ic=20):
    """Median |D| on the control's own attractor at a = 0.97."""
    P = Params(np.full(n_ic, A_REF), **kw)
    Q = iterate(ensemble(n_ic), P, 1000)
    Ds = []
    for _ in range(100):
        Ds.append(np.abs(terms(Q, P)[3]).ravel())
        Q = iterate(Q, P, 1)
    return float(np.median(np.concatenate(Ds)))


def match_medD(kw_unit, target, s0, sign=1.0, nbis=9):
    """Scale s (force_scale = sign * s) for which the control's own median |D| equals `target`.

    A coarse log-grid around the rms-matched value s0 locates a sign change of
    log(median|D| / target); it is then refined by bisection on log s.  If no sign change
    exists the grid point closest to the target is returned and matched=False.
    """
    def f(s):
        kw = dict(kw_unit)
        kw["force_scale"] = sign * s
        return np.log(own_D_median(kw) / target)

    grid = s0 * 2.0 ** (np.arange(-8, 9) / 2.0)
    vals = np.array([f(s) for s in grid])
    pairs = [i for i in range(len(grid) - 1) if vals[i] * vals[i + 1] <= 0]
    if not pairs:
        i = int(np.argmin(np.abs(vals)))
        return float(grid[i]), False
    i = min(pairs, key=lambda k: abs(np.log(np.sqrt(grid[k] * grid[k + 1]) / s0)))
    lo, hi = np.log(grid[i]), np.log(grid[i + 1])
    flo = vals[i]
    for _ in range(nbis):
        mid = 0.5 * (lo + hi)
        fm = f(np.exp(mid))
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return float(np.exp(0.5 * (lo + hi))), True


def evaluate(kw):
    """Same indicators as exp_mechanism.evaluate (identical seeds and lengths)."""
    out = {}
    lam = lyap_max(ensemble(20), Params(np.full(20, A_REF), **kw), n_tr=1000, n=5000)
    lam = np.where(np.isfinite(lam), lam, np.nan)
    out["lam_ref_median"] = float(np.nanmedian(lam))
    nic = 10
    a = np.repeat(A_GRID, nic)
    Q0 = np.tile(ensemble(nic, seed=7), (len(A_GRID), 1, 1))
    with np.errstate(all="ignore"):
        lg = lyap_max(Q0, Params(a, **kw), n_tr=500, n=3000)
    med = np.nanmedian(np.where(np.isfinite(lg), lg, np.nan).reshape(len(A_GRID), nic), axis=1)
    out["chaotic_ratio"] = float(np.mean(med > THR))
    out["lam_grid_median"] = med.tolist()
    sp, fo, co = spectrum(ensemble(5, seed=11), Params(np.full(5, A_REF), **kw),
                          n_tr=500, n=4000, return_decomp=True)
    out["spectrum"] = sp.mean(0).tolist()
    out["n_pos"] = float(np.mean((sp > THR).sum(1)))
    out["sum"] = float(sp.sum(1).mean())
    out["fold_term"] = float(fo.mean())
    out["coupling_term"] = float(co.mean())
    out["D_median"] = own_D_median(kw)
    return out
