"""Mechanism analysis: where does the expansion come from, and does the
gravitational (1/r^2) structure of the coupling matter?

Outputs results/mechanism.json
"""
import json, numpy as np
from rgmap.core import Params, ensemble, iterate, terms, lyap_max, spectrum

A_REF = 0.97
A_GRID = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
THR = 0.057
rng = np.random.default_rng(12345)

# --- attractor sample of the reference map, used to match coupling magnitudes
Pref = Params(np.full(20, A_REF))
Q = iterate(ensemble(20), Pref, 1000)
states = []
for _ in range(200):
    Q = iterate(Q, Pref, 1)
    states.append(Q.copy())
S = np.concatenate(states)                       # (4000,3,3)

def rms_coupling(P):
    Ps = Params(np.full(len(S), A_REF), eps=P.eps[0], p=P.p, W=P.W, offset=P.offset,
                force_scale=1.0)
    return np.sqrt((terms(S, Ps)[1] ** 2).mean())

F_RMS = rms_coupling(Params([A_REF]))
Dref = terms(S, Params(np.full(len(S), A_REF)))[3]

def make(kind, seed=0):
    r = np.random.default_rng(seed)
    if kind == "newton":    return dict(p=3.0)
    if kind == "repulsive": return dict(p=3.0, force_scale=-1.0)
    if kind == "none":      return dict(force_scale=0.0)
    if kind.startswith("p="):
        p = float(kind[2:]); s = F_RMS / rms_coupling(Params([A_REF], p=p))
        return dict(p=p, force_scale=s)
    if kind == "linear":
        s = F_RMS / rms_coupling(Params([A_REF], p=0.0)); return dict(p=0.0, force_scale=s)
    if kind == "random_linear":
        W = r.normal(size=(9, 9)); s = F_RMS / rms_coupling(Params([A_REF], W=W))
        return dict(W=W, force_scale=s)
    if kind == "offset":
        off = r.normal(size=(3, 3)); s = F_RMS / rms_coupling(Params([A_REF], offset=off))
        return dict(offset=off, force_scale=s)
    raise ValueError(kind)

def evaluate(kw):
    out = {}
    lam = lyap_max(ensemble(20), Params(np.full(20, A_REF), **kw), n_tr=1000, n=5000)
    lam = np.where(np.isfinite(lam), lam, np.nan)
    out["lam_ref_median"] = float(np.nanmedian(lam))
    out["lam_ref_q"] = [float(np.nanpercentile(lam, 25)), float(np.nanpercentile(lam, 75))]
    # chaotic ratio over the a-grid (10 ICs per a)
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
    out["fold_term"] = float(fo.mean()); out["coupling_term"] = float(co.mean())
    # typical |D| on the variant's own attractor
    Pv = Params(np.full(20, A_REF), **kw)
    Qv = iterate(ensemble(20), Pv, 1000)
    Ds = []
    for _ in range(100):
        Ds.append(np.abs(terms(Qv, Pv)[3]).ravel()); Qv = iterate(Qv, Pv, 1)
    Ds = np.concatenate(Ds)
    out["D_median"] = float(np.median(Ds)); out["D_p10"] = float(np.percentile(Ds, 10))
    return out

results = {"F_RMS": float(F_RMS), "a_grid": A_GRID.tolist(),
           "Dref_median": float(np.median(np.abs(Dref))),
           "Dref_p10": float(np.percentile(np.abs(Dref), 10))}
kinds = ["newton", "none", "repulsive", "p=1", "p=2", "p=5", "linear"]
for k in kinds:
    results[k] = evaluate(make(k)); print(k, {x: results[k][x] for x in ["lam_ref_median", "chaotic_ratio", "n_pos", "sum", "fold_term", "coupling_term", "D_median"]}, flush=True)
for k in ["random_linear", "offset"]:
    reps = [evaluate(make(k, seed=s)) for s in range(3)]
    results[k] = {"reps": reps,
                  "lam_ref_median": float(np.mean([r["lam_ref_median"] for r in reps])),
                  "chaotic_ratio": float(np.mean([r["chaotic_ratio"] for r in reps])),
                  "n_pos": float(np.mean([r["n_pos"] for r in reps])),
                  "sum": float(np.mean([r["sum"] for r in reps])),
                  "fold_term": float(np.mean([r["fold_term"] for r in reps])),
                  "coupling_term": float(np.mean([r["coupling_term"] for r in reps])),
                  "D_median": float(np.mean([r["D_median"] for r in reps]))}
    print(k, [(round(r["lam_ref_median"], 3), round(r["chaotic_ratio"], 3), r["n_pos"]) for r in reps], flush=True)
json.dump(results, open("results/mechanism.json", "w"), indent=1)
