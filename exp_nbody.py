"""Generalisation: number of bodies N and unequal masses.  Outputs results/nbody.json"""
import json, numpy as np
from rgmap.core import Params, ensemble, lyap_max, spectrum

A_GRID = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
THR = 0.057
NIC = 10

def scan(masses, base_seed):
    N = len(masses)
    a = np.repeat(A_GRID, NIC)
    Q0 = np.tile(ensemble(NIC, seed=base_seed, N=N), (len(A_GRID), 1, 1))
    with np.errstate(all="ignore"):
        lg = lyap_max(Q0, Params(a, masses=masses), n_tr=500, n=3000)
    med = np.nanmedian(np.where(np.isfinite(lg), lg, np.nan).reshape(len(A_GRID), NIC), axis=1)
    return med

def spec_at(masses, a, base_seed):
    N = len(masses)
    sp = spectrum(ensemble(5, seed=base_seed + 1, N=N), Params(np.full(5, a), masses=masses),
                  n_tr=1000, n=5000)
    return sp

out = {"a_grid": A_GRID.tolist(), "N": {}, "mu": {}}
for N in [1, 2, 3, 4, 5, 6]:
    m = (1.0,) * N
    med = scan(m, 2026)
    i = int(np.nanargmax(med))
    sp_star = spec_at(m, A_GRID[i], 2026)
    sp_ref = spec_at(m, 0.97, 2026)
    out["N"][N] = dict(lam_grid=med.tolist(), ratio=float(np.mean(med > THR)),
                       a_star=float(A_GRID[i]), lam_star=float(med[i]),
                       spec_star=sp_star.mean(0).tolist(),
                       npos_star=float(np.mean((sp_star > THR).sum(1))),
                       spec_ref=sp_ref.mean(0).tolist(),
                       npos_ref=float(np.mean((sp_ref > THR).sum(1))),
                       npos_ref_each=(sp_ref > THR).sum(1).tolist())
    print("N", N, "ratio", round(out["N"][N]["ratio"], 3), "a*", A_GRID[i], "lam*", round(med[i], 3),
          "npos*", out["N"][N]["npos_star"], "npos(0.97)", out["N"][N]["npos_ref_each"], flush=True)

for mu in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    m = (1.0, 1.0, mu)
    med = scan(m, 2026)
    sp = spec_at(m, 0.97, 2026)
    lam_ref = lyap_max(ensemble(20), Params(np.full(20, 0.97), masses=m), n_tr=1000, n=5000)
    out["mu"][mu] = dict(lam_grid=med.tolist(), ratio=float(np.mean(med > THR)),
                         lam_ref=float(np.median(lam_ref)), lam_ref_sd=float(np.std(lam_ref)),
                         spec_ref=sp.mean(0).tolist(), npos_ref=float(np.mean((sp > THR).sum(1))))
    print("mu", mu, "ratio", round(out["mu"][mu]["ratio"], 3), "lam(0.97)", round(out["mu"][mu]["lam_ref"], 3),
          "npos", out["mu"][mu]["npos_ref"], flush=True)

json.dump(out, open("results/nbody.json", "w"), indent=1)
