"""Best-case comparison of coupling families: strength scan (reviewer question 7c, second part).

Magnitude matching (rms, median |F|, median |D|) depends on an arbitrary criterion.  Here each
coupling family is instead given its *own best strength*: the multiplier s of the coupling is
scanned on a logarithmic grid (factor 2^(1/2) per step, 2^-4 ... 2^8 times the median-|F|
matched value) and the strength with the largest chaotic ratio, and the one with the largest
chaotic ratio among those that are hyperchaotic at a = 0.97 (>= 6 positive exponents), are
re-evaluated with the full protocol of exp_mechanism.py.

Outputs results/strength.json.   Usage: python exp_strength.py [n_workers]
"""
import json, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rgmap.core import Params, ensemble, lyap_max, spectrum
from rgmap.controls import (A_GRID, A_REF, THR, attractor_sample, unit_kw, coupling_stat, evaluate)

A_COARSE = A_GRID[::2]
K = np.arange(-8, 17)                     # strength = s0 * 2**(k/2), i.e. 2^-4 ... 2^8 times s0
KINDS = ["newton", "repulsive", "p=1", "p=1.5", "p=2", "p=2.5", "p=3.5", "p=4", "p=5", "p=6",
         "linear", "random_linear", "offset"]
SEEDS = {"random_linear": [0, 1, 2], "offset": [0, 1, 2]}
_G = {}


def _init():
    S = attractor_sample()
    _G["S"] = S
    _G["ref"] = coupling_stat(S, unit_kw("newton"), "medF")


def coarse(kw):
    nic = 6
    a = np.repeat(A_COARSE, nic)
    Q0 = np.tile(ensemble(nic, seed=7), (len(A_COARSE), 1, 1))
    with np.errstate(all="ignore"):
        lg = lyap_max(Q0, Params(a, **kw), n_tr=300, n=2000)
    med = np.nanmedian(np.where(np.isfinite(lg), lg, np.nan).reshape(len(A_COARSE), nic), axis=1)
    sp = spectrum(ensemble(3, seed=11), Params(np.full(3, A_REF), **kw), n_tr=500, n=2500)
    return float(np.mean(med > THR)), float((sp > THR).sum(1).mean())


def task(args):
    kind, seed = args
    t0 = time.time()
    if kind == "repulsive":
        kw_unit, sign = unit_kw("newton"), -1.0
        s0 = 1.0
    else:
        kw_unit, sign = unit_kw(kind, seed), 1.0
        s0 = _G["ref"] / coupling_stat(_G["S"], kw_unit, "medF")
    sweep = []
    for k in K:
        s = s0 * 2.0 ** (k / 2.0)
        r, n = coarse(dict(kw_unit, force_scale=sign * s))
        sweep.append(dict(k=int(k), s=float(s), ratio=r, npos=n))
    i_r = int(np.argmax([d["ratio"] for d in sweep]))
    hyper = [i for i, d in enumerate(sweep) if d["npos"] >= 6]
    i_h = max(hyper, key=lambda i: sweep[i]["ratio"]) if hyper else None
    out = dict(sweep=sweep, s0=float(s0))
    out["best_ratio"] = dict(evaluate(dict(kw_unit, force_scale=sign * sweep[i_r]["s"])), s=sweep[i_r]["s"])
    if i_h is not None:
        out["best_hyper"] = dict(evaluate(dict(kw_unit, force_scale=sign * sweep[i_h]["s"])), s=sweep[i_h]["s"])
    return kind, seed, out, time.time() - t0


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(k, s) for k in KINDS for s in SEEDS.get(k, [0])]
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(nw, initializer=_init) as ex:
        for kind, seed, out, dt in ex.map(task, jobs):
            res.setdefault(kind, []).append(out)
            b = out["best_ratio"]; h = out.get("best_hyper")
            print(f"{kind:14s} seed {seed}: best ratio {b['chaotic_ratio']:.2f} (npos {b['n_pos']:.1f}, "
                  f"s/s0 {b['s']/out['s0']:.2f}); best strongly hyperchaotic "
                  + (f"ratio {h['chaotic_ratio']:.2f} (npos {h['n_pos']:.1f}, s/s0 {h['s']/out['s0']:.2f})" if h else "none")
                  + f"  [{dt:.0f}s, total {time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open("results/strength.json", "w"), indent=1)


if __name__ == "__main__":
    main()
