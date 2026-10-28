"""Robustness of the coupling controls (reviewer questions 7c and the extra p scan).

Repeats the controls of exp_mechanism.py with three different magnitude-matching criteria
(rms, median |F|, median |D| on the control's own attractor) and with a finer set of
distance exponents p.  The "rms" rows reproduce exp_mechanism.json.

Outputs results/matching.json.   Usage: python exp_matching.py [n_workers]
"""
import json, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rgmap.core import Params
from rgmap.controls import (CRITERIA, attractor_sample, unit_kw, coupling_stat, own_D_median,
                            match_medD, evaluate)

POWERS = ["p=1", "p=1.5", "p=2", "p=2.5", "p=3.5", "p=4", "p=5", "p=6"]
KINDS = ["repulsive", "linear"] + POWERS + ["random_linear", "offset"]
SEEDS = {"random_linear": [0, 1, 2], "offset": [0, 1, 2]}

_G = {}


def _init():
    S = attractor_sample()
    _G["S"] = S
    _G["ref"] = {h: coupling_stat(S, unit_kw("newton"), h) for h in ("rms", "medF")}
    _G["Dtarget"] = own_D_median(dict(p=3.0))


def task(args):
    crit, kind, seed = args
    S, ref = _G["S"], _G["ref"]
    t0 = time.time()
    if kind == "repulsive":
        kw_unit, sign = unit_kw("newton"), -1.0
    else:
        kw_unit, sign = unit_kw(kind, seed), 1.0
    if kind == "repulsive":
        s_stat = 1.0                                  # same magnitude by construction
    else:
        s_stat = ref["rms" if crit == "medD" else crit] / coupling_stat(S, kw_unit, "rms" if crit == "medD" else crit)
    matched = True
    if crit == "medD":
        s, matched = match_medD(kw_unit, _G["Dtarget"], s_stat, sign=sign)
    else:
        s = s_stat
    kw = dict(kw_unit)
    kw["force_scale"] = sign * s
    res = evaluate(kw)
    res.update(scale=float(s), matched=bool(matched),
               F_rms=coupling_stat(S, dict(kw_unit, force_scale=sign * s), "rms"),
               F_med=coupling_stat(S, dict(kw_unit, force_scale=sign * s), "medF"))
    return crit, kind, seed, res, time.time() - t0


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = []
    for crit in CRITERIA:
        for kind in KINDS:
            for seed in SEEDS.get(kind, [0]):
                jobs.append((crit, kind, seed))
    out = {c: {} for c in CRITERIA}
    reps = {c: {} for c in CRITERIA}
    t0 = time.time()
    with ProcessPoolExecutor(nw, initializer=_init) as ex:
        # newtonian and 'none' references (criterion independent) are evaluated in the parent
        for crit, kind, seed, res, dt in ex.map(task, jobs):
            reps[crit].setdefault(kind, []).append(res)
            r = res
            print(f"{crit:5s} {kind:14s} seed {seed} scale {r['scale']:.3g}{'' if r['matched'] else ' (UNMATCHED)'} "
                  f"lam {r['lam_ref_median']:.2f} ratio {r['chaotic_ratio']:.2f} npos {r['n_pos']:.1f} "
                  f"Dmed {r['D_median']:.2f}  [{dt:.0f}s, total {time.time()-t0:.0f}s]", flush=True)
    for crit in CRITERIA:
        for kind, rl in reps[crit].items():
            if len(rl) == 1:
                out[crit][kind] = rl[0]
            else:
                keys = ["lam_ref_median", "chaotic_ratio", "n_pos", "sum", "fold_term", "coupling_term",
                        "D_median", "scale", "F_rms", "F_med"]
                out[crit][kind] = {k: float(np.mean([r[k] for r in rl])) for k in keys}
                out[crit][kind]["reps"] = rl
    # criterion-independent rows
    _init()
    from rgmap.core import Params as _P
    out["newton"] = evaluate(dict(p=3.0))
    out["none"] = evaluate(dict(force_scale=0.0))
    out["Dtarget"] = _G["Dtarget"]
    out["ref_stats"] = _G["ref"]
    json.dump(out, open("results/matching.json", "w"), indent=1)


if __name__ == "__main__":
    main()
