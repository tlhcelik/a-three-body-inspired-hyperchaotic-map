"""Re-tuning of the softening length eps_s and the boundary parameter R^2 for N = 4, 5, 6 bodies
(reviewer question 7b).  N = 3 is included as a control: if the reference setting is a good
choice for three bodies, re-tuning should not improve it much.

Stage 1: coarse (eps, R^2) grid, chaotic ratio over a in [-6, 8] (step 0.2, 6 initial states).
Stage 2: for each N the best grid cells, the paper's setting (0.1, 2) and a "scaled" setting
         eps = 0.1 (N-1)/2 are re-evaluated with the full protocol of exp_nbody.py (step 0.1,
         10 initial states), on a wider range of a (to see where the chaotic window ends) and
         with the Lyapunov spectrum at the most chaotic a.

Outputs results/retune.json.   Usage: python exp_retune.py [n_workers]
"""
import json, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rgmap.core import Params, ensemble, lyap_max, spectrum

THR = 0.057
A_FULL = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
A_COARSE = np.round(np.arange(-6.0, 8.0001, 0.2), 10)
A_WIDE = np.round(np.arange(-14.0, 14.0001, 0.2), 10)
EPS = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
R2S = [1.0, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0, 64.0, 10000.0]
NS = [3, 4, 5, 6]


def scan(N, eps, R2, agrid, nic, n_tr, n, seed=2026):
    a = np.repeat(agrid, nic)
    Q0 = np.tile(ensemble(nic, seed=seed, N=N), (len(agrid), 1, 1))
    with np.errstate(all="ignore"):
        lg = lyap_max(Q0, Params(a, eps=eps, R2=R2, masses=(1.0,) * N), n_tr=n_tr, n=n)
    lg = np.where(np.isfinite(lg), lg, np.nan).reshape(len(agrid), nic)
    return np.nanmedian(lg, axis=1)


def runs(a, chaotic, min_width=0.3):
    out, i = [], 0
    while i < len(a):
        if chaotic[i]:
            j = i
            while j + 1 < len(a) and chaotic[j + 1]:
                j += 1
            if a[j] - a[i] >= min_width:
                out.append([float(a[i]), float(a[j])])
            i = j + 1
        else:
            i += 1
    return out


def coarse(args):
    N, eps, R2 = args
    med = scan(N, eps, R2, A_COARSE, 6, 300, 2000)
    i = int(np.nanargmax(med))
    return args, dict(ratio=float(np.mean(med > THR)), lam_max=float(med[i]), a_star=float(A_COARSE[i]))


def full(args):
    N, eps, R2 = args
    t0 = time.time()
    med = scan(N, eps, R2, A_FULL, 10, 500, 3000)
    i = int(np.nanargmax(med))
    wide = scan(N, eps, R2, A_WIDE, 6, 300, 2000)
    a_star = float(A_FULL[i])
    sp = spectrum(ensemble(5, seed=2027, N=N), Params(np.full(5, a_star), eps=eps, R2=R2, masses=(1.0,) * N),
                  n_tr=1000, n=5000).mean(0)
    sp_ref = spectrum(ensemble(5, seed=2027, N=N), Params(np.full(5, 0.97), eps=eps, R2=R2, masses=(1.0,) * N),
                      n_tr=1000, n=5000)
    return args, dict(ratio=float(np.mean(med > THR)), lam_max=float(med[i]), a_star=a_star,
                      lam_grid=med.tolist(), runs=runs(A_FULL, med > THR),
                      ratio_wide=float(np.mean(wide > THR)), runs_wide=runs(A_WIDE, wide > THR),
                      spec_star=sp.tolist(), npos_star=int((sp > THR).sum()),
                      npos_ref_each=(sp_ref > THR).sum(1).tolist(), seconds=time.time() - t0)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    res = {"eps": EPS, "R2": R2S, "coarse": {}, "full": {}}
    with ProcessPoolExecutor(nw) as ex:
        jobs = [(N, e, r) for N in NS for e in EPS for r in R2S]
        for (N, e, r), d in ex.map(coarse, jobs):
            res["coarse"][f"{N}|{e}|{r}"] = d
        print(f"stage 1 done in {time.time() - t0:.0f} s", flush=True)
        stage2 = []
        for N in NS:
            cells = sorted(((res["coarse"][f"{N}|{e}|{r}"]["ratio"], res["coarse"][f"{N}|{e}|{r}"]["lam_max"], e, r)
                            for e in EPS for r in R2S), reverse=True)[:3]
            cand = {(e, r) for _, _, e, r in cells} | {(0.1, 2.0), (round(0.1 * (N - 1) / 2, 4), 2.0)}
            stage2 += [(N, e, r) for e, r in sorted(cand)]
        for (N, e, r), d in ex.map(full, stage2):
            res["full"][f"{N}|{e}|{r}"] = d
            print(f"N={N} eps={e} R2={r}: ratio {d['ratio']:.2f} (wide {d['ratio_wide']:.2f}) a*={d['a_star']} "
                  f"lam*={d['lam_max']:.2f} npos*={d['npos_star']}  runs {d['runs']}", flush=True)
    json.dump(res, open("results/retune.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
