"""Structure of the Lyapunov spectrum with the number of bodies (reviewer question 7a).

The number of positive exponents at the most chaotic parameter is 2N+1 for N = 2..5 (Table nbody).
This script asks how exact that regularity is and what produces it.

A. zero-crossing index   i0 = k + lambda_k / (lambda_k - lambda_{k+1})  (k = last positive exponent):
   the count is 2N+1 whenever 2N+1 <= i0 < 2N+2.  Spectra with 12 initial states for N = 2..8,
   with the default (eps, R^2) and, if results/retune.json exists, with the re-tuned ones;
   the count is also reported for three thresholds and per initial state.
B. maximum over a: number of positive exponents over a in [-6, 8] (step 0.5), N = 2..6.
C. factorisation J = diag(g) J_D: mean fold gain per coordinate <ln|g|>, the exponents mu_i of the
   coupling Jacobian alone along the same orbit, and the static estimate lambda_i ~ <ln|g|> + mu_i.
D. the same spectra plotted against (i - 1/2)/(3N) to quantify the collapse.

Outputs results/structure.json.   Usage: python exp_structure.py [n_workers]
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rgmap.core import Params, ensemble, iterate, step, jacobian, lyap_max, spectrum

THR = 0.057


def crossing(sp):
    """Fractional index at which the (descending) spectrum crosses zero, linear interpolation."""
    sp = np.asarray(sp)
    k = int((sp > 0).sum())
    if k == 0:
        return 0.0
    if k == len(sp):
        return float(len(sp))
    return k + sp[k - 1] / (sp[k - 1] - sp[k])


def find_astar(N, eps, R2, lo=-3.0, hi=2.0):
    ag = np.round(np.arange(lo, hi + 1e-9, 0.1), 10)
    nic = 6
    a = np.repeat(ag, nic)
    Q0 = np.tile(ensemble(nic, seed=2026, N=N), (len(ag), 1, 1))
    with np.errstate(all="ignore"):
        lg = lyap_max(Q0, Params(a, eps=eps, R2=R2, masses=(1.0,) * N), n_tr=500, n=3000)
    med = np.nanmedian(np.where(np.isfinite(lg), lg, np.nan).reshape(len(ag), nic), axis=1)
    return float(ag[int(np.nanargmax(med))])


def part_A(args):
    N, eps, R2, a, tag = args
    nic = 12
    sp = spectrum(ensemble(nic, seed=2027, N=N), Params(np.full(nic, a), eps=eps, R2=R2, masses=(1.0,) * N),
                  n_tr=1000, n=6000)
    m = sp.mean(0)
    return dict(N=N, eps=eps, R2=R2, a=a, tag=tag, spec=m.tolist(), spec_sd=sp.std(0).tolist(),
                cross=float(crossing(m)), cross_each=[float(crossing(s)) for s in sp],
                npos={str(t): int((m > t).sum()) for t in (0.0, THR, 0.1)},
                npos_each=(sp > THR).sum(1).tolist())


def part_B(args):
    N, a = args
    nic = 3
    sp = spectrum(ensemble(nic, seed=2027, N=N), Params(np.full(nic, a), masses=(1.0,) * N), n_tr=500, n=3000)
    return dict(N=N, a=a, npos=float((sp > THR).sum(1).mean()), npos_each=(sp > THR).sum(1).tolist())


def part_C(args):
    N, a = args
    nic, n_tr, n = 5, 1000, 4000
    P = Params(np.full(nic, a), masses=(1.0,) * N)
    Q = iterate(ensemble(nic, seed=2027, N=N), P, n_tr)
    d = 3 * N
    B1 = np.broadcast_to(np.eye(d), (nic, d, d)).copy(); B2 = B1.copy()
    s1 = np.zeros((nic, d)); s2 = np.zeros((nic, d)); lg = np.zeros(nic)
    for _ in range(n):
        J, g, JD = jacobian(Q, P)
        Q = step(Q, P)
        B1, R1 = np.linalg.qr(J @ B1)
        s1 += np.log(np.abs(np.diagonal(R1, axis1=1, axis2=2)))
        B2, R2_ = np.linalg.qr(JD @ B2)
        s2 += np.log(np.abs(np.diagonal(R2_, axis1=1, axis2=2)))
        lg += np.log(np.abs(g)).mean(1)
    lam = (-np.sort(-s1 / n, axis=1)).mean(0)
    mu = (-np.sort(-s2 / n, axis=1)).mean(0)
    gbar = float((lg / n).mean())
    est = gbar + mu
    return dict(N=N, a=a, lam=lam.tolist(), mu=mu.tolist(), gbar=gbar, npos=int((lam > THR).sum()),
                npos_static=int((est > THR).sum()), mean_abs_err=float(np.abs(est - lam).mean()))


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(nw) as ex:
        nb = json.load(open("results/nbody.json"))
        astar = {N: nb["N"][str(N)]["a_star"] for N in range(2, 7)}
        for N in (7, 8):
            astar[N] = find_astar(N, 0.1, 2.0)
        out["astar_default"] = {str(k): v for k, v in astar.items()}
        jobsA = [(N, 0.1, 2.0, astar[N], "default") for N in range(2, 9)]
        if os.path.exists("results/retune.json"):
            rt = json.load(open("results/retune.json"))["full"]
            for N in (3, 4, 5, 6):
                cands = [(v["ratio"], k, v) for k, v in rt.items() if k.split("|")[0] == str(N)]
                _, k, v = max(cands, key=lambda c: (c[0], c[2]["lam_max"]))
                _, e, r = k.split("|")
                jobsA.append((N, float(e), float(r), v["a_star"], "retuned"))
        out["A"] = list(ex.map(part_A, jobsA))
        print(f"A done {time.time()-t0:.0f} s", flush=True)
        AB = np.round(np.arange(-6.0, 8.0001, 0.5), 10)
        out["B"] = list(ex.map(part_B, [(N, float(a)) for N in range(2, 7) for a in AB]))
        print(f"B done {time.time()-t0:.0f} s", flush=True)
        out["C"] = list(ex.map(part_C, [(N, astar[N]) for N in range(2, 7)]))
        print(f"C done {time.time()-t0:.0f} s", flush=True)
    json.dump(out, open("results/structure.json", "w"), indent=1)
    for r in out["A"]:
        print(f"N={r['N']} {r['tag']:8s} eps={r['eps']} R2={r['R2']} a*={r['a']}: npos {r['npos']} cross {r['cross']:.2f} "
              f"(per IC {np.round(r['cross_each'], 2).tolist()})  2N+1={2*r['N']+1}")


if __name__ == "__main__":
    main()
