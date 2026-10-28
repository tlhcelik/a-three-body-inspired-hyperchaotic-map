"""Independent reproduction of the paper's core analyses with English-labelled figures.
Usage: python repro.py [name ...]   (default: all)"""
import sys, json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import fsolve
from rgmap.core import Params, ensemble, iterate, step, lyap_max, terms, dD_dq, jacobian, REF_STATE
from rgmap import hp

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                     "legend.frameon": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
THR = 0.057
A_GRID = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
R = {}


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    plt.close(fig)


def tag(ax, s):
    ax.text(1.0, 1.02, s, transform=ax.transAxes, ha="right", va="bottom")


# ------------------------------------------------------------------ 1. scan + bifurcation
def scan_bif():
    nic = 20
    a = np.repeat(A_GRID, nic); Q0 = np.tile(ensemble(nic), (len(A_GRID), 1, 1))
    with np.errstate(all="ignore"):
        lam = lyap_max(Q0, Params(a), n_tr=500, n=3000).reshape(len(A_GRID), nic)
    med = np.median(lam, 1); q1, q3 = np.percentile(lam, [25, 75], axis=1)
    R["scan"] = dict(ratio=float(np.mean(med > THR)), nonchaotic=A_GRID[med <= THR].tolist())
    ab = np.linspace(-3, 6, 1500); Qb = iterate(np.repeat(REF_STATE[None], len(ab), 0), Params(ab), 500)
    xs = []
    for _ in range(200):
        Qb = step(Qb, Params(ab)); xs.append(Qb[:, 0, 0].copy())
    xs = np.array(xs)
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax[0].plot(np.repeat(ab[None], 200, 0).ravel(), xs.ravel(), ",", color=INK, alpha=0.25, rasterized=True)
    ax[0].set_ylabel("$x_1$"); tag(ax[0], "(a)")
    ax[1].fill_between(A_GRID, q1, q3, color=C[0], alpha=0.25, lw=0)
    ax[1].plot(A_GRID, med, color=C[0], lw=1.6)
    ax[1].axhline(THR, color=INK2, ls=":", lw=0.8)
    for v in (0.97, 5.2):
        for k in range(2): ax[k].axvline(v, color=C[1], ls="--", lw=0.8)
    ax[1].set_xlabel("control parameter $a$"); ax[1].set_ylabel(r"$\lambda_{\max}$"); tag(ax[1], "(b)")
    ax[1].set_xlim(-6, 8)
    fig.tight_layout(); save(fig, "en_fig_scan_bifurcation")


# ------------------------------------------------------------------ 2. phase portrait + density
def phase_density():
    P = Params([0.97]); Q = iterate(REF_STATE[None], P, 1000); pts = []
    for _ in range(20000):
        Q = step(Q, P); pts.append(Q[0, 0].copy())
    pts = np.array(pts)
    P2 = Params([5.2]); Q2 = iterate(REF_STATE[None], P2, 3000); cyc = []
    for _ in range(10):
        Q2 = step(Q2, P2); cyc.append(Q2[0, 0].copy())
    cyc = np.array(cyc)
    Pd = Params(np.full(5, 0.97)); Qd = iterate(ensemble(5, seed=3), Pd, 1000); xs = []
    for _ in range(200000 // 1):
        Qd = step(Qd, Pd); xs.append(Qd[:, 0, 0].copy())
        if len(xs) >= 200000: break
    xs = np.array(xs).ravel()
    h, e = np.histogram(xs, bins=256, range=(-1, 1), density=True)
    pk = h * (e[1] - e[0])
    R["density"] = dict(H=float(-(pk[pk > 0] * np.log2(pk[pk > 0])).sum()))
    fig = plt.figure(figsize=(7.8, 2.7))
    ax1 = fig.add_subplot(131, projection="3d"); ax2 = fig.add_subplot(132, projection="3d"); ax3 = fig.add_subplot(133)
    ax1.scatter(*pts.T, s=0.05, c=C[0], alpha=0.4, rasterized=True)
    ax2.scatter(*cyc.T, s=25, c=C[1])
    for ax, t in [(ax1, "(a) $a=0.97$"), (ax2, "(b) $a=5.2$")]:
        ax.set_xlabel("$x_1$", labelpad=-8); ax.set_ylabel("$y_1$", labelpad=-8); ax.set_zlabel("$z_1$", labelpad=-8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=6, pad=-3)
    c = 0.5 * (e[1:] + e[:-1])
    ax3.bar(c, h, width=e[1] - e[0], color=C[0], alpha=0.6, lw=0, label="measured")
    xx = np.linspace(-0.995, 0.995, 400)
    ax3.plot(xx, 1 / (np.pi * np.sqrt(1 - xx ** 2)), color=C[1], lw=1.4, label="arcsine")
    ax3.axhline(0.5, color=INK2, ls="--", lw=1, label="uniform")
    ax3.set_ylim(0, 3); ax3.set_xlabel("$x_1$"); ax3.set_ylabel("density"); ax3.legend(fontsize=7, loc="upper center")
    tag(ax3, "(c)")
    fig.subplots_adjust(wspace=0.45, left=0.02, right=0.98, bottom=0.18); save(fig, "en_fig_phase_density")


# ------------------------------------------------------------------ 3. complexity: 0-1, PE scan, D2
def series(a, n, tr=1000, seed_state=REF_STATE):
    P = Params([a]); Q = iterate(seed_state[None], P, tr); out = np.empty(n)
    for i in range(n):
        Q = step(Q, P); out[i] = Q[0, 0, 0]
    return out


def pe(x, m=7, dec=7):
    x = np.round(x, dec)
    w = np.lib.stride_tricks.sliding_window_view(x, m)
    pat = np.argsort(w, axis=1, kind="stable")
    code = (pat * (m ** np.arange(m))).sum(1)
    _, c = np.unique(code, return_counts=True); p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def complexity():
    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.5, 1.5]})
    for k, a in enumerate((0.97, 5.2)):
        x = series(a, 20000); j = np.arange(1, len(x) + 1)
        p = np.cumsum(x * np.cos(1.7 * j)); q = np.cumsum(x * np.sin(1.7 * j))
        ax[k].plot(p, q, lw=0.3, color=C[k]); ax[k].set_xlabel("$p_c$"); ax[k].set_ylabel("$q_c$")
        ax[k].tick_params(labelsize=6); tag(ax[k], "(a)" if k == 0 else "(b)")
        ax[k].set_title(f"$a={a}$", fontsize=8)
    # PE scan (vectorised over a)
    aa = np.linspace(-6, 8, 181); P = Params(aa)
    Q = iterate(np.repeat(REF_STATE[None], len(aa), 0), P, 10000); xs = np.empty((50000, len(aa)))
    for i in range(50000):
        Q = step(Q, P); xs[i] = Q[:, 0, 0]
    pes = np.array([pe(xs[:, i]) for i in range(len(aa))])
    R["pe_scan"] = dict(a=aa.tolist(), pe=pes.tolist())
    ax[2].plot(aa, pes, color=C[0]); ax[2].set_xlabel("$a$"); ax[2].set_ylabel(r"$H^{\rm norm}_{\rm PE}$ ($m=7$)")
    ax[2].set_ylim(0, 1.02); tag(ax[2], "(c)")
    # correlation dimension (local slopes)
    P = Params([0.97]); Q = iterate(REF_STATE[None], P, 1000); pts = []
    for _ in range(10000):
        Q = iterate(Q, P, 10); pts.append(Q[0].ravel().copy())
    pts = np.array(pts); r = np.geomspace(1e-2, 3, 40); res = {}
    for d, lab, col in [(3, "3-D", C[2]), (6, "6-D", C[3]), (9, "9-D", C[6])]:
        tr = cKDTree(pts[:, :d]); cnt = tr.count_neighbors(tr, r) - len(pts)
        Cr = cnt / (len(pts) * (len(pts) - 1))
        with np.errstate(all="ignore"):
            sl = np.gradient(np.log(Cr), np.log(r))
        ok = cnt >= 2000
        ax[3].semilogx(r[ok], sl[ok], color=col, label=lab)
        res[lab] = sl.tolist()
    R["d2"] = res
    ax[3].set_xlabel(r"$\varrho$"); ax[3].set_ylabel(r"$d\ln C/d\ln\varrho$"); ax[3].legend(fontsize=7); tag(ax[3], "(d)")
    fig.tight_layout(); save(fig, "en_fig_complexity")


# ------------------------------------------------------------------ 4. sensitivity (on attractor + 1-ulp key change)
def hp_floor(Q0, a, n):
    q = hp.from_np(Q0); Qd = Q0[None].copy(); out = []
    for _ in range(n):
        q = hp.step(q, a); Qd = step(Qd, Params([a]))
        out.append(np.linalg.norm(np.array(hp.to_float(q)) - Qd[0]))
    return np.array(out)


def sensitivity():
    rng = np.random.default_rng(11); n = 16
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for k, a in enumerate((0.97, 5.2)):
        P = Params(np.full(100, a)); Q = iterate(np.repeat(REF_STATE[None], 100, 0), P, 1000)
        v = rng.normal(size=Q.shape); v *= 1e-8 / np.linalg.norm(v.reshape(100, -1), axis=1)[:, None, None]
        X = Q + v; D = [np.full(100, 1e-8)]
        for _ in range(n):
            Q = step(Q, P); X = step(X, P); D.append(np.linalg.norm((X - Q).reshape(100, -1), axis=1))
        D = np.array(D)
        ax[0].semilogy(range(n + 1), np.median(D, 1), "o-", ms=3, color=C[k], label=f"$a={a}$")
        ax[0].fill_between(range(n + 1), *np.percentile(D, [10, 90], axis=1), color=C[k], alpha=0.2, lw=0)
    Q0 = iterate(REF_STATE[None], Params([0.97]), 1000)[0]
    fl = hp_floor(Q0, 0.97, n)
    ax[0].semilogy(range(1, n + 1), np.maximum(fl, 1e-17), "--", color=INK2, lw=1, label="double vs 60-digit")
    ax[0].set_xlabel("iteration $n$"); ax[0].set_ylabel("$D_n$"); ax[0].legend(fontsize=7); tag(ax[0], "(a)")
    R["horizon"] = int(np.argmax(fl > 1e-2)) + 1
    # 1-ulp key change from the initial ensemble (no transient)
    Q0 = ensemble(20); Q1 = Q0.copy(); Q1[:, 0, 0] = np.nextafter(Q1[:, 0, 0], np.inf)
    P = Params(np.full(20, 0.97)); D = [np.linalg.norm((Q1 - Q0).reshape(20, -1), axis=1)]; A, B = Q0, Q1
    for _ in range(20):
        A = step(A, P); B = step(B, P); D.append(np.linalg.norm((B - A).reshape(20, -1), axis=1))
    D = np.array(D); med = np.median(D, 1)
    R["key_step"] = int(np.argmax(med > 1e-2)); R["key_d0"] = float(med[0])
    ax[1].semilogy(range(21), np.maximum(med, 1e-18), "o-", ms=3, color=C[0], label="1-ulp change of $x_1(0)$")
    ax[1].fill_between(range(21), *np.maximum(np.percentile(D, [25, 75], axis=1), 1e-18), color=C[0], alpha=0.2, lw=0)
    ax[1].semilogy(range(1, 21), np.maximum(hp_floor(Q0[0], 0.97, 20), 1e-18), "--", color=INK2, lw=1,
                   label="double vs 60-digit")
    ax[1].axhline(1e-2, color=C[1], ls=":", lw=0.8)
    ax[1].set_xlabel("iteration $n$"); ax[1].set_ylabel("trajectory difference"); ax[1].legend(fontsize=7)
    tag(ax[1], "(b)")
    fig.tight_layout(); save(fig, "en_fig_sensitivity")


# ------------------------------------------------------------------ 5. ablation
def lam_curve(make_P, aa, nic=10, pure1d=False):
    a = np.repeat(aa, nic); Q0 = np.tile(ensemble(nic, seed=7), (len(aa), 1, 1))
    with np.errstate(all="ignore"):
        if pure1d:
            l = lyap_1d(a, Q0)
        else:
            l = lyap_max(Q0, make_P(a), n_tr=500, n=3000)
    l = np.where(np.isfinite(l), l, np.nan).reshape(len(aa), nic)
    return np.nanmedian(l, 1)


def lyap_1d(a, Q0):
    q = Q0.reshape(len(Q0), -1)
    f = lambda q: np.sin(1.0 / (a[:, None] * q))
    for _ in range(500): q = f(q)
    s = np.zeros(q.shape)
    for _ in range(3000):
        d = np.abs(np.cos(1 / (a[:, None] * q)) / (a[:, None] * q ** 2)); q = f(q); s += np.log(d + 1e-300)
    return (s / 3000).max(1)


def ablation():
    aa = np.round(np.arange(-3.0, 5.5001, 0.05), 10)
    v = [("full model", lambda a: Params(a), C[0]),
         (r"$\Phi_i=0$", lambda a: Params(a, use_phi=False), C[1]),
         (r"$F_{i,k}=0$", lambda a: Params(a, force_scale=0.0), C[2]),
         (r"$\Phi_i=F_{i,k}=0$", lambda a: Params(a, use_phi=False, force_scale=0.0), C[6])]
    f = [("sine", lambda a: Params(a), C[0]), ("triangle", lambda a: Params(a, fold="tri"), C[1]),
         ("sawtooth", lambda a: Params(a, fold="saw"), C[2]), ("tanh (monotone)", lambda a: Params(a, fold="tanh"), C[6])]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for lab, mk, col in v:
        ax[0].plot(aa, lam_curve(mk, aa), color=col, label=lab)
    m = np.abs(aa) > 0.05
    ax[0].plot(aa[m], lam_curve(None, aa[m], pure1d=True), color=INK2, lw=1, label=r"pure 1-D $\sin(G/aq)$")
    for lab, mk, col in f:
        ax[1].plot(aa, lam_curve(mk, aa), color=col, label=lab)
    for k in range(2):
        ax[k].axhline(THR, color=INK2, ls=":", lw=0.8); ax[k].set_xlabel("$a$")
        ax[k].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax[k].set_ylim(-3, 6.5)
    ax[0].set_ylabel(r"median $\lambda_{\max}$"); tag(ax[0], "(a)"); tag(ax[1], "(b)")
    fig.tight_layout(); save(fig, "en_fig_ablation")


# ------------------------------------------------------------------ 6. fixed points
def fixed_points():
    a = 0.97; P = Params([a])
    guesses = [np.full((3, 3), 0.001015),
               np.array([[0.605, 0.886, 0.389], [-0.083, -0.735, 0.410], [-0.525, 0.418, -0.722]]),
               np.array([[0.892, -0.878, -0.044], [-0.573, 0.840, -0.307], [-0.520, -0.139, 0.712]])]
    sols = []
    for g in guesses:
        fun = lambda x: (step(x.reshape(1, 3, 3), P) - x.reshape(1, 3, 3)).ravel()
        x = fsolve(fun, g.ravel(), xtol=1e-14); sols.append(x.reshape(3, 3))
    res = [float(np.abs(step(s[None], P)[0] - s).max()) for s in sols]
    eig = [np.abs(np.linalg.eigvals(jacobian(s[None], P)[0][0])) for s in sols]
    R["fixed_points"] = dict(points=[s.tolist() for s in sols], residual=res,
                             min_abs_eig=[float(e.min()) for e in eig], max_abs_eig=[float(e.max()) for e in eig])
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    t = np.linspace(0, 2 * np.pi, 200)
    for k, s in enumerate(sols):
        ax[k].plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), "--", color=INK2, lw=0.8)
        for i, (mk, col) in enumerate(zip("os^", C[:3])):
            ax[k].plot(s[i, 0], s[i, 1], mk, ms=8 - 2 * (k == 0) * i, color=col, mec=INK, mew=0.5,
                       label=f"body {i+1}")
        if k: [ax[k].plot(*s[[i, j], :2].T, ":", color=INK2, lw=0.7) for i in range(3) for j in range(i)]
        ax[k].set_aspect("equal"); ax[k].set_xlim(-1.6, 1.6); ax[k].set_ylim(-1.6, 1.6); ax[k].set_xlabel("$x$")
        tag(ax[k], "(" + "abc"[k] + ")")
    ax[0].set_ylabel("$y$"); ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "en_fig_fixed_points")


# ------------------------------------------------------------------ 7. reliability
def reliability():
    P = Params(np.full(300, 0.97)); Q = iterate(np.repeat(REF_STATE[None], 300, 0) +
                                                 np.random.default_rng(4).uniform(-.15, .15, (300, 3, 3)), P, 1000)
    Q = iterate(Q, P, 7)
    J, _, _ = jacobian(Q, P); hs = np.logspace(-9, -2, 8); med, p90 = [], []
    for h in hs:
        Jh = np.zeros_like(J)
        for c in range(9):
            E = np.zeros(9); E[c] = h; E = E.reshape(3, 3)
            Jh[:, :, c] = ((step(Q + E, P) - step(Q - E, P)).reshape(300, -1)) / (2 * h)
        err = np.linalg.norm(Jh - J, axis=(1, 2)) / np.linalg.norm(J, axis=(1, 2))
        med.append(np.median(err)); p90.append(np.percentile(err, 90))
    rho = np.abs(np.linalg.eigvals(J)).max(1)
    R["jac"] = dict(h=hs.tolist(), med=med, p90=p90, rho_med=float(np.median(rho)), rho_max=float(rho.max()))
    d0s = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]; m, s = [], []
    for d0 in d0s:
        l = lyap_max(ensemble(20), Params(np.full(20, 0.97)), n_tr=1000, n=10000, d0=d0); m.append(l.mean()); s.append(l.std())
    R["delta0"] = dict(d0=d0s, mean=m, sd=s)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].loglog(hs, med, "o-", color=C[0], label="median"); ax[0].loglog(hs, p90, "^--", color=C[1], label="90th percentile")
    ax[0].set_xlabel("finite-difference step $h$"); ax[0].set_ylabel(r"$\|J_h-J\|_F/\|J\|_F$"); ax[0].legend(fontsize=7)
    tag(ax[0], "(a)")
    ax[1].errorbar(d0s, m, yerr=s, fmt="s-", color=C[0], capsize=3); ax[1].set_xscale("log")
    ax[1].set_xlabel(r"initial separation $\delta_0$"); ax[1].set_ylabel(r"$\lambda_{\max}$ (mean $\pm$ sd)"); tag(ax[1], "(b)")
    fig.tight_layout(); save(fig, "en_fig_reliability")


ALL = dict(scan=scan_bif, phase=phase_density, complexity=complexity, sensitivity=sensitivity,
           ablation=ablation, fixed=fixed_points, reliability=reliability)
if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    try:
        R.update(json.load(open("results/repro.json")))
    except Exception:
        pass
    for n in names:
        ALL[n](); print("done", n, flush=True)
        json.dump(R, open("results/repro.json", "w"), indent=1)
