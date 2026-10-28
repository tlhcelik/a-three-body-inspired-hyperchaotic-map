"""Figures for the new analyses (PDF + PNG, 300 dpi)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "legend.frameon": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
DIV = LinearSegmentedColormap.from_list("div", ["#1f5fae", "#9fbfe6", "#ececea", "#f2b08f", "#c4471b"])
THR = 0.057

import os, matplotlib.text
# English -> Turkish labels for the Turkish manuscript (figures_tr/); applied to every text object of a figure
TRMAP = [
    ("Newtonian $p=3$", "Newton $p=3$"), ("constant offset", "sabit öteleme"), ("linear ($p=0$)", "doğrusal ($p=0$)"),
    ("no coupling", "bağlaşımsız"), ("control parameter $a$", "kontrol parametresi $a$"),
    ("equal rms of $F$", "eşit rms $F$"), ("equal median $|F|$", "eşit medyan $|F|$"), ("equal median $|D|$", "eşit medyan $|D|$"),
    ("best strength", "en iyi güç"), ("distance exponent $p$", "uzaklık üssü $p$"),
    ("positive exponents at $a=0.97$", "$a=0.97$'de pozitif üstel"), ("chaotic ratio over $a\\in[-6,8]$", "$a\\in[-6,8]$ üzerinde kaotik oran"),
    ("chaotic ratio", "kaotik oran"), ("normalised index $i/(3N)$", "normalize indeks $i/(3N)$"),
    ("number of bodies $N$", "cisim sayısı $N$"), ("number of positive exponents", "pozitif üstel sayısı"),
    ("positive exponents", "pozitif üstel"), ("mass ratio $\\mu=m_3/m_1$", "kütle oranı $\\mu=m_3/m_1$"),
    ("softening $\\varepsilon_s$", "yumuşatma $\\varepsilon_s$"), ("boundary $R^2$", "sınır $R^2$"),
    ("perturbation size $h$", "sapma büyüklüğü $h$"), ("fraction of pairs on\ndifferent attractors", "farklı çekicilere giden\nçift oranı"),
    ("measured", "ölçülen"), ("chance", "rastgele düzey"),
    ("transient length (iterations)", "geçici rejim uzunluğu (iterasyon)"), ("count", "sayı"),
    ("retuned", "yeniden ayarlı"), ("default", "varsayılan"), ("$2N+1$ positive", "$2N+1$ pozitif"),
    ("zero-crossing index $i_0-2N$", "sıfır geçiş indeksi $i_0-2N$"),
    ("contribution to", "katkı:"), ("median $\\lambda_{\\max}$", "medyan $\\lambda_{\\max}$"),
    ("$\\lambda_i$ at $a^*(N)$", "$a^*(N)$'de $\\lambda_i$"),
]


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    os.makedirs("figures_tr", exist_ok=True)
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        for a, b in sorted(TRMAP, key=lambda ab: -len(ab[0])):
            if a in s:
                s = s.replace(a, b)
        t.set_text(s)
    fig.savefig(f"figures_tr/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures_tr/{name}.png", bbox_inches="tight")
    plt.close(fig)

# ---------------- mechanism: coupling controls under four ways of setting the coupling strength
MA = json.load(open("results/matching.json")); ST = json.load(open("results/strength.json"))


def ma(crit, k):
    return MA["newton"] if k == "newton" else (MA["none"] if k == "none" else MA[crit][k])


def st_best(k, key):
    recs = [r[key] for r in ST[k] if key in r]
    if len(recs) < len(ST[k]) or not recs:
        return None
    return dict(ratio=np.mean([r["chaotic_ratio"] for r in recs]), npos=np.mean([r["n_pos"] for r in recs]))


fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
a = np.round(np.arange(-6.0, 8.0001, 0.1), 10)
curves = [("newton", r"Newtonian $p=3$", C[0], 2.0), ("p=2", r"$p=2$", C[2], 1.2), ("p=4", r"$p=4$", C[3], 1.2),
          ("offset", "constant offset", C[1], 1.2), ("linear", r"linear ($p=0$)", C[6], 1.2), ("none", "no coupling", INK2, 1.2)]
def grid(r):
    return r["lam_grid_median"] if "lam_grid_median" in r else np.mean([x["lam_grid_median"] for x in r["reps"]], axis=0)


for k, lab, col, lw in curves:
    ax[0].plot(a, grid(ma("medD", k)), color=col, lw=lw, label=lab, zorder=3 if k == "newton" else 2)
ax[0].axhline(THR, color=INK2, lw=0.8, ls=":")
ax[0].set_xlabel(r"control parameter $a$"); ax[0].set_ylabel(r"median $\lambda_{\max}$"); ax[0].set_ylim(-3, 6)
ax[0].legend(fontsize=6.5, loc="lower center", ncol=2, frameon=True, framealpha=0.85)
ax[0].text(1.0, 1.02, "(a)", transform=ax[0].transAxes, ha="right")
ps = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
sch = [("rms", "equal rms of $F$", C[1], "o"), ("medF", r"equal median $|F|$", C[2], "s"),
       ("medD", r"equal median $|D|$", C[3], "^")]
for qi, (key, ylab) in enumerate([("chaotic_ratio", "chaotic ratio"), ("n_pos", "positive exponents at $a=0.97$")]):
    axx = ax[qi + 1]
    for crit, lab, col, mk in sch:
        axx.plot(ps, [ma(crit, "newton" if p == 3 else f"p={p:g}")[key] for p in ps], marker=mk, ms=3.5, lw=1.0, color=col, label=lab)
    hy = [st_best("newton" if p == 3 else f"p={p:g}", "best_hyper") for p in ps]
    axx.plot([p for p, h in zip(ps, hy) if h], [(h["ratio"] if key == "chaotic_ratio" else h["npos"]) for h in hy if h],
             marker="D", ms=3.5, lw=1.6, color=C[0], label="best strength")
    axx.axvline(3, color=INK2, lw=0.8, ls=":")
    axx.set_xlabel("distance exponent $p$"); axx.set_ylabel(ylab)
    axx.text(1.0, 1.02, "(b)" if qi == 0 else "(c)", transform=axx.transAxes, ha="right")
ax[1].set_ylim(0, 1.05)
ax[1].legend(fontsize=6.2, loc="lower center", frameon=True, framealpha=0.85)
fig.tight_layout(); save(fig, "fig_mechanism")

# ---------------- N bodies and masses
NB = json.load(open("results/nbody.json"))
Ns = [int(k) for k in NB["N"]]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))
for i, N in enumerate(Ns):
    sp = np.array(NB["N"][str(N)]["spec_star"])
    ax[0].plot(np.arange(1, len(sp) + 1) / len(sp), sp, marker="o", ms=3.2, lw=1.2, color=C[i], label=f"$N$={N}")
ax[0].axhline(0, color=INK2, lw=0.8)
ax[0].set_xlabel(r"normalised index $i/(3N)$"); ax[0].set_ylabel(r"$\lambda_i$ at $a^*(N)$")
ax[0].legend(fontsize=7, ncol=2, loc="lower left"); ax[0].text(1.0, 1.02, "(a)", transform=ax[0].transAxes, ha="right")
npos = [NB["N"][str(N)]["npos_star"] for N in Ns]
ks = [float(np.sum(np.clip(NB["N"][str(N)]["spec_star"], 0, None))) for N in Ns]
ax[1].plot(Ns, [2 * N + 1 for N in Ns], color=INK2, ls="--", lw=1, label="$2N+1$")
ax[1].plot(Ns, npos, "o", color=C[0], ms=6, label="positive exponents")
ax[1].set_xlabel("number of bodies $N$"); ax[1].set_ylabel("number of positive exponents")
ax[1].legend(fontsize=7, loc="upper left"); ax[1].text(1.0, 1.02, "(b)", transform=ax[1].transAxes, ha="right")
mus = sorted(NB["mu"], key=float)
ax[2].plot([float(m) for m in mus], [NB["mu"][m]["ratio"] for m in mus], "o-", color=C[0], label="chaotic ratio")
ax2 = ax[2]
ax2.set_xscale("log"); ax2.set_xlabel(r"mass ratio $\mu=m_3/m_1$"); ax2.set_ylabel("chaotic ratio")
ax2.set_ylim(0, 1.05)
for m in mus:
    ax2.annotate(f"{NB['mu'][m]['npos_ref']:.0f}", (float(m), NB["mu"][m]["ratio"]), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=7, color=INK2)
ax2.text(1.0, 1.02, "(c)", transform=ax2.transAxes, ha="right")
fig.tight_layout(); save(fig, "fig_nbody")

# ---------------- two-parameter planes
import sys
if not __import__("os").path.exists("results/twoparam.npz"): sys.exit(0)
T = np.load("results/twoparam.npz")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6)
for k, (Z, yv, yl, yref) in enumerate([(T["L_eps"], T["EPS"], r"softening $\varepsilon_s$", 0.1),
                                       (T["L_R2"], T["R2"], r"boundary $R^2$", 2.0)]):
    im = ax[k].pcolormesh(T["A"], yv, np.clip(Z, -3, 6), cmap=DIV, norm=norm, shading="nearest", rasterized=True)
    ax[k].set_yscale("log"); ax[k].set_ylabel(yl); ax[k].set_xlabel(r"$a$"); ax[k].grid(False)
    ax[k].plot([0.97], [yref], marker="*", ms=9, color="white", mec=INK, mew=0.8)
    ax[k].text(0.02, 0.95, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, va="top",
               bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02); cb.set_label(r"median $\lambda_{\max}$")
save(fig, "fig_twoparam")

# ---------------- basins / transient chaos
g = T["basin_grid"]; l12 = T["basin_m1"]
U = json.load(open("results/basin_uncert.json")); TR = json.load(open("results/transient.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.05]})
ax[0].semilogx(U["h"], U["frac"], "o-", color=C[0], ms=4, label="measured")
ax[0].axhline(1 - 1 / U["n_attr"], color=INK2, ls="--", lw=1, label=f"chance ($1-1/{U['n_attr']}$)")
ax[0].set_ylim(0.9, 1.0); ax[0].set_xlabel("perturbation size $h$"); ax[0].set_ylabel("fraction of pairs on\ndifferent attractors")
ax[0].legend(fontsize=7, loc="lower right"); tag = lambda a_, s_: a_.text(1.0, 1.02, s_, transform=a_.transAxes, ha="right")
tag(ax[0], "(a)")
Tt = np.array(TR["T"]); bins = np.geomspace(max(1, Tt.min()), Tt.max() + 1, 30)
ax[1].hist(np.maximum(Tt, 1), bins=bins, color=C[0], alpha=0.8, lw=0)
ax[1].set_xscale("log"); ax[1].set_xlabel("transient length (iterations)"); ax[1].set_ylabel("count"); tag(ax[1], "(b)")
cls = (l12[..., 1] > THR).astype(int)
ax[2].imshow(cls, origin="lower", extent=[-1, 1, -1, 1], cmap=ListedColormap([C[1], C[0]]), interpolation="nearest")
from matplotlib.patches import Patch
ax[2].legend(handles=[Patch(color=C[0], label=r"$\lambda_2>0$"), Patch(color=C[1], label=r"$\lambda_2\leq0$")],
             fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax[2].set_xlabel("$x_1(0)$"); ax[2].set_ylabel("$y_1(0)$"); ax[2].grid(False); tag(ax[2], "(c)")
fig.tight_layout(); save(fig, "fig_basins")
print("hyperchaotic fraction at a=-1:", cls.mean())

# ---------------- re-tuning of eps_s and R^2 for N = 3..6 (coarse grid)
RT = json.load(open("results/retune.json")); eps, r2 = RT["eps"], RT["R2"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f7fb", "#9fbfe6", "#2a78d6", "#0b3d7a"])
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.6), sharey=True)
for i, N in enumerate((3, 4, 5, 6)):
    Z = np.array([[RT["coarse"][f"{N}|{e}|{r}"]["ratio"] for e in eps] for r in r2])
    im = ax[i].pcolormesh(np.arange(len(eps) + 1) - 0.5, np.arange(len(r2) + 1) - 0.5, Z, vmin=0, vmax=1, cmap=BLUES)
    ax[i].set_xticks(range(len(eps))); ax[i].set_xticklabels([f"{e:g}" for e in eps], rotation=90, fontsize=6.5)
    ax[i].set_xlabel(r"$\varepsilon_s$"); ax[i].grid(False)
    ax[i].plot([eps.index(0.1)], [r2.index(2.0)], marker="*", ms=9, color="white", mec=INK, mew=0.8)
    j = np.unravel_index(np.argmax(Z), Z.shape)
    ax[i].plot([j[1]], [j[0]], marker="o", ms=6, mfc="none", mec=C[1], mew=1.5)
    ax[i].set_title(f"$N={N}$", fontsize=9)
ax[0].set_yticks(range(len(r2))); ax[0].set_yticklabels([f"{r:g}" if r < 1e3 else r"$\infty$" for r in r2], fontsize=6.5)
ax[0].set_ylabel(r"$R^2$")
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02); cb.set_label("chaotic ratio over $a\\in[-6,8]$")
save(fig, "fig_retune")

# ---------------- structure of the spectrum with N
SR = json.load(open("results/structure.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
for i, r in enumerate([r for r in SR["A"] if r["tag"] == "default"]):
    sp = np.array(r["spec"]); N = r["N"]
    ax[0].plot((np.arange(1, len(sp) + 1) - 0.5) / len(sp), sp, lw=1.1, marker="o", ms=2.4, color=C[i % len(C)], label=f"$N$={N}")
ax[0].axhline(0, color=INK2, lw=0.8); ax[0].set_xlabel(r"$(i-\frac{1}{2})/(3N)$"); ax[0].set_ylabel(r"$\lambda_i$ at $a^*(N)$")
ax[0].legend(fontsize=6, ncol=2, loc="upper right"); ax[0].text(1.0, 1.02, "(a)", transform=ax[0].transAxes, ha="right")
for r in SR["A"]:
    N = r["N"]; mk = "o" if r["tag"] == "default" else "s"
    ax[1].errorbar(N, r["cross"] - 2 * N, yerr=np.std(r["cross_each"]), fmt=mk, ms=4, color=C[0] if r["tag"] == "default" else C[1],
                   capsize=2, label=r["tag"])
h, l = ax[1].get_legend_handles_labels(); u = dict(zip(l, h)); ax[1].legend(u.values(), u.keys(), fontsize=6.5, loc="upper right")
ax[1].axhspan(1, 2, color=GRID, zorder=0); ax[1].text(2.05, 1.5, r"$2N+1$ positive", fontsize=6.5, color=INK2, va="center")
ax[1].set_xlabel("number of bodies $N$"); ax[1].set_ylabel(r"zero-crossing index $i_0-2N$"); ax[1].set_ylim(0, 3)
ax[1].text(1.0, 1.02, "(b)", transform=ax[1].transAxes, ha="right")
for i, N in enumerate((3, 4, 5, 6)):
    pts = sorted((r["a"], r["npos"]) for r in SR["B"] if r["N"] == N)
    ax[2].plot([p[0] for p in pts], [p[1] for p in pts], lw=1.0, marker=".", ms=3, color=C[i], label=f"$N$={N}")
    ax[2].axhline(2 * N + 1, color=C[i], lw=0.6, ls=":")
ax[2].set_xlabel("$a$"); ax[2].set_ylabel(r"positive exponents"); ax[2].legend(fontsize=6, ncol=2, loc="upper left")
ax[2].text(1.0, 1.02, "(c)", transform=ax[2].transAxes, ha="right")
fig.tight_layout(); save(fig, "fig_structure")

# ---------------- graphical abstract (used by the journal versions)
M = json.load(open("results/mechanism.json"))
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.3))
vals = [M["newton"]["fold_term"], M["newton"]["coupling_term"], M["newton"]["sum"]]
ax[0].bar([0, 1, 2], vals, color=[C[0], C[1], INK2], width=0.62)
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["fold", "coupling", "sum"])
ax[0].axhline(0, color=INK2, lw=0.8); ax[0].set_ylabel(r"contribution to $\sum_i\lambda_i$")
ax[0].set_title("expansion comes from the fold", fontsize=10)
ps_ga = [1, 1.5, 2, 2.5, 3, 3.5, 4]
rat = [st_best("newton" if p == 3 else f"p={p:g}", "best_hyper") for p in ps_ga]
ax[1].plot(ps_ga, [r["ratio"] for r in rat], marker="D", ms=4, lw=1.6, color=C[0])
ax[1].axvline(3, color=INK2, lw=0.8, ls=":")
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("distance exponent $p$"); ax[1].set_ylabel("chaotic ratio (hyperchaotic regime)")
ax[1].text(0.5, 0.10, "constant, linear, random couplings:\nno hyperchaos at any strength", transform=ax[1].transAxes,
           ha="center", fontsize=8, color=C[1])
ax[1].set_title("coupling must depend on distance", fontsize=10)
Ns_ga = [2, 3, 4, 5, 6]
ax[2].plot(Ns_ga, [2 * N + 1 for N in Ns_ga], color=INK2, ls="--", lw=1, label="$2N+1$")
ax[2].plot(Ns_ga, [NB["N"][str(N)]["npos_star"] for N in Ns_ga], "o", color=C[0], label="positive exponents")
ax[2].set_xlabel("number of bodies $N$"); ax[2].set_ylabel("positive Lyapunov exponents")
ax[2].legend(fontsize=9, loc="upper left"); ax[2].set_title("dimension grows with $N$", fontsize=10)
fig.tight_layout(); save(fig, "fig_graphical_abstract")
