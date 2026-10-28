"""Write LaTeX macros and English tables for the new analyses into ../tables_en/ (next to makale_en.tex)."""
import json, numpy as np, os
OUT = "../tables_en"; os.makedirs(OUT, exist_ok=True)
M = json.load(open("results/mechanism.json")); NB = json.load(open("results/nbody.json"))
CP = json.load(open("results/compare.json")); TR = json.load(open("results/transient.json"))
BU = json.load(open("results/basin_uncert.json")); RP = json.load(open("results/repro.json"))
TP = np.load("results/twoparam.npz")
MA = json.load(open("results/matching.json")); ST = json.load(open("results/strength.json"))
RT = json.load(open("results/retune.json")); SR = json.load(open("results/structure.json"))
M0 = json.load(open("results_ilk_teslim/mechanism.json"))     # first delivery (other platform)
THR = 0.057


def f(x, d=2): return f"{x:.{d}f}"


def g1(x):
    """One decimal, without the sign of a negative zero."""
    return f"{0.0 if abs(x) < 0.05 else x:.1f}"


def ma(crit, k):
    return MA["newton"] if k == "newton" else (MA["none"] if k == "none" else MA[crit][k])


def st_mean(k, key, field):
    """Mean over the seeds of a family of field `field` of the best-strength record `key`; None if absent."""
    recs = [(r, r[key]) for r in ST[k] if key in r]
    if len(recs) < len(ST[k]) or not recs:
        return None
    return float(np.mean([(b["s"] / r["s0"]) if field == "s_rel" else b[field] for r, b in recs]))


def st_maxnpos(k):
    """Largest number of positive exponents at a = 0.97 found anywhere in the strength sweep (mean over seeds)."""
    return float(np.mean([max(d["npos"] for d in r["sweep"]) for r in ST[k]]))


def st_maxratio(k):
    return float(np.mean([r["best_ratio"]["chaotic_ratio"] for r in ST[k]]))


# ---------------------------------------------------------------- Turkish copies of the tables (tables_tr/)
OUT_TR = "../tables_tr"; os.makedirs(OUT_TR, exist_ok=True)
TRPAIRS = [
    (r"Newtonian, $p=3$ (proposed)", r"Newton, $p=3$ (\"onerilen)"), (r"Newtonian, repulsive", r"Newton, itici"),
    (r"power law, $p=", r"kuvvet yasas{\i}, $p="), (r"linear diffusive, $p=0$", r"do\u{g}rusal yay{\i}n{\i}ml{\i}, $p=0$"),
    (r"random linear", r"rastgele do\u{g}rusal"), (r"constant offset (no coupling)", r"sabit \"oteleme (ba\u{g}la\c{s}{\i}ms{\i}z)"),
    (r"constant offset", r"sabit \"oteleme"), (r"none, $F_{i,k}=0$", r"yok, $F_{i,k}=0$"),
    (r"\shortstack{equal rms\\of $F$}", r"\shortstack{e\c{s}it rms\\($F$)}"),
    (r"\shortstack{equal median\\of $|F|$}", r"\shortstack{e\c{s}it medyan\\($|F|$)}"),
    (r"\shortstack{equal median\\of $|D|$}", r"\shortstack{e\c{s}it medyan\\($|D|$)}"),
    (r"\shortstack{best hyperchaotic\\strength ($s/s_0$)}", r"\shortstack{en iyi hiperkaotik\\g\"u\c{c} ($s/s_0$)}"),
    (r"\shortstack{largest ratio\\at any strength}", r"\shortstack{en b\"uy\"uk oran\\(herhangi g\"u\c{c})}"),
    (r"median $|D|$", r"medyan $|D|$"), (r"Folding", r"Katlama"), (r"& Coupling \\", r"& Ba\u{g}la\c{s}{\i}m terimi \\"),
    (r"Coupling &", r"Kuplaj &"),
    (r"& none &", r"& yok &"), (r"& none \\", r"& yok \\"),
    (r"Bodies &", r"Cisimler &"), (r"Dimension", r"Boyut"),
    (r"\shortstack{Chaotic\\ratio}", r"\shortstack{Kaotik\\oran}"),
    (r"\shortstack{Positive\\exponents at $a^{*}$}", r"\shortstack{Pozitif \"ustel\\($a^{*}$'da)}"),
    (r"\shortstack{$\sum\lambda_i^{+}$\\at $a^{*}$}", r"\shortstack{$\sum\lambda_i^{+}$\\($a^{*}$'da)}"),
    (r"Masses $(1,1,\mu)$", r"K\"utleler $(1,1,\mu)$"),
    (r"\shortstack{Positive\\exponents at $0.97$}", r"\shortstack{Pozitif \"ustel\\($0.97$'de)}"),
    (r"\shortstack{ratio over\\$[-14,14]$}", r"\shortstack{$[-14,14]$'te\\oran}"),
    (r"\shortstack{Positive\\exponents}", r"\shortstack{Pozitif\\\"ustel}"),
    (r"Setting", r"Ayar"), (r"reference", r"referans"), (r"best ratio", r"en iyi oran"),
    (r"default", r"varsay{\i}lan"), (r"retuned", r"yeniden ayarl{\i}"),
    (r"\shortstack{Zero-crossing\\index $i_0$}", r"\shortstack{S{\i}f{\i}r ge\c{c}i\c{s}\\indeksi $i_0$}"),
    (r"Map &", r"Harita &"), (r"Dim.", r"Boy."), (r"\shortstack{Parameter\\range}", r"\shortstack{Parametre\\aral{\i}\u{g}{\i}}"),
    (r"(nats/iter.)", r"(nat/iter.)"), (r"Logistic ", r"Lojistik "), (r"Sine ", r"Sin\"us "), (r"\textbf{Proposed}", r"\textbf{\"Onerilen}"),
]


def emit(name, text):
    open(f"{OUT}/{name}.tex", "w").write(text)
    tr = text
    for a, b in TRPAIRS:
        tr = tr.replace(a, b)
    open(f"{OUT_TR}/{name}.tex", "w").write(tr)


mac = {
 # decomposition of the exponent sum (first-delivery run is kept for the rms sensitivity statement only)
 "vMechFold": f(M["newton"]["fold_term"]), "vMechCoup": f(M["newton"]["coupling_term"]),
 "vMechSum": f(M["newton"]["sum"]), "vDmed": f(M["Dref_median"]), "vDpTen": f(M["Dref_p10"], 3),
 "vMechNoneFold": f(MA["none"]["fold_term"], 1), "vMechNoneD": f(MA["none"]["D_median"], 1),
 # sensitivity of the rms criterion
 "vFrmsFirst": f(M0["F_RMS"]), "vFrmsRerun": f(M["F_RMS"]),
 "vFrmsChange": f"{100 * abs(M['F_RMS'] / M0['F_RMS'] - 1):.0f}",
 "vPoneLamFirst": f(M0["p=1"]["lam_ref_median"]), "vPoneLamRerun": f(M["p=1"]["lam_ref_median"]),
 "vMedF": f(MA["ref_stats"]["medF"], 2), "vDtarget": f(MA["Dtarget"], 2),
 # median criteria and best strength
 "vRepRatioMedD": f(ma("medD", "repulsive")["chaotic_ratio"]), "vRepPosMedD": f(ma("medD", "repulsive")["n_pos"], 1),
 "vNewtonRatioBest": f(st_mean("newton", "best_hyper", "chaotic_ratio")),
 "vNewtonSBest": f(st_mean("newton", "best_hyper", "s_rel")),
 "vNewtonNposBest": f(st_mean("newton", "best_hyper", "n_pos"), 0),
 "vRepRatioBest": f(st_mean("repulsive", "best_hyper", "chaotic_ratio")),
 "vOffsetMaxNpos": f(st_maxnpos("offset"), 1), "vRandMaxNpos": f(st_maxnpos("random_linear"), 1),
 "vLinMaxNpos": f(st_maxnpos("linear"), 1),
 "vOffsetMaxRatio": f(st_maxratio("offset")), "vRandMaxRatio": f(st_maxratio("random_linear")),
 "vLinMaxRatio": f(st_maxratio("linear")),
 "vTransMed": f"{TR['T_median']:.0f}", "vTransLo": f"{TR['T_q'][0]:.0f}", "vTransHi": f"{TR['T_q'][3]:.0f}",
 "vTransMax": f"{TR['T_max']}", "vTransLam": f(TR["lam_transient"], 1),
 "vBasinFracMin": f(min(BU["frac"]), 3), "vBasinChance": f(1 - 1 / BU["n_attr"], 3), "vBasinM": f"{BU['M']}",
 "vHyperFrac": f(float(np.mean(TP["basin_m1"][..., 1] > THR)), 2),
 "vMuHiRatio": f(NB["mu"]["5.0"]["ratio"]), "vMuMidRatioLo": f(min(NB["mu"][m]["ratio"] for m in ["0.01", "0.1", "0.5", "1.0", "2.0"])),
 "vNsixPos": f(NB["N"]["6"]["npos_star"], 1),
 "vRhoMedRep": f"{RP['jac']['rho_med']:.0f}",
 "vKeyStepRep": f"{RP['key_step']}",
}
PW = {"1": "One", "1.5": "OneFive", "2": "Two", "2.5": "TwoFive", "3.5": "ThreeFive", "4": "Four", "5": "Five", "6": "Six"}
NW = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}
mac["vSbestThreeFive"] = f(st_mean("p=3.5", "best_hyper", "s_rel")); mac["vSbestFour"] = f(st_mean("p=4", "best_hyper", "s_rel"))
mac["vNewtonRatio"] = f(MA["newton"]["chaotic_ratio"]); mac["vNewtonNpos"] = f(MA["newton"]["n_pos"], 0)
mac["vNewtonLam"] = f(MA["newton"]["lam_ref_median"])
for p in ["5", "6"]:
    mac[f"vMaxNposP{PW[p]}"] = f(st_maxnpos(f"p={p}"), 1)
for p in ["1", "1.5", "2", "2.5", "3.5", "4"]:
    tag = PW[p]
    mac[f"vP{tag}Ratio"] = f(st_mean(f"p={p}", "best_hyper", "chaotic_ratio")) if st_mean(f"p={p}", "best_hyper", "chaotic_ratio") is not None else "--"
    mac[f"vP{tag}Pos"] = f(st_mean(f"p={p}", "best_hyper", "n_pos"), 0) if st_mean(f"p={p}", "best_hyper", "n_pos") is not None else "--"
    mac[f"vP{tag}RatioMedD"] = f(ma("medD", f"p={p}")["chaotic_ratio"])
    mac[f"vP{tag}PosMedD"] = f(ma("medD", f"p={p}")["n_pos"], 1)

# ---------------------------------------------------------------- nbody + compare + retune + structure macros
rt_full = RT["full"]


def best_cfg(N):
    cands = [(v["ratio"], v["lam_max"], k) for k, v in rt_full.items() if k.split("|")[0] == str(N) and k != f"{N}|0.1|2.0"]
    return max(cands)[2]


for N in (3, 4, 5, 6):
    d = rt_full[f"{N}|0.1|2.0"]; k = best_cfg(N); b = rt_full[k]; _, e, r = k.split("|")
    n = NW[N]
    mac[f"vRt{n}RatioDef"] = f(d["ratio"]); mac[f"vRt{n}RatioBest"] = f(b["ratio"])
    mac[f"vRt{n}RatioWideDef"] = f(d["ratio_wide"])
    mac[f"vRt{n}EpsBest"] = f"{float(e):g}"; mac[f"vRt{n}RBest"] = "\\infty" if float(r) > 1e3 else f"{float(r):g}"
    mac[f"vRt{n}PosBest"] = f"{b['npos_star']}"; mac[f"vRt{n}LamBest"] = f(b["lam_max"], 1)
    mac[f"vRt{n}PosDef"] = f"{d['npos_star']}"; mac[f"vRt{n}LamDef"] = f(d["lam_max"], 1)

# decomposition in the hyperchaotic median-|D| controls with a negative coupling term
_dk = ["repulsive", "p=1.5", "p=2", "p=2.5"]
mac["vDecFoldLo"] = f(min(MA["medD"][k]["fold_term"] for k in _dk), 1); mac["vDecFoldHi"] = f(max(MA["medD"][k]["fold_term"] for k in _dk), 1)
mac["vDecCoupLo"] = f(min(MA["medD"][k]["coupling_term"] for k in _dk), 1); mac["vDecCoupHi"] = f(max(MA["medD"][k]["coupling_term"] for k in _dk), 1)
mac["vDecFoldFour"] = f(MA["medD"]["p=4"]["fold_term"], 1); mac["vDecCoupFour"] = f(MA["medD"]["p=4"]["coupling_term"], 1)

# comparison with established maps
_prop = CP["Proposed (N=3)"]; _oth = [v for k, v in CP.items() if k != "Proposed (N=3)"]
mac["vCmpRatio"] = f(_prop["chaotic_ratio"]); mac["vCmpPos"] = f"{_prop['n_pos']}"
mac["vCmpLam"] = f(_prop["lam_max"]); mac["vCmpKS"] = f"{_prop['ks_bound']:.0f}"; mac["vCmpKSdec"] = f(_prop["ks_bound"], 1)
mac["vCmpFactor"] = f(_prop["lam_max"] / max(v["lam_max"] for v in _oth), 1)
mac["vCmpPerCoord"] = f(_prop["ks_bound"] / 9, 1)

# N-body scaling
mac["vNOneRatio"] = f(NB["N"]["1"]["ratio"]); mac["vNRatioThree"] = f(NB["N"]["3"]["ratio"])
_oth_r = [NB["N"][n]["ratio"] for n in ("2", "4", "5", "6")]
mac["vNRatioOthersLo"] = f(min(_oth_r)); mac["vNRatioOthersHi"] = f(max(_oth_r))
_lam = [NB["N"][n]["spec_star"][0] for n in ("2", "3", "4", "5", "6")]
mac["vNLamLo"] = f(min(_lam), 1); mac["vNLamHi"] = f(max(_lam), 1)
_sum = [float(np.sum(np.clip(NB["N"][n]["spec_star"], 0, None))) for n in ("2", "6")]
mac["vNSumLo"] = f(_sum[0], 1); mac["vNSumHi"] = f(_sum[1], 1)

# zero-crossing index of the spectrum (structure analysis)
_A = [r for r in SR["A"] if r["tag"] == "default"]
_N = np.array([r["N"] for r in _A]); _i0 = np.array([r["cross"] for r in _A])
_c3 = np.polyfit(_N[_N >= 3], _i0[_N >= 3], 1)
mac["vSlopeIz"] = f(_c3[0]); mac["vIcptIz"] = f(_c3[1])
mac["vXzeroEight"] = f(_i0[_N == 8][0] / 24, 2); mac["vXzeroThree"] = f(_i0[_N == 3][0] / 9, 2)
for r in _A:
    mac[f"vIz{ {2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight'}[r['N']] }"] = f(r["cross"])
_Bn = {n: [x["npos"] for x in SR["B"] if x["N"] == n] for n in range(2, 7)}
for n in range(3, 6):
    mac[f"vShareMax{NW[n]}"] = f"{100 * np.mean(np.array(_Bn[n]) >= 2 * n + 1 - 0.01):.0f}"
mac["vMaxNposSix"] = f(max(_Bn[6]), 1)
_C = {r["N"]: r for r in SR["C"]}
mac["vStaticErrLo"] = f(min(r["mean_abs_err"] for r in _C.values()), 1); mac["vStaticErrHi"] = f(max(r["mean_abs_err"] for r in _C.values()), 1)
mac["vGbarThree"] = f(_C[3]["gbar"], 1); mac["vGbarSix"] = f(_C[6]["gbar"], 1)

with open(f"{OUT}/numbers_new.tex", "w") as fh:
    fh.write("%% generated by kod/gen_tables.py -- do not edit by hand\n")
    for k, v in mac.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- mechanism table: four ways of setting the strength
rows = [("newton", r"Newtonian, $p=3$ (proposed)"), ("repulsive", r"Newtonian, repulsive"),
        ("p=2.5", r"power law, $p=2.5$"), ("p=3.5", r"power law, $p=3.5$"), ("p=2", r"power law, $p=2$"),
        ("p=4", r"power law, $p=4$"), ("p=1.5", r"power law, $p=1.5$"), ("p=1", r"power law, $p=1$"),
        ("p=5", r"power law, $p=5$"), ("p=6", r"power law, $p=6$"), ("linear", r"linear diffusive, $p=0$"),
        ("random_linear", r"random linear$^{\dagger}$"), ("offset", r"constant offset (no coupling)$^{\dagger}$"),
        ("none", r"none, $F_{i,k}=0$")]


def cell(r, star=False):
    return f"${r['chaotic_ratio']:.2f}$ / ${r['n_pos']:.1f}$" + ("$^{\\ddagger}$" if star else "")


L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Coupling & \shortstack{equal rms\\of $F$} & \shortstack{equal median\\of $|F|$} & \shortstack{equal median\\of $|D|$} & "
     r"\shortstack{best hyperchaotic\\strength ($s/s_0$)} & \shortstack{largest ratio\\at any strength} \\", r"\midrule"]
for k, lab in rows:
    c = [cell(ma(cr, k), star=(cr == "medD" and k in MA["medD"] and not MA["medD"][k].get("matched", True))) for cr in ("rms", "medF", "medD")]
    if k == "none":
        L.append(f"{lab} & {c[0]} & {c[1]} & {c[2]} & --- & --- \\\\")
        continue
    rb, nb, sb = (st_mean(k, "best_hyper", x) for x in ("chaotic_ratio", "n_pos", "s_rel"))
    best = f"${rb:.2f}$ / ${nb:.1f}$ ({sb:.2f})" if rb is not None else "none"
    mr = st_maxratio(k)
    mn = float(np.mean([r["best_ratio"]["n_pos"] for r in ST[k]]))
    L.append(f"{lab} & {c[0]} & {c[1]} & {c[2]} & {best} & ${mr:.2f}$ / ${mn:.1f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_mechanism", "\n".join(L) + "\n")

# ---------------------------------------------------------------- decomposition table (median-|D| matched)
drows = [("newton", r"Newtonian, $p=3$ (proposed)"), ("repulsive", r"Newtonian, repulsive"), ("p=2.5", r"power law, $p=2.5$"),
         ("p=3.5", r"power law, $p=3.5$"), ("p=2", r"power law, $p=2$"), ("p=1.5", r"power law, $p=1.5$"),
         ("p=4", r"power law, $p=4$"), ("p=5", r"power law, $p=5$"),
         ("linear", r"linear diffusive, $p=0$"), ("random_linear", r"random linear$^{\dagger}$"),
         ("offset", r"constant offset$^{\dagger}$"), ("none", r"none, $F_{i,k}=0$")]
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Coupling & $s/s_0$ & median $|D|$ & $\sum_i\lambda_i$ & Folding & Coupling \\", r"\midrule"]
for k, lab in drows:
    r = ma("medD", k)
    s_rel = r.get("scale", 1.0)
    L.append(f"{lab} & ${s_rel:.2f}$ & ${r['D_median']:.2f}$ & ${r['sum']:.2f}$ & ${r['fold_term']:.2f}$ & ${r['coupling_term']:.2f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_decomposition", "\n".join(L) + "\n")

# ---------------------------------------------------------------- N-body table
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Bodies & Dimension & \shortstack{Chaotic\\ratio} & $a^{*}$ & $\lambda_{\max}(a^{*})$ & \shortstack{Positive\\exponents at $a^{*}$} & \shortstack{$\sum\lambda_i^{+}$\\at $a^{*}$} \\",
     r"\midrule"]
for N in ["1", "2", "3", "4", "5", "6"]:
    r = NB["N"][N]; sp = np.array(r["spec_star"])
    L.append(f"$N={N}$ & ${3*int(N)}$ & ${r['ratio']:.2f}$ & ${r['a_star']:.1f}$ & ${sp[0]:.2f}$ & ${r['npos_star']:.1f}$ & ${sp[sp>0].sum():.1f}$ \\\\")
L += [r"\midrule", r"Masses $(1,1,\mu)$ & & & & $\lambda_{\max}(0.97)$ & \shortstack{Positive\\exponents at $0.97$} & \\", r"\midrule"]
for mu in ["0.01", "0.1", "0.5", "1.0", "2.0", "5.0"]:
    r = NB["mu"][mu]
    L.append(f"$\\mu={float(mu):g}$ & $9$ & ${r['ratio']:.2f}$ & --- & ${r['lam_ref']:.2f}$ & ${r['npos_ref']:.1f}$ & \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_nbody", "\n".join(L) + "\n")

# ---------------------------------------------------------------- re-tuning table
L = [r"\begin{tabular}{llccccccc}", r"\toprule",
     r"$N$ & Setting & $\varepsilon_s$ & $R^2$ & \shortstack{Chaotic\\ratio} & \shortstack{ratio over\\$[-14,14]$} & $a^{*}$ & $\lambda_{\max}(a^{*})$ & \shortstack{Positive\\exponents} \\",
     r"\midrule"]
for N in (3, 4, 5, 6):
    for lab, k in (("reference", f"{N}|0.1|2.0"), ("best ratio", best_cfg(N))):
        v = rt_full[k]; _, e, r = k.split("|"); rr = r"$\infty$" if float(r) > 1e3 else f"${float(r):g}$"
        L.append(f"{N if lab == 'reference' else ''} & {lab} & ${float(e):g}$ & {rr} & ${v['ratio']:.2f}$ & ${v['ratio_wide']:.2f}$ & "
                 f"${g1(v['a_star'])}$ & ${v['lam_max']:.2f}$ & ${v['npos_star']}$ \\\\")
    if N != 6:
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_retune", "\n".join(L) + "\n")

# ---------------------------------------------------------------- structure table (zero-crossing index)
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r"$N$ & Setting & $a^{*}$ & $2N+1$ & \shortstack{Positive\\exponents} & \shortstack{Zero-crossing\\index $i_0$} & \shortstack{$\lambda_{2N+1}$} & $\lambda_{2N+2}$ \\",
     r"\midrule"]
for r in sorted(SR["A"], key=lambda r: (r["N"], r["tag"] != "default")):
    N = r["N"]; sp = np.array(r["spec"]); ce = np.array(r["cross_each"])
    L.append(f"{N} & {r['tag']} & ${g1(r['a'])}$ & ${2*N+1}$ & ${r['npos'][str(THR)]}$ & ${r['cross']:.2f}\\pm{ce.std():.2f}$ & "
             f"${sp[2*N]:.2f}$ & ${sp[2*N+1]:.2f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_structure", "\n".join(L) + "\n")

# ---------------------------------------------------------------- comparison table
cite = {"zhou2014new1d": r"\citep{zhou2014new1d}", "hua2016lasm": r"\citep{hua2016lasm}", "henon1976": r"\citep{henon1976}",
        "baier1990maximum": r"\citep{baier1990maximum}", "classical": "", "this work": ""}
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Map & Dim. & \shortstack{Parameter\\range} & \shortstack{Chaotic\\ratio} & $\lambda_{\max}$ & \shortstack{Positive\\exponents} & \shortstack{$\sum\lambda_i^{+}$\\(nats/iter.)} \\",
     r"\midrule"]
for name, r in CP.items():
    nm = name.replace("Henon", r"H\'enon").replace("Proposed (N=3)", r"\textbf{Proposed} ($N=3$)")
    nm = nm.replace(" (Zhou 2014)", "").replace(" (Hua 2016)", "").replace(" (Baier-Klein)", "")
    lo, hi = r["param_range"]
    L.append(f"{nm} {cite[r['ref']]} & ${r['dim']}$ & $[{lo:g},{hi:g}]$ & ${r['chaotic_ratio']:.2f}$ & ${r['lam_max']:.2f}$ & ${r['n_pos']}$ & ${r['ks_bound']:.2f}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit("tab_compare", "\n".join(L) + "\n")
for t in ("tab_mechanism", "tab_decomposition", "tab_retune", "tab_structure", "tab_compare", "tab_nbody", "numbers_new"):
    print(open(f"{OUT}/{t}.tex").read())
