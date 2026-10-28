"""
Regenerate the manuscript figures from results/ (revision). Figures carry no in-plot titles; captions live in
main.tex. Colour convention of the original figures is kept (chaotic #1f4e79, regular #2e7d32, markers crimson).

	fig_lyapunov_spectrum      median lambda_max(a) with IQR band, eps_s = 0.1   (results/epsilon_scan.npz)
	fig_bifurcation            x1 after 500 transients, 1500 a-values + zoom      (computed, results/bifurcation.npz)
	fig_phase_portrait         (x1, y1, z1) projections
	fig_zero_one_test          (p, q) plane, standard Gottwald-Melbourne layout
	fig_permutation_entropy    PE (m = 7, tie policy) versus a                    (computed, results/pe_scan.npz)
	fig_correlation_dimension  ln C vs ln r and local slopes for 3/6/9-D projections (results/canonical_reference_corr.npz)
	fig_initial_sensitivity    on-attractor D_n with percentile bands + double-precision floor
	fig_jacobian_stepsize      finite-difference Jacobian error vs h; lambda_max vs delta0 (results/spectrum.npz)
	fig_ablation               lambda_max(a) for component and fold variants     (results/gravity_ablation.npz)

Usage: python figures_revision.py [name ...]   (no names = all)

tab-indented, snake_case, Allman braces.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import settings as st
from gravity_map import advance, iterate_map_vec, map_step_vec, DEFAULT_INITIAL_STATE, R2_BOUNDARY
from analysis_utils import load_json, zero_one_pq, permutation_entropy_vec

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300, "savefig.dpi": 300})

C_CHAOTIC = "#1f4e79"
C_REGULAR = "#2e7d32"
C_MARK = "crimson"


def out_path(name):
	st.ensure_dirs()
	return os.path.join(st.FIGURES_DIR, name)


def fig_lyapunov_spectrum():
	data = np.load(st.result_path("epsilon_scan.npz"))
	eps_index = int(np.argmin(np.abs(data["eps_values"] - 0.1)))
	lam = data["lam"][eps_index]
	a_grid = data["a_grid"]
	threshold = load_json("noise_floor.json")["lambda_threshold"]
	fig, ax = plt.subplots(figsize=(8, 4.3))
	ax.fill_between(a_grid, np.percentile(lam, 25, axis=1), np.percentile(lam, 75, axis=1), color=C_CHAOTIC, alpha=0.22, linewidth=0, label="çeyrekler arası aralık (%d başlangıç koşulu)" % lam.shape[1])
	ax.plot(a_grid, np.median(lam, axis=1), color=C_CHAOTIC, linewidth=1.1, label=r"medyan $\lambda_{\max}$")
	ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
	ax.axhline(threshold, color="0.45", linewidth=0.8, linestyle=":", label=r"$\lambda_{\mathrm{eşik}}=%.3f$" % threshold)
	ax.axvline(st.A_CHAOTIC, color=C_MARK, linewidth=1.0, linestyle=":")
	ax.axvline(st.A_REGULAR, color=C_REGULAR, linewidth=1.0, linestyle=":")
	ax.set_xlabel("Kontrol parametresi $a$")
	ax.set_ylabel(r"Maksimum Lyapunov üsteli $\lambda_{\max}$")
	ax.legend(loc="lower left", fontsize=9)
	fig.tight_layout()
	fig.savefig(out_path("fig_lyapunov_spectrum.png"))
	plt.close(fig)


def bifurcation_data(a_min, a_max, n_a, n_transient=500, n_plot=200):
	a_values = np.linspace(a_min, a_max, n_a)
	states = np.tile(DEFAULT_INITIAL_STATE, (n_a, 1))
	states = advance(states, a_values, n_transient)
	values = iterate_map_vec(states, a_values, n_plot)[1:, :, 0]
	return a_values, values


def fig_bifurcation():
	a_all, v_all = bifurcation_data(-3.0, 6.0, 1500)
	a_zoom, v_zoom = bifurcation_data(4.2, 6.0, 1200)
	np.savez(st.result_path("bifurcation.npz"), a_all=a_all, v_all=v_all, a_zoom=a_zoom, v_zoom=v_zoom)
	fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.4, 1.0]})
	axes[0].plot(np.repeat(a_all, v_all.shape[0]), v_all.T.ravel(), ",", color="#111111", alpha=0.25, rasterized=True)
	axes[0].axvline(st.A_CHAOTIC, color=C_MARK, linewidth=0.8, linestyle=":")
	axes[0].axvline(st.A_REGULAR, color=C_REGULAR, linewidth=0.8, linestyle=":")
	axes[0].axvspan(4.2, 6.0, color="orange", alpha=0.08)
	axes[1].plot(np.repeat(a_zoom, v_zoom.shape[0]), v_zoom.T.ravel(), ",", color="#8a3b00", alpha=0.35, rasterized=True)
	axes[1].axvline(st.A_REGULAR, color=C_REGULAR, linewidth=0.9, linestyle=":")
	for ax, tag in zip(axes, ("(a)", "(b)")):
		ax.set_xlabel("Kontrol parametresi $a$")
		ax.set_ylabel("$x_1$ (geçici rejim sonrası)")
		ax.text(0.01, 0.98, tag, transform=ax.transAxes, va="top", fontsize=11)
	fig.tight_layout()
	fig.savefig(out_path("fig_bifurcation.png"))
	plt.close(fig)


def fig_phase_portrait():
	fig = plt.figure(figsize=(10, 4.8))
	for index, (a_param, color, style) in enumerate(((st.A_CHAOTIC, C_CHAOTIC, {"linewidth": 0.2, "alpha": 0.7}), (st.A_REGULAR, C_REGULAR, {"linewidth": 0.4, "marker": "o", "markersize": 2.0}))):
		traj = iterate_map_vec(DEFAULT_INITIAL_STATE, a_param, 4000)[1000:]
		ax = fig.add_subplot(1, 2, index + 1, projection="3d")
		ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, **style)
		ax.set_xlabel("$x_1$")
		ax.set_ylabel("$y_1$")
		ax.set_zlabel("$z_1$")
		ax.set_title("(%s) $a=%s$" % ("ab"[index], a_param), fontsize=10)
	fig.tight_layout()
	fig.savefig(out_path("fig_phase_portrait.png"))
	plt.close(fig)


def fig_zero_one_test():
	ref = load_json("canonical_reference.json")
	c_value = 1.7
	fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
	for ax, a_param, color in ((axes[0], st.A_CHAOTIC, C_CHAOTIC), (axes[1], st.A_REGULAR, C_REGULAR)):
		start = advance(DEFAULT_INITIAL_STATE, a_param, st.N_TRANSIENT)
		series = iterate_map_vec(start, a_param, 5000)[1:, 0]
		p, q = zero_one_pq(series, c_value)
		ax.plot(p, q, linewidth=0.4, color=color)
		ax.set_xlabel("$p(n)$")
		ax.set_ylabel("$q(n)$")
		ax.set_title("$a=%s$, medyan $K=%.3f$" % (a_param, ref[str(a_param)]["zero_one_K"]["median"]), fontsize=10)
	fig.tight_layout()
	fig.savefig(out_path("fig_zero_one_test.png"))
	plt.close(fig)


def fig_permutation_entropy():
	a_values = np.round(np.arange(-3.0, 6.0 + 1e-9, 0.05), 10)
	n_series = 50000
	states = advance(np.tile(DEFAULT_INITIAL_STATE, (a_values.size, 1)), a_values, st.PE_ROUND_TRANSIENT)
	series = np.empty((n_series, a_values.size))
	for n in range(n_series):
		states = map_step_vec(states, a_values)
		series[n] = states[:, 0]
	pe = np.array([permutation_entropy_vec(series[:, i], st.PE_EMBEDDING, round_decimals=st.PE_ROUND_DECIMALS)["normalized"] for i in range(a_values.size)])
	np.savez(st.result_path("pe_scan.npz"), a_values=a_values, pe=pe, m=st.PE_EMBEDDING, n_series=n_series)
	fig, ax = plt.subplots(figsize=(8, 4.2))
	ax.plot(a_values, pe, color="#6a1b9a", linewidth=1.0)
	ax.axvline(st.A_CHAOTIC, color=C_MARK, linewidth=1.0, linestyle=":")
	ax.axvline(st.A_REGULAR, color=C_REGULAR, linewidth=1.0, linestyle=":")
	ax.set_xlabel("Kontrol parametresi $a$")
	ax.set_ylabel(r"$H_{\mathrm{PE}}^{\mathrm{norm}}$ ($m=%d$)" % st.PE_EMBEDDING)
	ax.set_ylim(-0.02, 1.02)
	fig.tight_layout()
	fig.savefig(out_path("fig_permutation_entropy.png"))
	plt.close(fig)


def fig_correlation_dimension():
	data = np.load(st.result_path("canonical_reference_corr.npz"))
	ref = load_json("canonical_reference.json")[str(st.A_CHAOTIC)]["correlation_dimension"]
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
	colors = {"3d_body1": "#e65100", "6d_bodies12": "#6a1b9a", "9d_full": C_CHAOTIC}
	labels = {"3d_body1": "3B: $(x_1,y_1,z_1)$", "6d_bodies12": "6B: cisim 1–2", "9d_full": "9B: tam durum"}
	for key in ("3d_body1", "6d_bodies12", "9d_full"):
		prefix = "a%s_%s" % (st.A_CHAOTIC, key)
		log_r, log_c, slopes = data[prefix + "_log_r"], data[prefix + "_log_c"], data[prefix + "_slopes"]
		d2 = ref[key]["d2"]
		axes[0].plot(log_r, log_c, ".-", color=colors[key], markersize=3, linewidth=0.8, label=labels[key] + ("" if d2 is None else ", $D_2\\approx%.2f$" % d2))
		axes[1].plot(log_r, slopes, ".-", color=colors[key], markersize=3, linewidth=0.8)
		if ref[key]["window"] is not None:
			s, e = ref[key]["window"]
			axes[1].plot(log_r[s:e], slopes[s:e], color=colors[key], linewidth=3.0, alpha=0.4)
	axes[0].set_xlabel(r"$\ln \varrho$")
	axes[0].set_ylabel(r"$\ln C(\varrho)$")
	axes[0].legend(fontsize=9)
	axes[1].set_xlabel(r"$\ln \varrho$")
	axes[1].set_ylabel(r"yerel eğim $d\ln C/d\ln\varrho$")
	fig.tight_layout()
	fig.savefig(out_path("fig_correlation_dimension.png"))
	plt.close(fig)


def fig_initial_sensitivity():
	data = np.load(st.result_path("sensitivity.npz"))
	precision = load_json("precision_check.json")["validity_horizon"][str(st.A_CHAOTIC)]["max_abs_error_per_step"]
	fig, ax = plt.subplots(figsize=(8, 4.5))
	for a_param, color, marker in ((st.A_CHAOTIC, C_CHAOTIC, "o"), (st.A_REGULAR, C_REGULAR, "s")):
		dists = data["on_attractor_a%s" % a_param]
		n_axis = np.arange(dists.shape[0])
		ax.fill_between(n_axis, np.percentile(dists, 10, axis=1), np.percentile(dists, 90, axis=1), color=color, alpha=0.18, linewidth=0)
		ax.semilogy(n_axis, np.median(dists, axis=1), color=color, marker=marker, markersize=3, linewidth=1.0, label="$a=%s$ (medyan, %%10–%%90 bandı)" % a_param)
	ax.semilogy(np.arange(1, len(precision) + 1), np.maximum(precision, 1.0e-17), color="0.35", linestyle="--", linewidth=1.0, label="sayısal taban: double − 60 basamak")
	ax.set_xlabel("İterasyon $n$")
	ax.set_ylabel(r"$D_n=\|\mathbf{x}_n-\mathbf{x}'_n\|$")
	ax.set_xlim(0, 40)
	ax.legend(fontsize=9)
	fig.tight_layout()
	fig.savefig(out_path("fig_initial_sensitivity.png"))
	plt.close(fig)


def fig_jacobian_stepsize():
	data = np.load(st.result_path("spectrum.npz"))
	steps, errors = data["jac_step_sizes"], data["jac_errors"]
	fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
	axes[0].loglog(steps, np.median(errors, axis=0), marker="o", color="#e65100", label="medyan")
	axes[0].loglog(steps, np.percentile(errors, 90, axis=0), marker="^", color="#e65100", linestyle="--", label="%90 persentil")
	axes[0].set_xlabel("Sonlu fark adım büyüklüğü $h$")
	axes[0].set_ylabel(r"$\|J_h-J\|_F/\|J\|_F$")
	axes[0].legend(fontsize=9)
	d0, lam = data["delta0_values"], data["delta0_lambda"]
	axes[1].errorbar(d0, lam.mean(axis=1), yerr=lam.std(axis=1, ddof=1), marker="s", color="#0277bd", capsize=3)
	axes[1].set_xscale("log")
	axes[1].set_xlabel(r"Başlangıç ayrışımı $\delta_0$")
	axes[1].set_ylabel(r"$\lambda_{\max}$ (ortalama $\pm$ ss)")
	for ax, tag in zip(axes, ("(a)", "(b)")):
		ax.text(0.02, 0.97, tag, transform=ax.transAxes, va="top")
	fig.tight_layout()
	fig.savefig(out_path("fig_jacobian_stepsize.png"))
	plt.close(fig)


def fig_ablation():
	data = np.load(st.result_path("gravity_ablation.npz"))
	a_grid = data["a_grid"]
	fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
	component = (("V0_full", "tam model", C_CHAOTIC), ("V1_no_phi", r"$\Phi_i=0$", "#6a1b9a"), ("V2_no_force", r"$F_{i,k}=0$", "#e65100"), ("V3_no_phi_force", r"$\Phi_i=F_{i,k}=0$", "#8a3b00"), ("V4_pure_1d", r"saf 1B: $\sin(G/aq)$", "0.4"))
	folds = (("V0_full", r"$\sin$", C_CHAOTIC), ("F_triangle", "üçgen dalga", "#6a1b9a"), ("F_sawtooth", "testere dişi", "#e65100"), ("F_tanh", r"$\tanh$ (katlamasız)", "0.4"))
	for ax, series, tag in ((axes[0], component, "(a)"), (axes[1], folds, "(b)")):
		for key, label, color in series:
			ax.plot(a_grid, data["median_lambda_" + key], color=color, linewidth=1.0, label=label)
		ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
		ax.set_xlabel("Kontrol parametresi $a$")
		ax.set_ylim(-3.0, 5.5)
		ax.legend(fontsize=8.5, loc="lower left")
		ax.text(0.01, 0.98, tag, transform=ax.transAxes, va="top")
	axes[0].set_ylabel(r"medyan $\lambda_{\max}$ (eksen $[-3,5.5]$ ile sınırlı)")
	fig.tight_layout()
	fig.savefig(out_path("fig_ablation.png"))
	plt.close(fig)


def fig_fixed_points():
	"""The three fixed points at a = 0.97 in the (x, y) plane: positions of the three bodies (results/fixed_points.json)."""
	from matplotlib.lines import Line2D
	roots = load_json("fixed_points.json")["points"][str(st.A_CHAOTIC)]["roots"]
	roots = sorted(roots, key=lambda r: not r["all_coordinates_equal"])
	body_colors = [C_CHAOTIC, C_MARK, C_REGULAR]
	body_markers = ["o", "s", "^"]
	radius = np.sqrt(R2_BOUNDARY)
	theta = np.linspace(0.0, 2.0 * np.pi, 400)
	fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.6), sharex=True, sharey=True)
	for index, (ax, root) in enumerate(zip(axes, roots)):
		pos = np.array(root["x"]).reshape(3, 3)
		symmetric = root["all_coordinates_equal"]
		ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="0.45", linewidth=1.0, linestyle="--")
		ax.set_aspect("equal")
		ax.set_xlim(-1.65, 1.65)
		ax.set_ylim(-1.65, 1.65)
		if not symmetric:
			closed = np.vstack([pos, pos[:1]])
			ax.plot(closed[:, 0], closed[:, 1], color="0.55", linewidth=0.9, linestyle=":")
		sizes = [520, 260, 90] if symmetric else [170, 170, 170]
		handles = []
		for body in range(3):
			ax.scatter(pos[body, 0], pos[body, 1], s=sizes[body], marker=body_markers[body], color=body_colors[body], edgecolor="black", linewidth=0.8, zorder=3 + body)
			digits = 5 if symmetric else 3
			label = ("Cisim %d: (%s)" % (body + 1, ", ".join("%.*f" % (digits, v) for v in pos[body]))).replace("-", "−")
			if not symmetric:
				ax.annotate(str(body + 1), (pos[body, 0], pos[body, 1]), xytext=(10, 8), textcoords="offset points", fontsize=11, fontweight="bold", zorder=10)
			handles.append(Line2D([], [], marker=body_markers[body], color=body_colors[body], markeredgecolor="black", linestyle="", markersize=9, label=label))
		handles.append(Line2D([], [], color="0.45", linestyle="--", label=r"küresel sınırın izdüşümü, yarıçap $\sqrt{2}$"))
		if symmetric:
			ax.annotate("üç cisim aynı konumda", xy=(pos[0, 0], pos[0, 1]), xytext=(0.0, 0.85), ha="center", arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=10)
		ax.text(0.03, 0.97, "(%s)" % "abc"[index], transform=ax.transAxes, va="top", fontsize=13)
		ax.set_xlabel("$x$")
		ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=9, handletextpad=0.5, labelspacing=0.5)
	axes[0].set_ylabel("$y$")
	fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.31, wspace=0.10)
	fig.savefig(out_path("fig_fixed_points.png"))
	plt.close(fig)


FIGURES = {
	"lyapunov_spectrum": fig_lyapunov_spectrum,
	"bifurcation": fig_bifurcation,
	"phase_portrait": fig_phase_portrait,
	"zero_one_test": fig_zero_one_test,
	"permutation_entropy": fig_permutation_entropy,
	"correlation_dimension": fig_correlation_dimension,
	"initial_sensitivity": fig_initial_sensitivity,
	"jacobian_stepsize": fig_jacobian_stepsize,
	"ablation": fig_ablation,
	"fixed_points": fig_fixed_points,
}


if __name__ == "__main__":
	names = sys.argv[1:] or list(FIGURES)
	for name in names:
		FIGURES[name]()
		print("figure", name, "done")
