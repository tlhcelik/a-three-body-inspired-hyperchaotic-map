"""
R^2 (boundary parameter) sensitivity analysis, revised (reviewer items R8, R12).

The original r2_lambda_table used np.std over the whole (T, 9) array, i.e. the spread ACROSS the nine
coordinates, so fixed points with nine different constant coordinates scored ~0.94. The revised table
reports, per R^2 value and over N_IC initial states:
	lambda_max (two-trajectory)
	regime: fixed point / periodic (period p) / aperiodic, from the tail of each trajectory
	temporal spread: mean over the nine coordinates of each coordinate's standard deviation IN TIME
	Shannon entropy (bits, 256 bins) of x1

Outputs: results/r2_boundary.json, results/r2_boundary.npz; figures fig_r2_lyapunov_scan.png, fig_r2_comparison.png

tab-indented, snake_case, Allman braces.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import settings as st
from gravity_map import advance, iterate_map_vec, sample_initial_states, DEFAULT_INITIAL_STATE
from lyapunov import two_traj_batch
from analysis_utils import save_json, summary_stats, detect_period, shannon_entropy_bits

R2_TABLE = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def regime_label(period):
	"""Human-readable regime from a detected period."""
	if period is None:
		return "aperiodic"
	if period == 1:
		return "fixed_point"
	return "period_%d" % period


def r2_table(n_ic, n_transient, n_iter, n_tail):
	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	rows = []
	for r2 in R2_TABLE:
		lam = two_traj_batch(ics, st.A_CHAOTIC, n_transient, n_iter, st.DELTA0, perturb_seed=st.SEED_PERTURB, r2_boundary=r2)
		start = advance(ics, st.A_CHAOTIC, n_transient + n_iter, r2_boundary=r2)
		tail = iterate_map_vec(start, st.A_CHAOTIC, n_tail, r2_boundary=r2)[1:]
		regimes = {}
		spreads = []
		shannon = []
		for i in range(n_ic):
			label = regime_label(detect_period(tail[:, i, :]))
			regimes[label] = regimes.get(label, 0) + 1
			spreads.append(float(np.mean(np.std(tail[:, i, :], axis=0))))
			shannon.append(shannon_entropy_bits(tail[:, i, 0])[0])
		rows.append({
			"r2": r2,
			"lambda": summary_stats(lam),
			"regimes": regimes,
			"temporal_spread": summary_stats(spreads),
			"shannon_bits_x1": summary_stats(shannon),
		})
		print("R2=%5.1f lambda=%.3f+-%.3f regimes=%s spread=%.3f" % (r2, np.nanmean(lam), np.nanstd(lam), regimes, np.mean(spreads)))
	return rows


def r2_scan(n_ic, n_transient, n_iter, n_values):
	"""lambda_max vs R^2 as one batch (R^2 values x initial states)."""
	r2_values = np.linspace(0.2, 25.0, n_values)
	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	r2_vec = np.repeat(r2_values, n_ic)
	states = np.tile(ics, (n_values, 1))
	lam = two_traj_batch(states, st.A_CHAOTIC, n_transient, n_iter, st.DELTA0, perturb_seed=st.SEED_PERTURB, r2_boundary=r2_vec)
	return r2_values, lam.reshape(n_values, n_ic)


def fig_r2_lyapunov_scan(r2_values, lam, path):
	median = np.nanmedian(lam, axis=1)
	q1 = np.nanpercentile(lam, 25, axis=1)
	q3 = np.nanpercentile(lam, 75, axis=1)
	fig, ax = plt.subplots(figsize=(8, 4.2))
	ax.fill_between(r2_values, q1, q3, color="#6a1b9a", alpha=0.22, linewidth=0, label="çeyrekler arası aralık (%d başlangıç koşulu)" % lam.shape[1])
	ax.plot(r2_values, median, color="#6a1b9a", linewidth=1.2, label="medyan $\\lambda_{\\max}$")
	ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
	ax.axvline(2.0, color="crimson", linewidth=1.0, linestyle=":", label="$R^2=2.0$ (kullanılan değer)")
	ax.set_xlabel("Sınır parametresi $R^2$")
	ax.set_ylabel(r"Maksimum Lyapunov üsteli $\lambda_{\max}$")
	ax.legend(fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=300)
	plt.close(fig)


def fig_r2_comparison(path, n_transient=500, n_show=150):
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
	colors = ["#c0392b", "#1f4e79", "#1b7a3d"]
	for ax, r2, color in zip(axes, (0.5, 2.0, 20.0), colors):
		traj = iterate_map_vec(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, n_transient + n_show, r2_boundary=r2)
		ax.plot(np.arange(n_transient, n_transient + n_show + 1), traj[n_transient:, 0], linewidth=0.7, color=color)
		ax.set_xlabel("İterasyon $n$")
		ax.set_ylabel("$x_1$")
		ax.set_title("$R^2=%s$" % r2, fontsize=10)
		ax.set_ylim(-1.05, 1.05)
	fig.tight_layout()
	fig.savefig(path, dpi=300)
	plt.close(fig)


def main(quick=False):
	n_ic = 3 if quick else st.N_IC
	n_transient = 200 if quick else st.N_TRANSIENT
	n_iter = 500 if quick else 5000
	rows = r2_table(n_ic, n_transient, n_iter, 300)
	r2_values, lam = r2_scan(n_ic, 200 if quick else st.N_TRANSIENT_SCAN, 300 if quick else st.N_ITER_SCAN, 20 if quick else 100)
	name = "r2_boundary_quick" if quick else "r2_boundary"
	np.savez(st.result_path(name + ".npz"), r2_values=r2_values, lam=lam)
	save_json(name + ".json", {"a": st.A_CHAOTIC, "n_ic": n_ic, "n_transient": n_transient, "n_iter": n_iter, "table": rows})
	if not quick:
		st.ensure_dirs()
		fig_r2_lyapunov_scan(r2_values, lam, os.path.join(st.FIGURES_DIR, "fig_r2_lyapunov_scan.png"))
		fig_r2_comparison(os.path.join(st.FIGURES_DIR, "fig_r2_comparison.png"))
	print("done", name)


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
