"""
Invariant density and Shannon entropy of the coordinate series (reviewer item R22).

For a in A_BAND the x1 series (N_IC_DENSITY states x N_PER_IC samples after transient) and the pooled
nine coordinates are histogrammed on 256 bins over [-1, 1]. The arcsine density 1/(pi sqrt(1-x^2)) is
reported as a reference shape, with the Shannon entropy of its own 256-bin discretisation.

Outputs: results/shannon_density.json, results/shannon_density.npz, figure fig_density.png

tab-indented, snake_case, Allman braces.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import settings as st
from gravity_map import advance, iterate_map_vec, sample_initial_states
from analysis_utils import save_json, shannon_entropy_bits

N_IC_DENSITY = 5


def arcsine_bin_probabilities(n_bins):
	edges = np.linspace(-1.0, 1.0, n_bins + 1)
	cdf = 0.5 + np.arcsin(np.clip(edges, -1.0, 1.0)) / np.pi
	return np.diff(cdf)


def main(quick=False):
	n_per_ic = 20000 if quick else 200000
	edges = np.linspace(-1.0, 1.0, st.SHANNON_BINS + 1)
	arcsine = arcsine_bin_probabilities(st.SHANNON_BINS)
	arcsine_bits = float(-np.sum(arcsine[arcsine > 0] * np.log2(arcsine[arcsine > 0])))
	out = {"bins": st.SHANNON_BINS, "samples_per_ic": n_per_ic, "n_ic": N_IC_DENSITY, "arcsine_reference_bits": arcsine_bits, "points": {}}
	arrays = {"edges": edges, "arcsine_prob": arcsine}
	for a_param in st.A_BAND:
		start = advance(sample_initial_states(N_IC_DENSITY, st.IC_RADIUS, st.SEED_IC), a_param, st.N_TRANSIENT)
		traj = iterate_map_vec(start, a_param, n_per_ic)[1:]
		x1 = traj[:, :, 0].ravel()
		pooled = traj.reshape(-1, 9).ravel()
		hist_x1, _ = np.histogram(x1, bins=edges)
		hist_pool, _ = np.histogram(pooled, bins=edges)
		prob_x1 = hist_x1 / hist_x1.sum()
		out["points"][str(a_param)] = {
			"shannon_bits_x1": shannon_entropy_bits(x1)[0],
			"shannon_bits_pooled": shannon_entropy_bits(pooled)[0],
			"total_variation_to_uniform_x1": float(0.5 * np.sum(np.abs(prob_x1 - 1.0 / st.SHANNON_BINS))),
			"total_variation_to_arcsine_x1": float(0.5 * np.sum(np.abs(prob_x1 - arcsine))),
			"edge_mass_x1_abs_gt_0.9": float(np.mean(np.abs(x1) > 0.9)),
		}
		arrays["hist_x1_a%s" % a_param] = hist_x1
		arrays["hist_pooled_a%s" % a_param] = hist_pool
		print("a=%s H(x1)=%.3f bits H(pooled)=%.3f TV_uniform=%.3f TV_arcsine=%.3f" % (a_param, out["points"][str(a_param)]["shannon_bits_x1"], out["points"][str(a_param)]["shannon_bits_pooled"], out["points"][str(a_param)]["total_variation_to_uniform_x1"], out["points"][str(a_param)]["total_variation_to_arcsine_x1"]))
	name = "shannon_density_quick" if quick else "shannon_density"
	np.savez(st.result_path(name + ".npz"), **arrays)
	save_json(name + ".json", out)

	if not quick:
		centres = 0.5 * (edges[:-1] + edges[1:])
		width = edges[1] - edges[0]
		fig, ax = plt.subplots(figsize=(7.5, 3.8))
		hist = arrays["hist_x1_a%s" % st.A_CHAOTIC]
		ax.bar(centres, hist / hist.sum() / width, width=width, color="#1f77b4", alpha=0.75, label="$x_1$, $a=0.97$")
		ax.plot(centres, arcsine / width, color="#d62728", linewidth=1.2, label="arksinüs yoğunluğu")
		ax.axhline(0.5, color="0.3", linestyle="--", linewidth=0.9, label="düzgün yoğunluk")
		ax.set_xlabel("$x_1$")
		ax.set_ylabel("olasılık yoğunluğu")
		ax.set_xlim(-1.0, 1.0)
		ax.legend(fontsize=9)
		fig.tight_layout()
		st.ensure_dirs()
		fig.savefig(os.path.join(st.FIGURES_DIR, "fig_density.png"), dpi=300)
		plt.close(fig)
	print("done", name)


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
