"""
Parameter-space size and key sensitivity, revised (reviewer items R19, R21 middle path).

The manuscript's count N = (Delta / 1e-15)^10 = 2^508 is replaced by an explicitly labelled UPPER BOUND on the
number of distinct parameter settings, with three corrections:
	resolution     the IEEE-754 unit in the last place (ulp) at the magnitude of each parameter, not 1e-15
	usable         fraction of randomly drawn settings (x0 in [-1,1]^9, a in [-6, 8]) that are chaotic
	               (median-free single estimate lambda > noise-floor threshold)
	symmetry       settings related by the 288-element symmetry group give equivalent dynamics: -log2(288)
No brute-force security claim is derived. Key sensitivity is shown for a 1-ulp change of x1 over an ensemble,
together with the double-precision floor (double vs 60-digit trajectory from results/precision_check.json).

Outputs: results/key_space.json, figure fig_key_sensitivity.png

tab-indented, snake_case, Allman braces.
"""

import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import settings as st
from gravity_map import advance, map_step_vec, sample_initial_states
from lyapunov import two_traj_batch
from analysis_utils import save_json, load_json, wilson_interval

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300, "savefig.dpi": 300})

COORD_RANGE = (-1.0, 1.0)
A_RANGE = (-6.0, 8.0)
SYMMETRY_ORDER = 288


def resolution_bits():
	"""log2 of (range width / ulp at the largest magnitude of the range) per parameter."""
	coord_bits = math.log2((COORD_RANGE[1] - COORD_RANGE[0]) / np.spacing(max(abs(COORD_RANGE[0]), abs(COORD_RANGE[1]))))
	a_bits = math.log2((A_RANGE[1] - A_RANGE[0]) / np.spacing(max(abs(A_RANGE[0]), abs(A_RANGE[1]))))
	return coord_bits, a_bits


def usable_fraction(n_keys, threshold, seed, quick):
	rng = np.random.default_rng(seed)
	states = rng.uniform(COORD_RANGE[0], COORD_RANGE[1], size=(n_keys, 9))
	a_values = rng.uniform(A_RANGE[0], A_RANGE[1], size=n_keys)
	lam = two_traj_batch(states, a_values, 100 if quick else st.N_TRANSIENT_SCAN, 300 if quick else st.N_ITER_SCAN, st.DELTA0, perturb_seed=seed + 1)
	chaotic = lam > threshold
	return int(chaotic.sum()), n_keys, lam, a_values


def key_sensitivity(n_keys, n_steps, seed):
	keys = advance(sample_initial_states(n_keys, st.IC_RADIUS, seed), st.A_CHAOTIC, 0)
	perturbed = keys.copy()
	perturbed[:, 0] = np.nextafter(perturbed[:, 0], np.inf)
	dists = np.empty((n_steps + 1, n_keys))
	dists[0] = np.linalg.norm(perturbed - keys, axis=1)
	a_state, b_state = keys, perturbed
	for n in range(1, n_steps + 1):
		a_state = map_step_vec(a_state, st.A_CHAOTIC)
		b_state = map_step_vec(b_state, st.A_CHAOTIC)
		dists[n] = np.linalg.norm(b_state - a_state, axis=1)
	return dists


def main(quick=False):
	threshold = load_json("noise_floor.json")["lambda_threshold"]
	coord_bits, a_bits = resolution_bits()
	upper_bits = 9 * coord_bits + a_bits
	k, n, lam, a_values = usable_fraction(200 if quick else 2000, threshold, st.SEED_NIST, quick)
	fraction = k / n
	lo, hi = wilson_interval(k, n)
	symmetry_bits = math.log2(SYMMETRY_ORDER)
	usable_bits = upper_bits + math.log2(fraction) - symmetry_bits
	out = {
		"coordinate_range": COORD_RANGE,
		"a_range": A_RANGE,
		"resolution_bits_per_coordinate": coord_bits,
		"resolution_bits_a": a_bits,
		"upper_bound_bits_resolution_only": upper_bits,
		"manuscript_bits": 508.3,
		"chaotic_fraction_random_settings": fraction,
		"chaotic_fraction_wilson95": (lo, hi),
		"chaotic_fraction_bits_correction": math.log2(fraction),
		"symmetry_group_order": SYMMETRY_ORDER,
		"symmetry_bits_correction": -symmetry_bits,
		"upper_bound_bits_corrected": usable_bits,
		"threshold": threshold,
		"n_random_settings": n,
		"note": "Upper bound on distinct chaotic parameter settings; not a security estimate.",
	}
	dists = key_sensitivity(20, 40, st.SEED_PERTURB)
	out["key_sensitivity_median_distance"] = np.median(dists, axis=1)
	out["key_sensitivity_first_step_median_above_1e-2"] = int(np.argmax(np.median(dists, axis=1) > 1.0e-2))
	print("upper bound %.1f bits; chaotic fraction %.3f; corrected %.1f bits; sensitivity O(1e-2) at n=%d" % (upper_bits, fraction, usable_bits, out["key_sensitivity_first_step_median_above_1e-2"]))
	save_json("key_space_quick.json" if quick else "key_space.json", out)

	if not quick:
		floor = np.array(load_json("precision_check.json")["validity_horizon"][str(st.A_CHAOTIC)]["max_abs_error_per_step"])
		fig, ax = plt.subplots(figsize=(8, 4.4))
		n_axis = np.arange(dists.shape[0])
		ax.fill_between(n_axis, np.percentile(dists, 10, axis=1), np.percentile(dists, 90, axis=1), color="#8e24aa", alpha=0.18, linewidth=0)
		ax.semilogy(n_axis, np.median(dists, axis=1), color="#8e24aa", marker="o", markersize=3, linewidth=1.0, label=r"başlangıç farkı $%s\times10^{%d}$ (medyan, %%10–%%90)" % (("%.1e" % np.median(dists[0])).split("e")[0], int(("%.1e" % np.median(dists[0])).split("e")[1])))
		ax.semilogy(np.arange(1, floor.size + 1), np.maximum(floor, 1.0e-17), color="0.35", linestyle="--", linewidth=1.0, label="sayısal taban: double − 60 basamak")
		ax.set_xlabel("İterasyon $n$")
		ax.set_ylabel(r"$\|\mathbf{x}_n-\mathbf{x}'_n\|$")
		ax.legend(fontsize=9)
		fig.tight_layout()
		st.ensure_dirs()
		fig.savefig(os.path.join(st.FIGURES_DIR, "fig_key_sensitivity.png"), dpi=300)
		plt.close(fig)


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
