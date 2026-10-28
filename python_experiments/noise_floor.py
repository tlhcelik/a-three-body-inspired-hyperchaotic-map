"""
Noise floor of the finite-time largest-Lyapunov estimator (reviewer item R12).

A single scan estimate uses N_ITER_SCAN iterations. Its statistical spread is measured by
splitting long trajectories into consecutive windows of that length at the chaotic and the
regular reference parameters. Because the scan classifies each a by the median over N_IC
estimates, the threshold is three standard errors of that median (never below 0.01).

Output: results/noise_floor.json

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import map_step_vec, advance, sample_initial_states
from analysis_utils import save_json, summary_stats


def window_estimates(a_param, n_ic, n_windows, window_len, seed):
	"""Finite-time lambda estimates, shape (n_windows, n_ic), from consecutive windows."""
	states_a = advance(sample_initial_states(n_ic, st.IC_RADIUS, seed), a_param, st.N_TRANSIENT_SCAN)
	rng = np.random.default_rng(seed + 1)
	direction = rng.normal(size=states_a.shape)
	direction /= np.linalg.norm(direction, axis=1, keepdims=True)
	states_b = states_a + st.DELTA0 * direction
	estimates = np.empty((n_windows, n_ic))
	for window_index in range(n_windows):
		log_sum = np.zeros(n_ic)
		for _ in range(window_len):
			states_a = map_step_vec(states_a, a_param)
			states_b = map_step_vec(states_b, a_param)
			diff = states_b - states_a
			dist = np.maximum(np.linalg.norm(diff, axis=1), 1.0e-300)
			log_sum += np.log(dist / st.DELTA0)
			states_b = states_a + diff * (st.DELTA0 / dist)[:, None]
		estimates[window_index] = log_sum / window_len
	return estimates


def main(quick=False):
	n_ic = 4 if quick else 10
	n_windows = 4 if quick else 20
	window_len = 300 if quick else st.N_ITER_SCAN
	out = {"window_len": window_len, "n_windows": n_windows, "n_ic": n_ic, "points": {}}
	window_sds = []
	for a_param in (st.A_CHAOTIC, st.A_REGULAR):
		est = window_estimates(a_param, n_ic, n_windows, window_len, st.SEED_IC)
		per_ic_sd = est.std(axis=0, ddof=1)
		window_sds.append(float(np.max(per_ic_sd)))
		out["points"][str(a_param)] = {
			"all_estimates": summary_stats(est.ravel()),
			"window_sd_per_ic": summary_stats(per_ic_sd),
			"ic_means": est.mean(axis=0).tolist(),
		}
	out["max_window_sd"] = max(window_sds)
	# The classifier uses the median of N_IC independent window-length estimates; its standard error
	# is about sqrt(pi/2) * sd / sqrt(N_IC). The threshold is three standard errors, floored at 0.01.
	out["median_standard_error"] = float(np.sqrt(np.pi / 2.0) * out["max_window_sd"] / np.sqrt(st.N_IC))
	out["lambda_threshold"] = max(0.01, 3.0 * out["median_standard_error"])
	out["sensitivity_thresholds"] = [0.01, 0.05, 0.1, 0.2, 0.5]
	path = save_json("noise_floor_quick.json" if quick else "noise_floor.json", out)
	print("lambda_threshold =", out["lambda_threshold"], "->", path)


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
