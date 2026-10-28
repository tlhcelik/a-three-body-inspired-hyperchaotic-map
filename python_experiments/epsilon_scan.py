"""
Softening-parameter scan (reviewer items R12, R8; regenerates tab:epsilon_* which had no producer).

For every eps_s in settings.EPSILON_VALUES and every a in settings.A_GRID (N = 141), N_IC initial
states are iterated as one batch. Recorded per trajectory:
	lambda_max        two-trajectory estimate (random tangent direction, renormalised every step)
	min_distance      minimum raw inter-body distance over transient + measurement
	final_step        |x_{n+1} - x_n|_inf at the last step (fixed-point collapse detector)
Classification per a uses the median lambda over initial states against the noise-floor threshold
from results/noise_floor.json; chaotic ratios carry Wilson 95% intervals. The selection criterion for
eps_s reported here is non-circular: fixed-point collapse and close-encounter frequency.

Outputs: results/epsilon_scan.json, results/epsilon_scan.npz

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import map_step_vec, sample_initial_states
from analysis_utils import save_json, load_json, summary_stats, wilson_interval, min_pair_distance, zero_one_test_gm
from gravity_map import advance, iterate_map_vec

CLOSE_ENCOUNTER = 0.05
FIXED_POINT_TOL = 1.0e-10


def scan_one_softening(eps_s, a_grid, n_ic, n_transient, n_iter, seed):
	"""Batch scan for one softening value. Returns arrays shaped (len(a_grid), n_ic)."""
	ics = sample_initial_states(n_ic, st.IC_RADIUS, seed)
	n_a = a_grid.size
	a_vec = np.repeat(a_grid, n_ic)
	states_a = np.tile(ics, (n_a, 1))
	min_dist = min_pair_distance(states_a)
	for _ in range(n_transient):
		states_a = map_step_vec(states_a, a_vec, softening=eps_s)
		min_dist = np.minimum(min_dist, min_pair_distance(states_a))

	rng = np.random.default_rng(seed + 7)
	direction = rng.normal(size=states_a.shape)
	direction /= np.linalg.norm(direction, axis=1, keepdims=True)
	states_b = states_a + st.DELTA0 * direction
	log_sum = np.zeros(states_a.shape[0])
	final_step = np.zeros(states_a.shape[0])
	for step_index in range(n_iter):
		previous = states_a
		states_a = map_step_vec(states_a, a_vec, softening=eps_s)
		states_b = map_step_vec(states_b, a_vec, softening=eps_s)
		diff = states_b - states_a
		dist = np.maximum(np.linalg.norm(diff, axis=1), 1.0e-300)
		log_sum += np.log(dist / st.DELTA0)
		states_b = states_a + diff * (st.DELTA0 / dist)[:, None]
		min_dist = np.minimum(min_dist, min_pair_distance(states_a))
		if step_index == n_iter - 1:
			final_step = np.max(np.abs(states_a - previous), axis=1)
	lam = (log_sum / n_iter).reshape(n_a, n_ic)
	return lam, min_dist.reshape(n_a, n_ic), final_step.reshape(n_a, n_ic)


def classify(lam, threshold):
	"""Per-a chaotic flag from the median over initial states."""
	return np.nanmedian(lam, axis=1) > threshold


def main(quick=False):
	try:
		noise = load_json("noise_floor_quick.json" if quick else "noise_floor.json")
		threshold = noise["lambda_threshold"]
		sensitivity = noise["sensitivity_thresholds"]
	except FileNotFoundError:
		threshold = 0.05
		sensitivity = [0.01, 0.05, 0.1, 0.2, 0.5]

	a_grid = st.A_GRID[::10] if quick else st.A_GRID
	n_ic = 3 if quick else st.N_IC
	n_transient = 100 if quick else st.N_TRANSIENT_SCAN
	n_iter = 300 if quick else st.N_ITER_SCAN
	eps_values = st.EPSILON_VALUES

	lam_all = np.empty((len(eps_values), a_grid.size, n_ic))
	dist_all = np.empty_like(lam_all)
	step_all = np.empty_like(lam_all)
	out = {"threshold": threshold, "a_grid": a_grid, "n_ic": n_ic, "n_transient": n_transient, "n_iter": n_iter, "close_encounter_distance": CLOSE_ENCOUNTER, "fixed_point_tol": FIXED_POINT_TOL, "softening": {}}

	ref_index = int(np.argmin(np.abs(a_grid - st.A_CHAOTIC)))
	for eps_index, eps_s in enumerate(eps_values):
		lam, min_dist, final_step = scan_one_softening(eps_s, a_grid, n_ic, n_transient, n_iter, st.SEED_IC)
		lam_all[eps_index], dist_all[eps_index], step_all[eps_index] = lam, min_dist, final_step

		chaotic = classify(lam, threshold)
		k_chaotic = int(chaotic.sum())
		collapsed = (final_step < FIXED_POINT_TOL) & (lam < 0.0)
		collapsed_a = np.mean(collapsed, axis=1) > 0.5
		close = min_dist < CLOSE_ENCOUNTER

		states_ref = advance(sample_initial_states(5 if not quick else 2, st.IC_RADIUS, st.SEED_IC), st.A_CHAOTIC, n_transient, softening=eps_s)
		series = iterate_map_vec(states_ref, st.A_CHAOTIC, 2000 if quick else 5000, softening=eps_s)[1:, :, 0]
		k_ref = [zero_one_test_gm(series[:, i], n_c=10 if quick else st.ZERO_ONE_N_C, seed=st.SEED_ZERO_ONE)[0] for i in range(series.shape[1])]

		out["softening"][repr(eps_s)] = {
			"chaotic_ratio": k_chaotic / a_grid.size,
			"chaotic_count": k_chaotic,
			"chaotic_ratio_wilson95": wilson_interval(k_chaotic, a_grid.size),
			"chaotic_ratio_threshold_sensitivity": {repr(t): float(classify(lam, t).mean()) for t in sensitivity},
			"lambda_median_over_ic": summary_stats(np.nanmedian(lam, axis=1)),
			"lambda_max_over_grid": float(np.nanmax(np.nanmedian(lam, axis=1))),
			"fixed_point_collapse_a_count": int(collapsed_a.sum()),
			"fixed_point_collapse_trajectory_fraction": float(collapsed.mean()),
			"close_encounter_trajectory_fraction": float(close.mean()),
			"close_encounter_a_count": int(np.any(close, axis=1).sum()),
			"min_distance_absolute": float(np.min(min_dist)),
			"min_distance_mean": float(np.mean(min_dist)),
			"reference_a097": {
				"lambda": summary_stats(lam[ref_index]),
				"zero_one_K_median_over_ic": float(np.median(k_ref)),
				"zero_one_K_per_ic": k_ref,
			},
		}
		print("eps_s=%g ratio=%.3f collapse_a=%d close_frac=%.3f lam097=%.3f K097=%.3f" % (eps_s, k_chaotic / a_grid.size, collapsed_a.sum(), close.mean(), np.median(lam[ref_index]), np.median(k_ref)))

	name = "epsilon_scan_quick" if quick else "epsilon_scan"
	np.savez(st.result_path(name + ".npz"), eps_values=np.array(eps_values), a_grid=a_grid, lam=lam_all, min_dist=dist_all, final_step=step_all)
	print("->", save_json(name + ".json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
