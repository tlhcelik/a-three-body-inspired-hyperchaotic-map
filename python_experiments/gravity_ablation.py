"""
Component ablation (reviewer items R24, R27).

Variants of the map (all other settings fixed):
	V0_full            reference map
	V1_no_phi          Phi_i = 0
	V2_no_force        F_{i,k} = 0
	V3_no_phi_force    Phi_i = F_{i,k} = 0 (no gravitational terms)
	V4_pure_1d         Phi = F = 0 and no boundary term: q' = sin(G / (a q))
	F_sawtooth / F_triangle / F_tanh / F_none : reference map with the sin fold replaced

Per variant:
	lambda_max at a = 0.97 over N_IC states (NaN for diverged trajectories)
	lambda_max(a) over A_GRID (median over ABL_N_IC states) and the chaotic ratio with the noise-floor threshold
	boundedness: fraction of trajectories that stay finite with |x| < 1e8 during the scan
	0-1 test K and permutation entropy (m=7) at a = 0.97 when bounded

Outputs: results/gravity_ablation.json, results/gravity_ablation.npz

tab-indented, snake_case, Allman braces.
"""

import sys
import warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)

import settings as st
from gravity_map import map_step_vec, sample_initial_states, advance, iterate_map_vec
from analysis_utils import save_json, load_json, summary_stats, wilson_interval, zero_one_test_gm, permutation_entropy_vec

ABL_N_IC = 10
DIVERGENCE = 1.0e8

VARIANTS = {
	"V0_full": {},
	"V1_no_phi": {"use_phi": False},
	"V2_no_force": {"use_force": False},
	"V3_no_phi_force": {"use_phi": False, "use_force": False},
	"V4_pure_1d": {"use_phi": False, "use_force": False, "use_boundary": False},
	"F_sawtooth": {"fold": "sawtooth"},
	"F_triangle": {"fold": "triangle"},
	"F_tanh": {"fold": "tanh"},
	"F_none": {"fold": "none"},
}


def guarded_two_traj(states, a_vec, n_transient, n_iter, seed, kwargs):
	"""Two-trajectory lambda with divergence guard; returns (lambda, bounded_flag)."""
	with np.errstate(all="ignore"):
		state_a = np.array(states, dtype=float)
		alive = np.ones(state_a.shape[0], dtype=bool)
		for _ in range(n_transient):
			state_a = map_step_vec(state_a, a_vec, **kwargs)
			bad = ~np.all(np.isfinite(state_a), axis=1) | (np.max(np.abs(state_a), axis=1) > DIVERGENCE)
			alive &= ~bad
			state_a[~alive] = 0.3
		rng = np.random.default_rng(seed)
		direction = rng.normal(size=state_a.shape)
		direction /= np.linalg.norm(direction, axis=1, keepdims=True)
		state_b = state_a + st.DELTA0 * direction
		log_sum = np.zeros(state_a.shape[0])
		for _ in range(n_iter):
			state_a = map_step_vec(state_a, a_vec, **kwargs)
			state_b = map_step_vec(state_b, a_vec, **kwargs)
			bad = ~np.all(np.isfinite(state_a), axis=1) | (np.max(np.abs(state_a), axis=1) > DIVERGENCE) | ~np.all(np.isfinite(state_b), axis=1)
			alive &= ~bad
			state_a[~alive] = 0.3
			state_b[~alive] = 0.3 + st.DELTA0
			diff = state_b - state_a
			dist = np.maximum(np.linalg.norm(diff, axis=1), 1.0e-300)
			log_sum += np.log(dist / st.DELTA0)
			state_b = state_a + diff * (st.DELTA0 / dist)[:, None]
	lam = log_sum / n_iter
	lam[~alive] = np.nan
	return lam, alive


def main(quick=False):
	try:
		threshold = load_json("noise_floor.json")["lambda_threshold"]
	except FileNotFoundError:
		threshold = 0.05
	a_grid = st.A_GRID[::10] if quick else st.A_GRID
	n_ic_ref = 3 if quick else st.N_IC
	n_ic_scan = 2 if quick else ABL_N_IC
	n_tr_ref = 200 if quick else st.N_TRANSIENT
	n_it_ref = 500 if quick else 5000
	n_tr_scan = 100 if quick else st.N_TRANSIENT_SCAN
	n_it_scan = 300 if quick else st.N_ITER_SCAN
	out = {"threshold": threshold, "divergence_bound": DIVERGENCE, "variants": {}}
	arrays = {"a_grid": a_grid}
	ics_ref = sample_initial_states(n_ic_ref, st.IC_RADIUS, st.SEED_IC)
	ics_scan = sample_initial_states(n_ic_scan, st.IC_RADIUS, st.SEED_IC)
	ref_index = int(np.argmin(np.abs(a_grid - st.A_CHAOTIC)))

	for name, kwargs in VARIANTS.items():
		lam_ref, alive_ref = guarded_two_traj(ics_ref, st.A_CHAOTIC, n_tr_ref, n_it_ref, st.SEED_PERTURB, kwargs)
		a_vec = np.repeat(a_grid, n_ic_scan)
		states = np.tile(ics_scan, (a_grid.size, 1))
		lam_scan, alive_scan = guarded_two_traj(states, a_vec, n_tr_scan, n_it_scan, st.SEED_PERTURB, kwargs)
		lam_scan = lam_scan.reshape(a_grid.size, n_ic_scan)
		alive_scan = alive_scan.reshape(a_grid.size, n_ic_scan)
		with np.errstate(all="ignore"):
			median_lam = np.nanmedian(np.where(alive_scan, lam_scan, np.nan), axis=1)
		bounded_a = alive_scan.all(axis=1)
		chaotic = bounded_a & (np.nan_to_num(median_lam, nan=-np.inf) > threshold)
		entry = {
			"kwargs": kwargs,
			"lambda_a097": summary_stats(lam_ref),
			"bounded_fraction_a097": float(alive_ref.mean()),
			"bounded_fraction_scan": float(alive_scan.mean()),
			"bounded_a_count": int(bounded_a.sum()),
			"chaotic_ratio": float(chaotic.mean()),
			"chaotic_count": int(chaotic.sum()),
			"chaotic_ratio_wilson95": wilson_interval(int(chaotic.sum()), a_grid.size),
			"lambda_median_at_a097_scan": float(median_lam[ref_index]) if np.isfinite(median_lam[ref_index]) else None,
		}
		if alive_ref.all():
			start = advance(ics_ref[:3], st.A_CHAOTIC, n_tr_ref, **kwargs)
			series = iterate_map_vec(start, st.A_CHAOTIC, 2000 if quick else 5000, **kwargs)[1:, :, 0]
			entry["zero_one_K_median"] = float(np.median([zero_one_test_gm(series[:, i], n_c=10 if quick else st.ZERO_ONE_N_C, seed=st.SEED_ZERO_ONE)[0] for i in range(series.shape[1])]))
			long_series = iterate_map_vec(start[0], st.A_CHAOTIC, 20000 if quick else 100000, **kwargs)[1:, 0]
			entry["permutation_entropy_m7"] = permutation_entropy_vec(long_series, 7, round_decimals=st.PE_ROUND_DECIMALS)["normalized"]
		out["variants"][name] = entry
		arrays["median_lambda_" + name] = median_lam
		arrays["bounded_" + name] = bounded_a
		print("%-16s lam097=%s bounded097=%.2f ratio=%.3f K=%s PE7=%s" % (name, "%.3f+-%.3f" % (entry["lambda_a097"].get("mean", np.nan), entry["lambda_a097"].get("sd", np.nan)) if entry["lambda_a097"]["n"] else "diverged", entry["bounded_fraction_a097"], entry["chaotic_ratio"], entry.get("zero_one_K_median"), entry.get("permutation_entropy_m7")))

	name = "gravity_ablation_quick" if quick else "gravity_ablation"
	np.savez(st.result_path(name + ".npz"), **arrays)
	print("->", save_json(name + ".json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
