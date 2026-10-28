"""
Phase 1 verification: vectorised map vs scalar reference, analytic vs numerical Jacobian,
batched Lyapunov vs reference, clamp counter, timing. Writes results/verify_phase1.json.

tab-indented, snake_case, Allman braces.
"""

import json
import time
import numpy as np

import gravity_map as gm
from gravity_map import map_step, map_step_vec, DEFAULT_INITIAL_STATE, advance, sample_initial_states
from jacobian import analytic_jacobian
from lyapunov import max_lyapunov_exponent_two_traj, two_traj_batch, numerical_jacobian_vec, lyapunov_spectrum_qr, kaplan_yorke_dimension
import settings as st


def compare_trajectories(a_param, n_steps):
	"""Per-step max |x_ref - x_vec| and |x_ref - x_batch| from DEFAULT_INITIAL_STATE."""
	ref = DEFAULT_INITIAL_STATE.copy()
	vec = DEFAULT_INITIAL_STATE.copy()
	batch = np.tile(DEFAULT_INITIAL_STATE, (4, 1))
	rows = []
	first_nonzero_vec = None
	first_nonzero_batch = None
	for n in range(1, n_steps + 1):
		ref = map_step(ref, a_param)
		vec = map_step_vec(vec, a_param)
		batch = map_step_vec(batch, a_param)
		dv = float(np.max(np.abs(ref - vec)))
		db = float(np.max(np.abs(ref - batch[0])))
		if dv > 0.0 and first_nonzero_vec is None:
			first_nonzero_vec = n
		if db > 0.0 and first_nonzero_batch is None:
			first_nonzero_batch = n
		rows.append((n, dv, db))
	return {
		"max_diff_first_8_vec": max(r[1] for r in rows[:8]),
		"max_diff_first_8_batch": max(r[2] for r in rows[:8]),
		"first_step_with_nonzero_diff_vec": first_nonzero_vec,
		"first_step_with_nonzero_diff_batch": first_nonzero_batch,
	}


def jacobian_accuracy(a_param, n_points):
	"""Relative Frobenius error of analytic vs central-difference Jacobians on attractor points."""
	state = advance(DEFAULT_INITIAL_STATE, a_param, 500)
	errors = {1e-5: [], 1e-6: [], 1e-7: []}
	spectral_radius = []
	for _ in range(n_points):
		state = map_step_vec(state, a_param)
		exact = analytic_jacobian(state, a_param)
		spectral_radius.append(float(np.max(np.abs(np.linalg.eigvals(exact)))))
		norm = np.linalg.norm(exact)
		for h in errors:
			approx = numerical_jacobian_vec(state, a_param, h)
			errors[h].append(float(np.linalg.norm(approx - exact) / norm))
	out = {}
	for h, vals in errors.items():
		arr = np.array(vals)
		out["h=%g" % h] = {"median_rel_err": float(np.median(arr)), "p90_rel_err": float(np.percentile(arr, 90)), "max_rel_err": float(arr.max())}
	rho = np.array(spectral_radius)
	out["spectral_radius_on_attractor"] = {"median": float(np.median(rho)), "p90": float(np.percentile(rho, 90)), "max": float(rho.max())}
	return out


def main():
	results = {}
	gm.clamp_counter["count"] = 0

	results["trajectory_equivalence_a097"] = compare_trajectories(st.A_CHAOTIC, 60)
	results["trajectory_equivalence_a52"] = compare_trajectories(st.A_REGULAR, 60)

	t0 = time.perf_counter()
	lam_ref = max_lyapunov_exponent_two_traj(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, 500, 1500)
	t_ref = time.perf_counter() - t0
	t0 = time.perf_counter()
	lam_batch = float(two_traj_batch(DEFAULT_INITIAL_STATE[None, :], st.A_CHAOTIC, 500, 1500)[0])
	t_batch = time.perf_counter() - t0
	results["two_traj_reference_vs_batch_500_1500"] = {"reference": lam_ref, "batch": lam_batch, "seconds_reference": t_ref, "seconds_batch": t_batch}

	ics = sample_initial_states(100, st.IC_RADIUS, st.SEED_IC)
	t0 = time.perf_counter()
	advance(ics, st.A_CHAOTIC, 1000)
	results["timing_batch100_per_iter_seconds"] = (time.perf_counter() - t0) / 1000.0

	results["jacobian_accuracy_a097"] = jacobian_accuracy(st.A_CHAOTIC, 300)

	spec = lyapunov_spectrum_qr(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, 500, 3000, jacobian="analytic")
	spec_num = lyapunov_spectrum_qr(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, 500, 3000, jacobian="numerical")
	results["spectrum_a097_500_3000"] = {"analytic": spec.tolist(), "numerical_h1e-6": spec_num.tolist(), "kaplan_yorke_analytic": kaplan_yorke_dimension(spec), "sum_analytic": float(spec.sum())}

	variants = {"V0_full": {}, "V3_no_phi_no_force": {"use_phi": False, "use_force": False}, "V4_pure_1d": {"use_phi": False, "use_force": False, "use_boundary": False}}
	results["ablation_sanity_lambda_a097"] = {name: float(two_traj_batch(DEFAULT_INITIAL_STATE[None, :], st.A_CHAOTIC, 500, 1500, **kw)[0]) for name, kw in variants.items()}

	results["clamp_count_total"] = gm.clamp_counter["count"]

	with open(st.result_path("verify_phase1.json"), "w") as f:
		json.dump(results, f, indent=2)
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
