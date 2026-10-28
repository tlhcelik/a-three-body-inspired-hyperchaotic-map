"""
Regular reference point a = 5.2: attractors, basins and on-attractor sensitivity (reviewer item R13),
plus the rebuilt tab:initial_sensitivity_values (median D_n over random perturbation directions,
started ON the attractor after a transient) for a = 0.97 and a = 5.2.

	basin_scan         initial states around the default state and uniformly in [-0.9, 0.9]^9;
	                   each attractor is identified by its period and orbit signature
	on_attractor       D_n for 100 random directions, delta0 = 1e-8, after N_TRANSIENT iterations
	original_protocol  the manuscript's protocol (no transient, perturbation along x1) for comparison

Outputs: results/a52_multistability.json, results/sensitivity.npz

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import advance, iterate_map_vec, map_step_vec, sample_initial_states, DEFAULT_INITIAL_STATE
from lyapunov import two_traj_batch
from analysis_utils import save_json, summary_stats, detect_period

N_STEPS_DN = 60


def orbit_signature(tail, period):
	"""Phase-independent orbit representation: the `period` points sorted lexicographically."""
	points = np.asarray(tail[:period])
	order = np.lexsort(points.T[::-1])
	return points[order]


def same_orbit_points(sig_a, sig_b, tol=1.0e-6):
	"""True when two sorted orbit point sets coincide within `tol`."""
	return sig_a.shape == sig_b.shape and float(np.max(np.abs(sig_a - sig_b))) < tol


def basin_scan(a_param, n_local, n_global, n_transient, seed):
	rng = np.random.default_rng(seed)
	local = sample_initial_states(n_local, st.IC_RADIUS, seed)
	global_states = rng.uniform(-0.9, 0.9, size=(n_global, 9))
	states = np.concatenate([local, global_states], axis=0)
	origin = np.array(["local"] * n_local + ["global"] * n_global)
	end = advance(states, a_param, n_transient)
	tail = iterate_map_vec(end, a_param, 200)[1:]
	lam = two_traj_batch(end, a_param, 0, 2000, st.DELTA0, perturb_seed=seed + 3)
	attractors = []
	for i in range(states.shape[0]):
		period = detect_period(tail[:, i, :], max_period=64, tol=1.0e-8)
		sig = None if period is None else orbit_signature(tail[:, i, :], period)
		entry = None
		for candidate in attractors:
			if period is None and candidate["period"] is None:
				entry = candidate
				break
			if period is not None and candidate["period"] == period and same_orbit_points(candidate["signature"], sig):
				entry = candidate
				break
		if entry is None:
			entry = {"period": period, "signature": sig, "count_local": 0, "count_global": 0, "lambda": [], "example_point": tail[0, i, :]}
			attractors.append(entry)
		entry["count_" + origin[i]] += 1
		entry["lambda"].append(lam[i])
	summary = []
	for entry in attractors:
		summary.append({
			"period": entry["period"],
			"count_local": entry["count_local"],
			"count_global": entry["count_global"],
			"fraction_local": entry["count_local"] / n_local,
			"fraction_global": entry["count_global"] / n_global,
			"lambda": summary_stats(entry["lambda"]),
			"example_point": entry["example_point"],
		})
	summary.sort(key=lambda s: -(s["count_local"] + s["count_global"]))
	periodic = [e for e in attractors if e["period"] is not None]
	separations = []
	for i in range(len(periodic)):
		for j in range(i + 1, len(periodic)):
			if periodic[i]["period"] == periodic[j]["period"]:
				separations.append(float(np.max(np.abs(periodic[i]["signature"] - periodic[j]["signature"]))))
	diagnostics = {
		"n_periodic_attractors": len(periodic),
		"pairwise_orbit_separation": summary_stats(separations) if separations else None,
	}
	return summary, diagnostics


def on_attractor_divergence(a_param, n_dirs, seed):
	"""D_n (n = 0..N_STEPS_DN) for random perturbation directions, starting on the attractor."""
	base = advance(DEFAULT_INITIAL_STATE, a_param, st.N_TRANSIENT)
	rng = np.random.default_rng(seed)
	direction = rng.normal(size=(n_dirs, 9))
	direction /= np.linalg.norm(direction, axis=1, keepdims=True)
	batch_a = np.tile(base, (n_dirs, 1))
	batch_b = batch_a + st.DELTA0 * direction
	dists = np.empty((N_STEPS_DN + 1, n_dirs))
	dists[0] = np.linalg.norm(batch_b - batch_a, axis=1)
	for n in range(1, N_STEPS_DN + 1):
		batch_a = map_step_vec(batch_a, a_param)
		batch_b = map_step_vec(batch_b, a_param)
		dists[n] = np.linalg.norm(batch_b - batch_a, axis=1)
	return dists


def original_protocol(a_param, n_steps):
	"""Manuscript protocol: default state, perturb x1 by delta0, no transient."""
	a_traj = iterate_map_vec(DEFAULT_INITIAL_STATE, a_param, n_steps)
	b0 = DEFAULT_INITIAL_STATE.copy()
	b0[0] += st.DELTA0
	b_traj = iterate_map_vec(b0, a_param, n_steps)
	dist = np.linalg.norm(a_traj - b_traj, axis=1)
	a_long = advance(a_traj[-1], a_param, 3000)
	b_long = advance(b_traj[-1], a_param, 3000)
	a_tail = iterate_map_vec(a_long, a_param, 200)[1:]
	b_tail = iterate_map_vec(b_long, a_param, 200)[1:]
	pa = detect_period(a_tail, tol=1.0e-8)
	pb = detect_period(b_tail, tol=1.0e-8)
	same_orbit = None
	if pa is not None and pb is not None and pa == pb:
		same_orbit = same_orbit_points(orbit_signature(a_tail, pa), orbit_signature(b_tail, pb))
	return dist, {"period_unperturbed": pa, "period_perturbed": pb, "same_orbit": same_orbit, "final_distance": float(dist[-1]), "distance_after_3600": float(np.linalg.norm(a_tail[0] - b_tail[0]))}


def main(quick=False):
	n_local = 20 if quick else 250
	n_global = 20 if quick else 250
	n_dirs = 10 if quick else 100
	out = {}
	out["basin_scan_a52"], out["basin_diagnostics_a52"] = basin_scan(st.A_REGULAR, n_local, n_global, 500 if quick else 3000, st.SEED_IC)
	out["n_attractors_a52"] = len(out["basin_scan_a52"])
	print("a=5.2 basin diagnostics:", out["basin_diagnostics_a52"])
	arrays = {}
	for a_param in (st.A_CHAOTIC, st.A_REGULAR):
		dists = on_attractor_divergence(a_param, n_dirs, st.SEED_PERTURB)
		arrays["on_attractor_a%s" % a_param] = dists
		out["on_attractor_a%s" % a_param] = {
			"median_Dn": np.median(dists, axis=1),
			"p10_Dn": np.percentile(dists, 10, axis=1),
			"p90_Dn": np.percentile(dists, 90, axis=1),
		}
		dist, info = original_protocol(a_param, 600)
		arrays["original_protocol_a%s" % a_param] = dist
		out["original_protocol_a%s" % a_param] = info
		print("a=%s median D_10=%.3e original final D=%.3f info=%s" % (a_param, np.median(dists[10]), info["final_distance"], info))
	print("a=5.2 attractors:", [(s["period"], s["count_local"], s["count_global"], round(s["lambda"].get("median", float("nan")), 3)) for s in out["basin_scan_a52"]])
	name = "a52_multistability_quick" if quick else "a52_multistability"
	np.savez(st.result_path(("sensitivity_quick" if quick else "sensitivity") + ".npz"), **arrays)
	print("->", save_json(name + ".json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
