"""
Full Lyapunov spectra and numerical-reliability studies on the attractor (reviewer items R9, R15).

	spectra            a in A_BAND + A_REGULAR, several initial states, analytic Jacobian
	convergence        running spectrum estimate at a = 0.97 (default state)
	jacobian_accuracy  finite-difference Jacobian error vs step h on attractor points (analytic = exact)
	delta0_study       two-trajectory lambda_max vs initial separation delta0 over the IC ensemble
	spectral_radius    distribution of rho(J) along the attractor

Outputs: results/spectrum.json, results/spectrum.npz

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import advance, map_step_vec, sample_initial_states, DEFAULT_INITIAL_STATE
from jacobian import analytic_jacobian
from lyapunov import lyapunov_spectrum_qr, kaplan_yorke_dimension, numerical_jacobian_vec, two_traj_batch
from analysis_utils import save_json, summary_stats

STEP_SIZES = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
DELTA0_VALUES = [1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12]


def main(quick=False):
	n_transient = 200 if quick else st.N_TRANSIENT
	n_iter = 500 if quick else st.N_ITER_LYAP
	n_spec_ic = 2 if quick else 5
	out = {"n_transient": n_transient, "n_iter": n_iter, "spectra": {}}
	arrays = {}

	ics = sample_initial_states(max(n_spec_ic, 2), st.IC_RADIUS, st.SEED_IC)
	for a_param in st.A_BAND + [st.A_REGULAR]:
		spectra = np.array([lyapunov_spectrum_qr(ics[i], a_param, n_transient, n_iter, jacobian="analytic") for i in range(n_spec_ic)])
		out["spectra"][str(a_param)] = {
			"mean": spectra.mean(axis=0),
			"sd": spectra.std(axis=0, ddof=1),
			"per_ic": spectra,
			"positive_count": summary_stats((spectra > 0).sum(axis=1)),
			"sum": summary_stats(spectra.sum(axis=1)),
			"kaplan_yorke": summary_stats([kaplan_yorke_dimension(s) for s in spectra]),
		}
		print("a=%s spectrum mean=%s KY=%.2f" % (a_param, np.round(spectra.mean(axis=0), 3), out["spectra"][str(a_param)]["kaplan_yorke"]["mean"]))

	spec, history = lyapunov_spectrum_qr(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, n_transient, n_iter, jacobian="analytic", return_history=True)
	arrays["convergence_history"] = history
	out["convergence_final"] = spec

	n_points = 60 if quick else 300
	state = advance(DEFAULT_INITIAL_STATE, st.A_CHAOTIC, n_transient)
	errors = np.empty((n_points, len(STEP_SIZES)))
	rho = np.empty(n_points)
	for p in range(n_points):
		for _ in range(7):
			state = map_step_vec(state, st.A_CHAOTIC)
		exact = analytic_jacobian(state, st.A_CHAOTIC)
		rho[p] = np.max(np.abs(np.linalg.eigvals(exact)))
		norm = np.linalg.norm(exact)
		for h_index, h in enumerate(STEP_SIZES):
			errors[p, h_index] = np.linalg.norm(numerical_jacobian_vec(state, st.A_CHAOTIC, h) - exact) / norm
	out["jacobian_accuracy"] = {repr(h): summary_stats(errors[:, i]) for i, h in enumerate(STEP_SIZES)}
	out["spectral_radius_on_attractor"] = summary_stats(rho)
	arrays["jac_step_sizes"] = np.array(STEP_SIZES)
	arrays["jac_errors"] = errors

	n_ic = 4 if quick else st.N_IC
	ens = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	delta_rows = {}
	delta_matrix = np.empty((len(DELTA0_VALUES), n_ic))
	for d_index, d0 in enumerate(DELTA0_VALUES):
		lam = two_traj_batch(ens, st.A_CHAOTIC, n_transient, 1000 if quick else 5000, d0, perturb_seed=st.SEED_PERTURB)
		delta_matrix[d_index] = lam
		delta_rows[repr(d0)] = summary_stats(lam)
	out["delta0_study"] = delta_rows
	arrays["delta0_values"] = np.array(DELTA0_VALUES)
	arrays["delta0_lambda"] = delta_matrix

	name = "spectrum_quick" if quick else "spectrum"
	np.savez(st.result_path(name + ".npz"), **arrays)
	print("->", save_json(name + ".json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
