"""
Canonical reference values at a = 0.97 (chaotic) and a = 5.2 (regular) over an ensemble of
initial states (reviewer items R7, R9, R14, R28). Every number in tab:summary_results and
tab:zero_one_vs_lyapunov is taken from this file.

Per a:
	lambda_max (two-trajectory, random direction)           N_IC states, N_TRANSIENT + N_ITER_LYAP
	full spectrum, sum, Kaplan-Yorke dimension (QR, analytic Jacobian)
	0-1 test K (standard Gottwald-Melbourne layout)         20000-sample x1 series per state
	permutation entropy m=5 and m=7, tie policy = round to PE_ROUND_DECIMALS; unrounded m=5 kept for comparison
	Shannon entropy (256 bins) of x1
	correlation integral / D2 plateau for projections of dimension 3, 6 and 9 (10^4 points)

Outputs: results/canonical_reference.json, results/canonical_reference_corr.npz

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import advance, iterate_map_vec, sample_initial_states
from lyapunov import two_traj_batch, lyapunov_spectrum_qr, kaplan_yorke_dimension
from analysis_utils import save_json, summary_stats, zero_one_test_gm, permutation_entropy_vec, shannon_entropy_bits, correlation_integral, plateau_estimate, detect_period


def reference_point(a_param, quick):
	n_ic = 3 if quick else st.N_IC
	n_transient = 200 if quick else st.N_TRANSIENT
	n_lyap = 500 if quick else st.N_ITER_LYAP
	n_series = 4000 if quick else st.N_SERIES
	n_pe = 20000 if quick else 200000
	n_corr = 1500 if quick else st.CORR_N_POINTS
	n_spec_ic = 2 if quick else 5

	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	result = {"a": a_param, "n_ic": n_ic}

	lam = two_traj_batch(ics, a_param, n_transient, n_lyap, st.DELTA0, perturb_seed=st.SEED_PERTURB)
	result["lambda_two_traj"] = summary_stats(lam)
	result["lambda_two_traj_per_ic"] = lam

	spectra = np.array([lyapunov_spectrum_qr(ics[i], a_param, n_transient, n_lyap, jacobian="analytic") for i in range(n_spec_ic)])
	result["spectrum_mean"] = spectra.mean(axis=0)
	result["spectrum_sd"] = spectra.std(axis=0, ddof=1) if n_spec_ic > 1 else np.zeros(9)
	result["lambda_qr"] = summary_stats(spectra[:, 0])
	result["spectrum_sum"] = summary_stats(spectra.sum(axis=1))
	result["positive_exponents"] = summary_stats((spectra > 0.0).sum(axis=1))
	result["kaplan_yorke"] = summary_stats([kaplan_yorke_dimension(s) for s in spectra])

	start = advance(ics, a_param, n_transient)
	series_batch = iterate_map_vec(start, a_param, n_series)[1:, :, :]
	k_values = []
	periods = []
	shannon = []
	for i in range(n_ic):
		x1 = series_batch[:, i, 0]
		k_values.append(zero_one_test_gm(x1, n_c=st.ZERO_ONE_N_C, seed=st.SEED_ZERO_ONE)[0])
		periods.append(detect_period(series_batch[-300:, i, :]))
		shannon.append(shannon_entropy_bits(x1, st.SHANNON_BINS)[0])
	result["zero_one_K"] = summary_stats(k_values)
	result["period_per_ic"] = periods
	result["shannon_bits_x1"] = summary_stats(shannon)
	result["shannon_normalized_x1"] = summary_stats(np.array(shannon) / 8.0)

	long_series = iterate_map_vec(start[0], a_param, n_pe)[1:, 0]
	result["permutation_entropy"] = {
		"m5_rounded": permutation_entropy_vec(long_series, 5, round_decimals=st.PE_ROUND_DECIMALS),
		"m5_unrounded": permutation_entropy_vec(long_series, 5, round_decimals=None),
		"m7_rounded": permutation_entropy_vec(long_series, 7, round_decimals=st.PE_ROUND_DECIMALS),
	}

	traj = iterate_map_vec(start[0], a_param, n_corr * 10, record_every=10)[1:]
	corr = {}
	arrays = {}
	for label, cols in (("3d_body1", slice(0, 3)), ("6d_bodies12", slice(0, 6)), ("9d_full", slice(0, 9))):
		ci = correlation_integral(traj[:, cols], seed=st.SEED_CORR)
		corr[label] = plateau_estimate(ci["local_slopes"], ci["c"])
		corr[label]["slope_range"] = [float(np.min(ci["local_slopes"])), float(np.max(ci["local_slopes"]))] if ci["local_slopes"].size else None
		arrays[label + "_log_r"] = ci["log_r"]
		arrays[label + "_log_c"] = ci["log_c"]
		arrays[label + "_slopes"] = ci["local_slopes"]
	corr["n_points"] = int(traj.shape[0])
	corr["sampling"] = "every 10th iterate after transient"
	result["correlation_dimension"] = corr
	return result, arrays


def main(quick=False):
	out = {}
	arrays = {}
	for a_param in (st.A_CHAOTIC, st.A_REGULAR):
		res, arr = reference_point(a_param, quick)
		out[str(a_param)] = res
		arrays.update({"a%s_%s" % (a_param, k): v for k, v in arr.items()})
		print("a=%s lambda=%.4f+-%.4f qr=%.4f K=%.4f PE7=%.4f D2(9d)=%s KY=%.2f" % (a_param, res["lambda_two_traj"]["mean"], res["lambda_two_traj"]["sd"], res["lambda_qr"]["mean"], res["zero_one_K"]["median"], res["permutation_entropy"]["m7_rounded"]["normalized"], res["correlation_dimension"]["9d_full"]["d2"], res["kaplan_yorke"]["mean"]))
	name = "canonical_reference_quick" if quick else "canonical_reference"
	np.savez(st.result_path(name + "_corr.npz"), **arrays)
	print("->", save_json(name + ".json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
