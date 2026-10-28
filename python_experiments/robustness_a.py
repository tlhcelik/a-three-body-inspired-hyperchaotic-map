"""
Robustness of lambda_max at the parameter values used as the "chaotic band" in the NIST section
(reviewer item R20): a in A_BAND = [-1.0, 0.97, 2.0, 6.0].

	ensemble        N_IC initial states, N_TRANSIENT + N_ITER_LYAP, random tangent directions
	length_study    the same ensemble for measurement lengths 1500, 5000, 20000, 50000
	regimes         period detection on the tail of every trajectory

Output: results/robustness_a.json

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import advance, iterate_map_vec, sample_initial_states
from lyapunov import two_traj_batch
from analysis_utils import save_json, summary_stats, detect_period


def main(quick=False):
	n_ic = 4 if quick else st.N_IC
	n_transient = 200 if quick else st.N_TRANSIENT
	lengths = [300, 1000] if quick else [1500, 5000, 20000, 50000]
	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	out = {"n_ic": n_ic, "n_transient": n_transient, "points": {}}
	for a_param in st.A_BAND:
		entry = {"length_study": {}}
		for n_iter in lengths:
			lam = two_traj_batch(ics, a_param, n_transient, n_iter, st.DELTA0, perturb_seed=st.SEED_PERTURB)
			entry["length_study"][str(n_iter)] = summary_stats(lam)
		start = advance(ics, a_param, n_transient + lengths[-1])
		tail = iterate_map_vec(start, a_param, 300)[1:]
		periods = [detect_period(tail[:, i, :], tol=1.0e-8) for i in range(n_ic)]
		entry["periods_per_ic"] = periods
		entry["aperiodic_fraction"] = float(np.mean([p is None for p in periods]))
		out["points"][str(a_param)] = entry
		print("a=%5.2f " % a_param + " ".join("N=%s:%.3f+-%.3f" % (k, v["mean"], v["sd"]) for k, v in entry["length_study"].items()) + " aperiodic=%.2f" % entry["aperiodic_fraction"])
	print("->", save_json("robustness_a_quick.json" if quick else "robustness_a.json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
