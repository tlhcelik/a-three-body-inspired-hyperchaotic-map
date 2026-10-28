"""
Sensitivity of the permutation entropy to the tie-rounding precision (reviewer item R14, follow-up).

For every a in NIST_A_VALUES-like order (0.97, 2.0, 6.0, -1.0, 5.2) the x_1 series (N_PE samples after the
transient) of each of the N_IC initial states is rounded to d decimals for d in PE_ROUND_SCAN (and left
unrounded) and the normalised permutation entropy (m = PE_EMBEDDING, stable sort) is computed. The
reference value is the result at PE_ROUND_REFERENCE decimals, chosen a priori between the double-precision
noise of periodic cycles (about 1e-16, far below 1e-12) and the smallest genuine gaps of chaotic series.

The burn-in is PE_ROUND_TRANSIENT iterations (longer than the N_TRANSIENT of the other analyses): at a = 5.2
one of the 20 initial states is still converging to its period-2 cycle after N_TRANSIENT iterations, and the
residual transient (differences of 1e-12 to 1e-8) would be resolved by fine rounding and merged by coarse
rounding, which measures the transient instead of the rounding precision. From 3000 iterations on all 20 states
are on the cycle.

A precision d is called stable when, for every a and every initial state, it reproduces the reference entropy
to within PE_STABILITY_TOL (below the fourth printed decimal); differences in the number of observed patterns
are reported but do not enter the criterion. The stability window is the maximal
contiguous range of scanned precisions around the reference in which every d is stable: `stable_from_decimals`
is its coarsest edge (the smallest rounding precision that no longer changes the result) and
`stable_to_decimals` its finest edge (finer precisions start to split the floating-point noise of periodic
cycles and change the result again).

Output: results/pe_rounding_sensitivity.json

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
from gravity_map import advance, iterate_map_vec, map_step_vec, sample_initial_states, DEFAULT_INITIAL_STATE
from analysis_utils import save_json, permutation_entropy_vec

A_VALUES = [0.97, 2.0, 6.0, -1.0, 5.2]


def scan_comparison(quick):
	"""Entropy scan of the manuscript figure (same grid, series length and burn-in as figures_revision.fig_permutation_entropy)
	at the adopted precision versus the reference precision. The burn-in must be long (PE_ROUND_TRANSIENT): with the
	500 iterations of the Lyapunov scans, series near periodic windows still carry a residual transient, which fine rounding
	resolves and coarse rounding merges, so the two precisions differ there (and the entropy is overestimated)."""
	a_values = np.round(np.arange(-3.0, 6.0 + 1e-9, 0.05), 10)
	n_series = 5000 if quick else 50000
	burn_in = 500 if quick else st.PE_ROUND_TRANSIENT
	states = advance(np.tile(DEFAULT_INITIAL_STATE, (a_values.size, 1)), a_values, burn_in)
	series = np.empty((n_series, a_values.size))
	for n in range(n_series):
		states = map_step_vec(states, a_values)
		series[n] = states[:, 0]
	adopted = np.array([entropy_row(series[:, i], st.PE_ROUND_DECIMALS)[0] for i in range(a_values.size)])
	reference = np.array([entropy_row(series[:, i], st.PE_ROUND_REFERENCE)[0] for i in range(a_values.size)])
	diff = np.abs(adopted - reference)
	worst = int(np.argmax(diff))
	return {
		"n_a_values": int(a_values.size),
		"n_series": n_series,
		"burn_in": burn_in,
		"adopted_decimals": st.PE_ROUND_DECIMALS,
		"reference_decimals": st.PE_ROUND_REFERENCE,
		"n_above_tolerance": int((diff > st.PE_STABILITY_TOL).sum()),
		"max_abs_entropy_difference": float(diff.max()),
		"a_at_max_difference": float(a_values[worst]),
	}


def entropy_row(series, decimals):
	"""Normalised entropy and pattern count of one series at one rounding precision (None = unrounded)."""
	res = permutation_entropy_vec(series, st.PE_EMBEDDING, round_decimals=decimals)
	return float(res["normalized"]), int(res["patterns_observed"])


def main(quick=False):
	n_ic = 3 if quick else st.N_IC
	n_pe = 20000 if quick else st.N_PE
	n_transient = 200 if quick else st.PE_ROUND_TRANSIENT
	decimals = list(st.PE_ROUND_SCAN) + [None]
	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)

	per_a = {}
	for a in A_VALUES:
		start = advance(ics, a, n_transient)
		series = iterate_map_vec(start, a, n_pe)[1:, :, 0]
		values = np.empty((len(decimals), n_ic))
		patterns = np.empty((len(decimals), n_ic), dtype=int)
		for j, d in enumerate(decimals):
			for i in range(n_ic):
				values[j, i], patterns[j, i] = entropy_row(series[:, i], d)
		ref_row = decimals.index(st.PE_ROUND_REFERENCE)
		per_a[a] = {"values": values, "patterns": patterns, "ref_row": ref_row}
		print("a=%-5s " % a + "  ".join("%s:%.4f/%d" % ("raw" if d is None else d, values[j, 0], patterns[j, 0]) for j, d in enumerate(decimals)), flush=True)

	table = {}
	deviation = {}
	for a, entry in per_a.items():
		values, patterns, ref_row = entry["values"], entry["patterns"], entry["ref_row"]
		table[str(a)] = {}
		deviation[str(a)] = {}
		for j, d in enumerate(decimals):
			key = "raw" if d is None else str(d)
			table[str(a)][key] = {
				"entropy_reference_ic": float(values[j, 0]),
				"patterns_reference_ic": int(patterns[j, 0]),
				"entropy_mean": float(values[j].mean()),
				"entropy_sd": float(values[j].std(ddof=1)) if n_ic > 1 else 0.0,
				"patterns_min": int(patterns[j].min()),
				"patterns_max": int(patterns[j].max()),
			}
			deviation[str(a)][key] = {
				"max_abs_entropy_difference": float(np.abs(values[j] - values[ref_row]).max()),
				"max_abs_pattern_difference": int(np.abs(patterns[j] - patterns[ref_row]).max()),
			}

	def is_stable(d):
		key = str(d)
		return all(deviation[str(a)][key]["max_abs_entropy_difference"] < st.PE_STABILITY_TOL for a in A_VALUES)

	scan = list(st.PE_ROUND_SCAN)
	ref_index = scan.index(st.PE_ROUND_REFERENCE)
	low = ref_index
	while low > 0 and is_stable(scan[low - 1]):
		low -= 1
	high = ref_index
	while high < len(scan) - 1 and is_stable(scan[high + 1]):
		high += 1
	stable_from = scan[low]
	stable_to = scan[high]
	overall = {}
	for d in scan + ["raw"]:
		key = str(d)
		overall[key] = {
			"max_abs_entropy_difference": max(deviation[str(a)][key]["max_abs_entropy_difference"] for a in A_VALUES),
			"max_abs_pattern_difference": max(deviation[str(a)][key]["max_abs_pattern_difference"] for a in A_VALUES),
		}
	print("stability window (decimals): %s .. %s" % (stable_from, stable_to))
	out = {
		"embedding": st.PE_EMBEDDING,
		"n_series": n_pe,
		"n_transient": n_transient,
		"n_ic": n_ic,
		"a_values": A_VALUES,
		"decimals": [("raw" if d is None else d) for d in decimals],
		"reference_decimals": st.PE_ROUND_REFERENCE,
		"tolerance": st.PE_STABILITY_TOL,
		"stable_from_decimals": stable_from,
		"stable_to_decimals": stable_to,
		"table": table,
		"deviation_from_reference": deviation,
		"max_deviation_over_a": overall,
		"scan_comparison": scan_comparison(quick),
	}
	print("scan comparison:", out["scan_comparison"], flush=True)
	save_json("pe_rounding_sensitivity_quick.json" if quick else "pe_rounding_sensitivity.json", out)


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
