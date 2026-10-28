"""
Vectorised replacements for the three slow tests of nist_full_suite.py, algorithmically identical to the
reference implementations (same counting rules, same statistics), plus the full 148-template run.

	aperiodic_templates(m)                all aperiodic templates of length m (148 for m = 9), lexicographic
	non_overlapping_template_test_fast    Section 2.7, greedy non-overlapping count on 9-bit window codes
	overlapping_template_test_fast        Section 2.8, overlapping count of the all-ones window
	linear_complexity_test_fast           Section 2.10, Berlekamp-Massey run on all blocks at once
	run_full_nist_suite_fast              same keys/order as run_full_nist_suite_on_bits; `templates="all"`
	                                      runs the 148 aperiodic templates (188 p-values when excursions apply)
	nist_multi_sequence_assessment        NIST SP 800-22 Sec. 4.2: proportion of passing sequences and
	                                      uniformity of p-values (P-value_T) per p-value type

tab-indented, snake_case, Allman braces.
"""

import itertools
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gammaincc

import nist_full_suite as ref


def aperiodic_templates(m=9):
	"""Templates B of length m with no proper self-overlap (B[k:] != B[:m-k] for all 1 <= k < m)."""
	templates = []
	for bits in itertools.product("01", repeat=m):
		t = "".join(bits)
		if all(t[k:] != t[:m - k] for k in range(1, m)):
			templates.append(t)
	return templates


def _window_codes(block, m):
	weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
	return sliding_window_view(block, m) @ weights


def non_overlapping_template_test_fast(bits, templates=None, m=9, n_blocks=8):
	if templates is None:
		templates = ["000000001", "000000011", "000000101", "111111110", "000010011"]
	n = bits.shape[0]
	block_size = n // n_blocks
	mu = (block_size - m + 1) / (2.0 ** m)
	variance = block_size * (1.0 / (2.0 ** m) - (2.0 * m - 1.0) / (2.0 ** (2 * m)))
	block_codes = [_window_codes(bits[b * block_size:(b + 1) * block_size], m) for b in range(n_blocks)]
	results = {}
	for template_str in templates:
		code = int(template_str, 2)
		counts = np.empty(n_blocks, dtype=np.int64)
		for b, codes in enumerate(block_codes):
			positions = np.flatnonzero(codes == code)
			count = 0
			next_allowed = 0
			for p in positions:
				if p >= next_allowed:
					count += 1
					next_allowed = p + m
			counts[b] = count
		chi_sq = np.sum((counts - mu) ** 2 / variance)
		results[template_str] = gammaincc(n_blocks / 2.0, chi_sq / 2.0)
	return results


def overlapping_template_test_fast(bits, m=9, block_size=1032):
	if m != 9 or block_size != 1032:
		raise ValueError("only the standard m=9, M=1032 parametrization is supported")
	n = bits.shape[0]
	n_blocks = n // block_size
	blocks = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
	codes = sliding_window_view(blocks, m, axis=1) @ (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
	match_counts = np.sum(codes == (2 ** m - 1), axis=1)
	counts = np.bincount(np.minimum(match_counts, 5), minlength=6)
	pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
	chi_sq = np.sum((counts - n_blocks * pi) ** 2 / (n_blocks * pi))
	return gammaincc(5.0 / 2.0, chi_sq / 2.0)


def _berlekamp_massey_blocks(blocks):
	"""Linear complexity of every row of a (K, n) 0/1 array, same update rule as the reference."""
	k_blocks, n = blocks.shape
	seq = blocks.astype(np.uint8)
	c = np.zeros((k_blocks, n), dtype=np.uint8)
	b = np.zeros((k_blocks, n), dtype=np.uint8)
	c[:, 0] = 1
	b[:, 0] = 1
	l_c = np.zeros(k_blocks, dtype=np.int64)
	m_shift = np.full(k_blocks, -1, dtype=np.int64)
	rows = np.arange(k_blocks)[:, None]
	cols = np.arange(n)[None, :]
	for i in range(n):
		if i > 0:
			j = np.arange(1, i + 1)
			window = seq[:, i - j]
			mask = (j[None, :] <= l_c[:, None])
			disc = (seq[:, i] + np.sum(c[:, 1:i + 1] & window & mask, axis=1)) % 2
		else:
			disc = seq[:, 0].astype(np.int64)
		active = disc == 1
		if not active.any():
			continue
		t = c.copy()
		shift = (i - m_shift)[:, None]
		src = cols - shift
		shifted = np.where(src >= 0, b[rows, np.clip(src, 0, n - 1)], 0).astype(np.uint8)
		c = np.where(active[:, None], c ^ shifted, c)
		update = active & (l_c <= i / 2.0)
		l_c = np.where(update, i + 1 - l_c, l_c)
		m_shift = np.where(update, i, m_shift)
		b = np.where(update[:, None], t, b)
	return l_c


def linear_complexity_test_fast(bits, block_size=500):
	n = bits.shape[0]
	n_blocks = n // block_size
	if n_blocks < 200:
		raise ValueError("linear_complexity_test requires at least 200 blocks")
	mu = block_size / 2.0 + (9.0 + (-1.0) ** (block_size + 1)) / 36.0 - (block_size / 3.0 + 2.0 / 9.0) / (2.0 ** block_size)
	pi = np.array([0.010417, 0.031250, 0.125000, 0.500000, 0.250000, 0.062500, 0.020833])
	l_c = _berlekamp_massey_blocks(bits[:n_blocks * block_size].reshape(n_blocks, block_size))
	t_value = ((-1.0) ** block_size) * (l_c - mu) + 2.0 / 9.0
	category = np.digitize(t_value, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], right=True)
	v_counts = np.bincount(category, minlength=7)
	chi_sq = np.sum((v_counts - n_blocks * pi) ** 2 / (n_blocks * pi))
	return gammaincc(3.0, chi_sq / 2.0)


def run_full_nist_suite_fast(bits, alpha=0.01, templates=None):
	"""Same result keys as nist_full_suite.run_full_nist_suite_on_bits; templates='all' uses all 148."""
	bits = np.asarray(bits, dtype=np.int64)
	n = bits.shape[0]
	if templates == "all":
		templates = aperiodic_templates(9)
	results = {}
	results["1_frequency_monobit"] = ref.frequency_monobit_test(bits)
	results["2_block_frequency"] = ref.block_frequency_test(bits, block_size=128)
	results["3_runs"] = ref.runs_test(bits)
	results["4_longest_run_ones"] = ref.longest_run_ones_test(bits)
	results["5_binary_matrix_rank"] = ref.binary_matrix_rank_test(bits)
	results["6_dft_spectral"] = ref.dft_spectral_test(bits)
	for template_str, p_value in non_overlapping_template_test_fast(bits, templates).items():
		results["7_non_overlapping_template_" + template_str] = p_value
	results["8_overlapping_template"] = overlapping_template_test_fast(bits)
	results["9_universal_maurer"] = ref.universal_test(bits)
	results["10_linear_complexity"] = linear_complexity_test_fast(bits)
	serial_p1, serial_p2 = ref.serial_test(bits, m=2)
	results["11_serial_p1"] = serial_p1
	results["11_serial_p2"] = serial_p2
	results["12_approximate_entropy"] = ref.approximate_entropy_test(bits, m=2)
	results["13_cumulative_sums_forward"] = ref.cumulative_sums_test(bits, mode="forward")
	results["13_cumulative_sums_backward"] = ref.cumulative_sums_test(bits, mode="backward")
	excursions = ref.random_excursions_test(bits)
	if excursions is None:
		results["14_random_excursions"] = None
	else:
		for state, p_value in excursions.items():
			results["14_random_excursions_x%+d" % state] = p_value
	variant = ref.random_excursions_variant_test(bits)
	if variant is None:
		results["15_random_excursions_variant"] = None
	else:
		for state, p_value in variant.items():
			results["15_random_excursions_variant_x%+d" % state] = p_value
	verdicts = {k: (None if v is None else bool(v >= alpha)) for k, v in results.items()}
	return results, verdicts, n


def nist_multi_sequence_assessment(per_sequence_results, alpha=0.01):
	"""Section 4.2 assessment over m sequences.

	For each p-value type: m_eff = number of sequences for which it was computed, proportion passing,
	acceptance band (1-alpha) +- 3 sqrt(alpha(1-alpha)/m_eff), and P-value_T from a 10-bin chi-square on
	the p-values (required >= 1e-4; only reported when m_eff >= 55).
	"""
	keys = []
	for res in per_sequence_results:
		for k, v in res.items():
			if v is not None and k not in keys:
				keys.append(k)
	rows = {}
	for key in keys:
		pvals = np.array([res[key] for res in per_sequence_results if res.get(key) is not None], dtype=float)
		m_eff = pvals.size
		proportion = float(np.mean(pvals >= alpha))
		half = 3.0 * np.sqrt(alpha * (1.0 - alpha) / m_eff)
		lower = (1.0 - alpha) - half
		counts = np.histogram(pvals, bins=np.linspace(0.0, 1.0, 11))[0]
		chi_sq = float(np.sum((counts - m_eff / 10.0) ** 2 / (m_eff / 10.0)))
		p_value_t = float(gammaincc(9.0 / 2.0, chi_sq / 2.0))
		rows[key] = {
			"m": int(m_eff),
			"proportion": proportion,
			"proportion_lower_bound": float(lower),
			"proportion_ok": bool(proportion >= lower),
			"p_value_T": p_value_t,
			"uniformity_ok": bool(p_value_t >= 1.0e-4) if m_eff >= 55 else None,
		}
	n_types = len(rows)
	prop_ok = sum(1 for r in rows.values() if r["proportion_ok"])
	unif_rows = [r for r in rows.values() if r["uniformity_ok"] is not None]
	unif_ok = sum(1 for r in unif_rows if r["uniformity_ok"])
	both_ok = sum(1 for r in rows.values() if r["proportion_ok"] and (r["uniformity_ok"] is not False))
	return {
		"n_sequences": len(per_sequence_results),
		"n_pvalue_types": n_types,
		"proportion_ok_types": prop_ok,
		"uniformity_assessed_types": len(unif_rows),
		"uniformity_ok_types": unif_ok,
		"both_ok_types": both_ok,
		"all_ok": bool(both_ok == n_types),
		"per_type": rows,
	}


if __name__ == "__main__":
	import time
	print("aperiodic templates m=9:", len(aperiodic_templates(9)))
	rng = np.random.default_rng(3)
	from gravity_map import iterate_map_vec, advance, DEFAULT_INITIAL_STATE
	from postprocessing import postprocess_bitstream
	traj = iterate_map_vec(advance(DEFAULT_INITIAL_STATE, 0.97, 2000), 0.97, 2 ** 20)[1:]
	samples = {
		"random": rng.integers(0, 2, 2 ** 20).astype(np.int64),
		"map_raw_a097": ref.to_bitstream(traj[:, 0]),
		"map_lsb_a097": postprocess_bitstream(traj, method="lsb_xor3"),
	}
	worst = 0.0
	for name, bits in samples.items():
		t0 = time.perf_counter()
		fast_nov = non_overlapping_template_test_fast(bits)
		fast_ov = overlapping_template_test_fast(bits)
		fast_lc = linear_complexity_test_fast(bits)
		t_fast = time.perf_counter() - t0
		t0 = time.perf_counter()
		ref_nov = ref.non_overlapping_template_test(bits)
		ref_ov = ref.overlapping_template_test(bits)
		ref_lc = ref.linear_complexity_test(bits)
		t_ref = time.perf_counter() - t0
		diffs = [abs(fast_nov[k] - ref_nov[k]) for k in ref_nov] + [abs(fast_ov - ref_ov), abs(fast_lc - ref_lc)]
		worst = max(worst, max(diffs))
		print("%-14s max|p_fast - p_ref| = %.3e   fast %.2fs  reference %.2fs" % (name, max(diffs), t_fast, t_ref))
	t0 = time.perf_counter()
	res, ver, _ = run_full_nist_suite_fast(samples["map_lsb_a097"], templates="all")
	print("full suite with 148 templates: %d p-values, %.2fs" % (sum(v is not None for v in res.values()), time.perf_counter() - t0))
	print("WORST DIFF", worst)
