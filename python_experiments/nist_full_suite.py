"""
Full NIST SP 800-22 Rev. 1a statistical test suite (Rukhin et al., 2010),
implemented from the official test definitions (Sections 2.1-2.15).

Implements all 15 tests:
	1.  Frequency (Monobit) Test
	2.  Frequency Test within a Block
	3.  Runs Test
	4.  Test for the Longest Run of Ones in a Block
	5.  Binary Matrix Rank Test
	6.  Discrete Fourier Transform (Spectral) Test
	7.  Non-overlapping Template Matching Test
	8.  Overlapping Template Matching Test
	9.  Maurer's Universal Statistical Test
	10. Linear Complexity Test
	11. Serial Test (two p-values)
	12. Approximate Entropy Test
	13. Cumulative Sums Test (forward and backward, two p-values)
	14. Random Excursions Test (up to 8 p-values, one per state)
	15. Random Excursions Variant Test (up to 18 p-values, one per state)

tab-indented, snake_case, Allman braces, no line-continuation splitting.
"""

import numpy as np
from scipy.special import gammaincc, erfc


def to_bitstream(time_series):
	"""Binarize a real-valued time series around its median."""
	median_value = np.median(time_series)
	bits = (time_series >= median_value).astype(np.int64)
	return bits


# ---------------------------------------------------------------------------
# 1-3. Already-validated tests (kept identical to nist_tests.py for continuity)
# ---------------------------------------------------------------------------

def frequency_monobit_test(bits):
	"""Section 2.1: Frequency (Monobit) Test."""
	n = bits.shape[0]
	s_n = np.sum(2 * bits - 1)
	s_obs = np.abs(s_n) / np.sqrt(n)
	p_value = erfc(s_obs / np.sqrt(2.0))
	return p_value


def block_frequency_test(bits, block_size=128):
	"""Section 2.2: Frequency Test within a Block."""
	n = bits.shape[0]
	n_blocks = n // block_size
	blocks = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
	proportions = np.mean(blocks, axis=1)
	chi_sq = 4.0 * block_size * np.sum((proportions - 0.5) ** 2)
	p_value = gammaincc(n_blocks / 2.0, chi_sq / 2.0)
	return p_value


def runs_test(bits):
	"""Section 2.3: Runs Test."""
	n = bits.shape[0]
	pi_hat = np.mean(bits)
	if abs(pi_hat - 0.5) >= (2.0 / np.sqrt(n)):
		return 0.0
	v_n = 1 + np.sum(bits[1:] != bits[:-1])
	numerator = abs(v_n - 2 * n * pi_hat * (1 - pi_hat))
	denominator = 2.0 * np.sqrt(2.0 * n) * pi_hat * (1 - pi_hat)
	p_value = erfc(numerator / denominator)
	return p_value


# ---------------------------------------------------------------------------
# 4. Longest Run of Ones in a Block
# ---------------------------------------------------------------------------

def longest_run_ones_test(bits):
	"""Section 2.4: Test for the Longest Run of Ones in a Block."""
	n = bits.shape[0]
	if n < 128:
		raise ValueError("longest_run_ones_test requires n >= 128")

	if n < 6272:
		block_size = 8
		k_df = 3
		v_categories = 4
		pi = np.array([0.2148, 0.3672, 0.2305, 0.1875])
		edges = (1, 2, 3)
	elif n < 750000:
		block_size = 128
		k_df = 5
		v_categories = 6
		pi = np.array([0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])
		edges = (4, 5, 6, 7, 8)
	else:
		block_size = 10000
		k_df = 6
		v_categories = 7
		pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
		edges = (10, 11, 12, 13, 14, 15)

	n_blocks = n // block_size
	blocks = bits[: n_blocks * block_size].reshape(n_blocks, block_size)

	longest_runs = np.empty(n_blocks, dtype=np.int64)
	for row_index in range(n_blocks):
		row = blocks[row_index]
		max_run = 0
		current_run = 0
		for bit_value in row:
			if bit_value == 1:
				current_run += 1
				max_run = max(max_run, current_run)
			else:
				current_run = 0
		longest_runs[row_index] = max_run

	v_counts = np.zeros(v_categories, dtype=np.int64)
	for run_length in longest_runs:
		category = 0
		for edge in edges:
			if run_length > edge:
				category += 1
			else:
				break
		v_counts[category] += 1

	chi_sq = np.sum((v_counts - n_blocks * pi) ** 2 / (n_blocks * pi))
	p_value = gammaincc(k_df / 2.0, chi_sq / 2.0)
	return p_value


# ---------------------------------------------------------------------------
# 5. Binary Matrix Rank Test
# ---------------------------------------------------------------------------

def _gf2_rank(matrix):
	"""Rank of a binary matrix over GF(2) via Gaussian elimination (XOR)."""
	m = matrix.copy().astype(np.int64)
	n_rows, n_cols = m.shape
	rank = 0
	pivot_col = 0
	for pivot_col in range(n_cols):
		pivot_row = None
		for row_index in range(rank, n_rows):
			if m[row_index, pivot_col] == 1:
				pivot_row = row_index
				break
		if pivot_row is None:
			continue
		m[[rank, pivot_row]] = m[[pivot_row, rank]]
		for row_index in range(n_rows):
			if row_index != rank and m[row_index, pivot_col] == 1:
				m[row_index] = m[row_index] ^ m[rank]
		rank += 1
		if rank == n_rows:
			break
	return rank


def binary_matrix_rank_test(bits, matrix_size=32):
	"""Section 2.5: Binary Matrix Rank Test."""
	n = bits.shape[0]
	block_bits = matrix_size * matrix_size
	n_blocks = n // block_bits
	if n_blocks < 38:
		raise ValueError("binary_matrix_rank_test requires at least 38 matrices")

	full_rank_count = 0
	full_minus_one_count = 0
	for block_index in range(n_blocks):
		chunk = bits[block_index * block_bits: (block_index + 1) * block_bits]
		matrix = chunk.reshape(matrix_size, matrix_size)
		rank = _gf2_rank(matrix)
		if rank == matrix_size:
			full_rank_count += 1
		elif rank == matrix_size - 1:
			full_minus_one_count += 1

	p_full = 0.2888
	p_full_minus_one = 0.5776
	p_rest = 0.1336
	rest_count = n_blocks - full_rank_count - full_minus_one_count

	chi_sq = (
		(full_rank_count - p_full * n_blocks) ** 2 / (p_full * n_blocks)
		+ (full_minus_one_count - p_full_minus_one * n_blocks) ** 2 / (p_full_minus_one * n_blocks)
		+ (rest_count - p_rest * n_blocks) ** 2 / (p_rest * n_blocks)
	)
	p_value = np.exp(-chi_sq / 2.0)
	return p_value


# ---------------------------------------------------------------------------
# 6. Discrete Fourier Transform (Spectral) Test
# ---------------------------------------------------------------------------

def dft_spectral_test(bits):
	"""Section 2.6: Discrete Fourier Transform (Spectral) Test."""
	n = bits.shape[0]
	x = 2 * bits.astype(np.float64) - 1
	fourier = np.fft.fft(x)
	magnitudes = np.abs(fourier[: n // 2])

	threshold = np.sqrt(np.log(1.0 / 0.05) * n)
	n0 = 0.95 * n / 2.0
	n1 = np.sum(magnitudes < threshold)
	d_stat = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
	p_value = erfc(np.abs(d_stat) / np.sqrt(2.0))
	return p_value


# ---------------------------------------------------------------------------
# 7. Non-overlapping Template Matching Test
# ---------------------------------------------------------------------------

def non_overlapping_template_test(bits, templates=None, m=9, n_blocks=8):
	"""Section 2.7: Non-overlapping Template Matching Test.
	Returns a dict {template_string: p_value} for each tested template."""
	if templates is None:
		templates = ["000000001", "000000011", "000000101", "111111110", "000010011"]

	n = bits.shape[0]
	block_size = n // n_blocks
	mu = (block_size - m + 1) / (2.0 ** m)
	variance = block_size * (1.0 / (2.0 ** m) - (2.0 * m - 1.0) / (2.0 ** (2 * m)))

	results = {}
	for template_str in templates:
		template = np.array([int(ch) for ch in template_str])
		counts = np.empty(n_blocks, dtype=np.int64)
		for block_index in range(n_blocks):
			block = bits[block_index * block_size: (block_index + 1) * block_size]
			match_count = 0
			position = 0
			limit = block_size - m
			while position <= limit:
				if np.array_equal(block[position: position + m], template):
					match_count += 1
					position += m
				else:
					position += 1
			counts[block_index] = match_count

		chi_sq = np.sum((counts - mu) ** 2 / variance)
		p_value = gammaincc(n_blocks / 2.0, chi_sq / 2.0)
		results[template_str] = p_value

	return results


# ---------------------------------------------------------------------------
# 8. Overlapping Template Matching Test
# ---------------------------------------------------------------------------

def overlapping_template_test(bits, m=9, block_size=1032):
	"""Section 2.8: Overlapping Template Matching Test (template = 1^m).
	Uses the official NIST SP 800-22 reference pi-table for the standard
	(m=9, M=1032) parametrization, since the renewal-theory correction for
	overlapping self-matches of an all-ones template has no simple closed
	form and NIST instead tabulates these values directly (Section 2.8.4)."""
	n = bits.shape[0]
	n_blocks = n // block_size
	template = np.ones(m, dtype=np.int64)

	if m != 9 or block_size != 1032:
		raise ValueError("overlapping_template_test: only the standard m=9, M=1032 parametrization is supported")

	counts = np.zeros(6, dtype=np.int64)
	for block_index in range(n_blocks):
		block = bits[block_index * block_size: (block_index + 1) * block_size]
		match_count = 0
		for position in range(block_size - m + 1):
			if np.array_equal(block[position: position + m], template):
				match_count += 1
		category = min(match_count, 5)
		counts[category] += 1

	pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])

	chi_sq = np.sum((counts - n_blocks * pi) ** 2 / (n_blocks * pi))
	p_value = gammaincc(5.0 / 2.0, chi_sq / 2.0)
	return p_value


# ---------------------------------------------------------------------------
# 9. Maurer's Universal Statistical Test
# ---------------------------------------------------------------------------

_UNIVERSAL_EXPECTED_VARIANCE = {
	6: (5.2177052, 2.954),
	7: (6.1962507, 3.125),
	8: (7.1836656, 3.238),
	9: (8.1764248, 3.311),
	10: (9.1723243, 3.356),
}


def universal_test(bits, block_length=None):
	"""Section 2.9: Maurer's Universal Statistical Test."""
	n = bits.shape[0]

	if block_length is None:
		if n >= 904960:
			block_length = 7
		elif n >= 387840:
			block_length = 6
		else:
			raise ValueError("universal_test requires n >= 387840")

	l_val = block_length
	q_blocks = 10 * (2 ** l_val)
	total_blocks = n // l_val
	k_blocks = total_blocks - q_blocks
	if k_blocks <= 0:
		raise ValueError("universal_test: not enough data for chosen block_length")

	powers_of_two = 2 ** np.arange(l_val)[::-1]
	symbols = bits[: total_blocks * l_val].reshape(total_blocks, l_val) @ powers_of_two

	table = np.zeros(2 ** l_val, dtype=np.int64)
	for block_index in range(q_blocks):
		table[symbols[block_index]] = block_index + 1

	log_sum = 0.0
	for block_index in range(q_blocks, total_blocks):
		symbol = symbols[block_index]
		position = block_index + 1
		last_seen = table[symbol]
		gap = position - last_seen if last_seen > 0 else position
		log_sum += np.log2(gap)
		table[symbol] = position

	fn = log_sum / k_blocks
	expected_value, variance = _UNIVERSAL_EXPECTED_VARIANCE[l_val]

	c_factor = 0.7 - 0.8 / l_val + (4.0 + 32.0 / l_val) * (k_blocks ** (-3.0 / l_val)) / 15.0
	sigma = c_factor * np.sqrt(variance / k_blocks)
	statistic = np.abs(fn - expected_value) / (np.sqrt(2.0) * sigma)
	p_value = erfc(statistic)
	return p_value


# ---------------------------------------------------------------------------
# 10. Linear Complexity Test
# ---------------------------------------------------------------------------

def _berlekamp_massey_gf2(sequence):
	"""Linear complexity of a binary sequence over GF(2) (Berlekamp-Massey)."""
	n = sequence.shape[0]
	c = np.zeros(n, dtype=np.int64)
	b = np.zeros(n, dtype=np.int64)
	c[0] = 1
	b[0] = 1
	l_complexity = 0
	m_shift = -1
	for i in range(n):
		discrepancy = sequence[i]
		for j in range(1, l_complexity + 1):
			discrepancy ^= c[j] & sequence[i - j]
		if discrepancy == 1:
			t = c.copy()
			shift = i - m_shift
			c[shift: shift + n - shift] ^= b[: n - shift]
			if l_complexity <= i / 2.0:
				l_complexity = i + 1 - l_complexity
				m_shift = i
				b = t
	return l_complexity


def linear_complexity_test(bits, block_size=500):
	"""Section 2.10: Linear Complexity Test."""
	n = bits.shape[0]
	n_blocks = n // block_size
	if n_blocks < 200:
		raise ValueError("linear_complexity_test requires at least 200 blocks")

	mu = (
		block_size / 2.0
		+ (9.0 + (-1.0) ** (block_size + 1)) / 36.0
		- (block_size / 3.0 + 2.0 / 9.0) / (2.0 ** block_size)
	)

	pi = np.array([0.010417, 0.031250, 0.125000, 0.500000, 0.250000, 0.062500, 0.020833])
	v_counts = np.zeros(7, dtype=np.int64)

	for block_index in range(n_blocks):
		block = bits[block_index * block_size: (block_index + 1) * block_size]
		l_complexity = _berlekamp_massey_gf2(block)
		t_value = ((-1.0) ** block_size) * (l_complexity - mu) + 2.0 / 9.0
		if t_value <= -2.5:
			category = 0
		elif t_value <= -1.5:
			category = 1
		elif t_value <= -0.5:
			category = 2
		elif t_value <= 0.5:
			category = 3
		elif t_value <= 1.5:
			category = 4
		elif t_value <= 2.5:
			category = 5
		else:
			category = 6
		v_counts[category] += 1

	chi_sq = np.sum((v_counts - n_blocks * pi) ** 2 / (n_blocks * pi))
	p_value = gammaincc(3.0, chi_sq / 2.0)
	return p_value


# ---------------------------------------------------------------------------
# 11. Serial Test
# ---------------------------------------------------------------------------

def _pattern_frequencies(bits, pattern_length):
	"""Overlapping (circular) pattern-frequency count for a given block length."""
	if pattern_length <= 0:
		return None
	n = bits.shape[0]
	extended = np.concatenate([bits, bits[: pattern_length - 1]])
	powers_of_two = 2 ** np.arange(pattern_length)[::-1]
	windows = np.lib.stride_tricks.sliding_window_view(extended, pattern_length)[:n]
	symbols = windows @ powers_of_two
	counts = np.bincount(symbols, minlength=2 ** pattern_length)
	return counts


def serial_test(bits, m=2):
	"""Section 2.11: Serial Test. Returns (p_value_1, p_value_2)."""
	n = bits.shape[0]

	def psi_sq(pattern_length):
		if pattern_length <= 0:
			return 0.0
		counts = _pattern_frequencies(bits, pattern_length)
		return (2.0 ** pattern_length / n) * np.sum(counts.astype(np.float64) ** 2) - n

	psi_m = psi_sq(m)
	psi_m1 = psi_sq(m - 1)
	psi_m2 = psi_sq(m - 2)

	delta_psi = psi_m - psi_m1
	delta2_psi = psi_m - 2.0 * psi_m1 + psi_m2

	p_value_1 = gammaincc(2.0 ** (m - 2), delta_psi / 2.0)
	p_value_2 = gammaincc(2.0 ** (m - 3), delta2_psi / 2.0)
	return p_value_1, p_value_2


# ---------------------------------------------------------------------------
# 12. Approximate Entropy Test
# ---------------------------------------------------------------------------

def approximate_entropy_test(bits, m=2):
	"""Section 2.12: Approximate Entropy Test."""
	n = bits.shape[0]

	def phi(pattern_length):
		counts = _pattern_frequencies(bits, pattern_length)
		probabilities = counts.astype(np.float64) / n
		nonzero = probabilities[probabilities > 0]
		return np.sum(nonzero * np.log(nonzero))

	phi_m = phi(m)
	phi_m1 = phi(m + 1)
	apen = phi_m - phi_m1

	chi_sq = 2.0 * n * (np.log(2.0) - apen)
	p_value = gammaincc(2.0 ** (m - 1), chi_sq / 2.0)
	return p_value


# ---------------------------------------------------------------------------
# 13. Cumulative Sums Test (forward and backward)
# ---------------------------------------------------------------------------

def cumulative_sums_test(bits, mode="forward"):
	"""Section 2.13: Cumulative Sums (Cusum) Test."""
	x = 2 * bits.astype(np.float64) - 1
	if mode == "backward":
		x = x[::-1]
	n = x.shape[0]
	cumulative = np.cumsum(x)
	z_max = np.max(np.abs(cumulative))
	if z_max == 0:
		return 1.0

	term1 = 0.0
	k_start = int(np.floor((-n / z_max + 1) / 4.0))
	k_end = int(np.floor((n / z_max - 1) / 4.0))
	for k_index in range(k_start, k_end + 1):
		from scipy.stats import norm
		term1 += norm.cdf(((4 * k_index + 1) * z_max) / np.sqrt(n)) - norm.cdf(((4 * k_index - 1) * z_max) / np.sqrt(n))

	term2 = 0.0
	k_start2 = int(np.floor((-n / z_max - 3) / 4.0))
	k_end2 = int(np.floor((n / z_max - 1) / 4.0))
	for k_index in range(k_start2, k_end2 + 1):
		from scipy.stats import norm
		term2 += norm.cdf(((4 * k_index + 3) * z_max) / np.sqrt(n)) - norm.cdf(((4 * k_index + 1) * z_max) / np.sqrt(n))

	p_value = 1.0 - term1 + term2
	p_value = float(np.clip(p_value, 0.0, 1.0))
	return p_value


# ---------------------------------------------------------------------------
# 14 & 15. Random Excursions Test and Random Excursions Variant Test
# ---------------------------------------------------------------------------

def _random_walk_cycles(bits):
	"""Return the zero-delimited cumulative-sum random walk and its cycle boundaries."""
	x = 2 * bits.astype(np.int64) - 1
	s = np.concatenate([[0], np.cumsum(x), [0]])
	zero_positions = np.where(s == 0)[0]
	return s, zero_positions


def random_excursions_test(bits, min_cycles=500):
	"""Section 2.14: Random Excursions Test. Returns dict {state: p_value} or None."""
	s, zero_positions = _random_walk_cycles(bits)
	n_cycles = len(zero_positions) - 1
	if n_cycles < min_cycles:
		return None

	states = [-4, -3, -2, -1, 1, 2, 3, 4]
	v_counts = {state: np.zeros(6, dtype=np.int64) for state in states}

	for cycle_index in range(n_cycles):
		start = zero_positions[cycle_index]
		end = zero_positions[cycle_index + 1]
		cycle_values = s[start + 1: end]
		for state in states:
			visits = np.sum(cycle_values == state)
			category = min(visits, 5)
			v_counts[state][category] += 1

	results = {}
	for state in states:
		abs_x = abs(state)
		pi = np.empty(6)
		pi[0] = 1.0 - 1.0 / (2.0 * abs_x)
		for k in range(1, 5):
			pi[k] = (1.0 / (4.0 * abs_x ** 2)) * (1.0 - 1.0 / (2.0 * abs_x)) ** (k - 1)
		pi[5] = (1.0 / (2.0 * abs_x)) * (1.0 - 1.0 / (2.0 * abs_x)) ** 4

		counts = v_counts[state]
		chi_sq = np.sum((counts - n_cycles * pi) ** 2 / (n_cycles * pi))
		results[state] = gammaincc(5.0 / 2.0, chi_sq / 2.0)

	return results


def random_excursions_variant_test(bits, min_cycles=500):
	"""Section 2.15: Random Excursions Variant Test. Returns dict {state: p_value} or None."""
	s, zero_positions = _random_walk_cycles(bits)
	n_cycles = len(zero_positions) - 1
	if n_cycles < min_cycles:
		return None

	interior = s[1:-1]
	states = list(range(-9, 0)) + list(range(1, 10))
	results = {}
	for state in states:
		xi = np.sum(interior == state)
		denom = np.sqrt(2.0 * n_cycles * (4.0 * abs(state) - 2.0))
		p_value = erfc(np.abs(xi - n_cycles) / denom)
		results[state] = p_value

	return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_nist_suite(time_series, alpha=0.01):
	"""Run all 15 NIST SP 800-22 tests on a scalar time series (median-binarized)."""
	bits = to_bitstream(time_series)
	return run_full_nist_suite_on_bits(bits, alpha=alpha)


def run_full_nist_suite_on_bits(bits, alpha=0.01):
	"""Run all 15 NIST SP 800-22 tests directly on a precomputed 0/1 bit array."""
	bits = np.asarray(bits, dtype=np.int64)
	n = bits.shape[0]

	results = {}
	results["1_frequency_monobit"] = frequency_monobit_test(bits)
	results["2_block_frequency"] = block_frequency_test(bits, block_size=128)
	results["3_runs"] = runs_test(bits)
	results["4_longest_run_ones"] = longest_run_ones_test(bits)
	results["5_binary_matrix_rank"] = binary_matrix_rank_test(bits)
	results["6_dft_spectral"] = dft_spectral_test(bits)

	non_overlap = non_overlapping_template_test(bits)
	for template_str, p_value in non_overlap.items():
		results[f"7_non_overlapping_template_{template_str}"] = p_value

	results["8_overlapping_template"] = overlapping_template_test(bits)
	results["9_universal_maurer"] = universal_test(bits)
	results["10_linear_complexity"] = linear_complexity_test(bits)

	serial_p1, serial_p2 = serial_test(bits, m=2)
	results["11_serial_p1"] = serial_p1
	results["11_serial_p2"] = serial_p2

	results["12_approximate_entropy"] = approximate_entropy_test(bits, m=2)

	results["13_cumulative_sums_forward"] = cumulative_sums_test(bits, mode="forward")
	results["13_cumulative_sums_backward"] = cumulative_sums_test(bits, mode="backward")

	excursions = random_excursions_test(bits)
	if excursions is None:
		results["14_random_excursions"] = None
	else:
		for state, p_value in excursions.items():
			results[f"14_random_excursions_x{state:+d}"] = p_value

	excursions_variant = random_excursions_variant_test(bits)
	if excursions_variant is None:
		results["15_random_excursions_variant"] = None
	else:
		for state, p_value in excursions_variant.items():
			results[f"15_random_excursions_variant_x{state:+d}"] = p_value

	verdicts = {}
	for name, p_value in results.items():
		if p_value is None:
			verdicts[name] = None
		else:
			verdicts[name] = bool(p_value >= alpha)

	return results, verdicts, n


def _print_suite(title, results, verdicts, n, alpha):
	n_pass = sum(1 for v in verdicts.values() if v is True)
	n_fail = sum(1 for v in verdicts.values() if v is False)
	n_na = sum(1 for v in verdicts.values() if v is None)
	print(f"=== {title} ===")
	print(f"n={n}  alpha={alpha}  gecti={n_pass}  basarisiz={n_fail}  uygulanamaz={n_na}")
	for name, p in results.items():
		mark = "N/A" if p is None else ("PASS" if verdicts[name] else "FAIL")
		p_str = "N/A" if p is None else f"{p:.4e}"
		print(f"  {name:45s} p={p_str:>12s}  {mark}")
	print()
	return n_pass, n_fail, n_na


if __name__ == "__main__":
	import sys
	import json
	sys.path.insert(0, ".")
	from gravity_map import iterate_map, DEFAULT_INITIAL_STATE

	n_target = 1048576
	transient = 2000
	alpha = 0.01

	rng = np.random.default_rng(0)
	control_series = rng.uniform(-1, 1, n_target)
	res_ctrl, verd_ctrl, n_ctrl = run_full_nist_suite(control_series, alpha=alpha)
	_print_suite("Kontrol: numpy PCG64 (bilinen iyi rastgelelik)", res_ctrl, verd_ctrl, n_ctrl, alpha)

	traj_chaotic = iterate_map(DEFAULT_INITIAL_STATE, 0.97, n_target + transient)
	series_chaotic = traj_chaotic[transient:, 0]
	res_chaotic, verd_chaotic, n_chaotic = run_full_nist_suite(series_chaotic, alpha=alpha)
	_print_suite("Onerilen harita: a=0.97 (kaotik referans)", res_chaotic, verd_chaotic, n_chaotic, alpha)

	traj_regular = iterate_map(DEFAULT_INITIAL_STATE, 5.2, n_target + transient)
	series_regular = traj_regular[transient:, 0]
	res_regular, verd_regular, n_regular = run_full_nist_suite(series_regular, alpha=alpha)
	_print_suite("Onerilen harita: a=5.2 (periyodik referans)", res_regular, verd_regular, n_regular, alpha)

	def to_jsonable(results, verdicts):
		out = {}
		for name in results:
			p = results[name]
			out[name] = {
				"p_value": None if p is None else float(p),
				"verdict": verdicts[name],
			}
		return out

	summary = {
		"n_target": n_target,
		"alpha": alpha,
		"control_pcg64": to_jsonable(res_ctrl, verd_ctrl),
		"chaotic_a_0_97": to_jsonable(res_chaotic, verd_chaotic),
		"regular_a_5_2": to_jsonable(res_regular, verd_regular),
	}
	with open("nist_full_suite_results.json", "w") as f:
		json.dump(summary, f, indent=2)
	print("Sonuclar nist_full_suite_results.json dosyasina kaydedildi.")
