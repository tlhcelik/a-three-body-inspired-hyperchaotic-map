"""
Mantissa bit-index scan and independence of the three XOR inputs (reviewer item R18).

For bit positions k = 0..51 (0 = least significant mantissa bit, the manuscript uses k = 30) the pass rate
of a fast NIST subset (10 p-values: monobit, block frequency, runs, longest run, DFT, serial x2,
approximate entropy, cumulative sums x2) is measured for
	map_x1     bit k of x1 of the map at a = 0.97
	map_xor3   XOR of bit k of x1, y2, z3 of the map at a = 0.97
	rot_xor3   XOR of bit k of sin(n*theta_j + phi_j), theta = (1, sqrt2, sqrt3)  (lambda = 0 control)
over N_SCAN_SEQ sequences of SCAN_BITS bits. At k = 30 the pairwise correlation and mutual information of
c1, c2, c3 (map, a = 0.97) and their bias are reported.

Output: results/bit_index_scan.json

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np

import settings as st
import nist_full_suite as ref
from gravity_map import advance, map_step_vec, sample_initial_states
from analysis_utils import save_json

SCAN_BITS = 2 ** 18
N_SCAN_SEQ = 5


def fast_subset(bits):
	b = bits.astype(np.int64)
	p = [
		ref.frequency_monobit_test(b),
		ref.block_frequency_test(b, block_size=128),
		ref.runs_test(b),
		ref.longest_run_ones_test(b),
		ref.dft_spectral_test(b),
	]
	p.extend(ref.serial_test(b, m=2))
	p.append(ref.approximate_entropy_test(b, m=2))
	p.append(ref.cumulative_sums_test(b, mode="forward"))
	p.append(ref.cumulative_sums_test(b, mode="backward"))
	return np.array(p, dtype=float)


def bits_at(values, k):
	return ((np.ascontiguousarray(values, dtype=np.float64).view(np.uint64) >> np.uint64(k)) & np.uint64(1)).astype(np.uint8)


def mutual_information(x, y):
	joint = np.histogram2d(x, y, bins=2)[0] / x.size
	px = joint.sum(axis=1)
	py = joint.sum(axis=0)
	mi = 0.0
	for i in range(2):
		for j in range(2):
			if joint[i, j] > 0:
				mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
	return float(mi)


def main(quick=False):
	n_bits = 2 ** 14 if quick else SCAN_BITS
	n_seq = 2 if quick else N_SCAN_SEQ
	k_values = list(range(0, 52, 8)) if quick else list(range(52))

	states = advance(sample_initial_states(n_seq, st.IC_RADIUS, st.SEED_NIST), st.A_CHAOTIC, 2000)
	coords = np.empty((n_bits, n_seq, 3))
	for n in range(n_bits):
		states = map_step_vec(states, st.A_CHAOTIC)
		coords[n] = states[:, [0, 4, 8]]
	rng = np.random.default_rng(st.SEED_NIST)
	n_axis = np.arange(1, n_bits + 1, dtype=np.float64)
	thetas = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
	rot = np.empty((n_bits, n_seq, 3))
	for i in range(n_seq):
		phases = rng.uniform(0.0, 2.0 * np.pi, 3)
		for j in range(3):
			rot[:, i, j] = np.sin(n_axis * thetas[j] + phases[j])

	rates = {"map_x1": [], "map_xor3": [], "rot_xor3": []}
	for k in k_values:
		for name, source, xor in (("map_x1", coords, False), ("map_xor3", coords, True), ("rot_xor3", rot, True)):
			passes = []
			for i in range(n_seq):
				bits = bits_at(source[:, i, 0], k)
				if xor:
					bits = bits ^ bits_at(source[:, i, 1], k) ^ bits_at(source[:, i, 2], k)
				passes.append(np.mean(fast_subset(bits) >= 0.01))
			rates[name].append(float(np.mean(passes)))
		print("k=%2d map_x1=%.2f map_xor3=%.2f rot_xor3=%.2f" % (k, rates["map_x1"][-1], rates["map_xor3"][-1], rates["rot_xor3"][-1]), flush=True)

	c = [bits_at(coords[:, 0, j], 30) for j in range(3)]
	independence = {
		"bias": [float(ci.mean()) for ci in c],
		"corr_12": float(np.corrcoef(c[0], c[1])[0, 1]),
		"corr_13": float(np.corrcoef(c[0], c[2])[0, 1]),
		"corr_23": float(np.corrcoef(c[1], c[2])[0, 1]),
		"mi_bits_12": mutual_information(c[0], c[1]),
		"mi_bits_13": mutual_information(c[0], c[2]),
		"mi_bits_23": mutual_information(c[1], c[2]),
		"n_bits": n_bits,
	}
	print("independence at k=30:", independence)
	save_json("bit_index_scan_quick.json" if quick else "bit_index_scan.json", {"k_values": k_values, "n_bits": n_bits, "n_sequences": n_seq, "pass_rate": rates, "independence_k30": independence, "subset": "monobit, block frequency, runs, longest run, DFT, serial x2, approximate entropy, cumsum x2"})


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
