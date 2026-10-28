"""
NIST SP 800-22 multi-sequence protocol with negative and positive controls (reviewer items R17, R18).

Every condition produces N_SEQ sequences of 2^20 bits, each run through the full suite with all 148
aperiodic non-overlapping templates (188 p-values when the random-excursion tests apply), and is assessed
with the Section 4.2 proportion and uniformity criteria (nist_fast.nist_multi_sequence_assessment).

Conditions
	map_raw_a{a}      median-thresholded x1 of the map                      a in NIST_A_VALUES
	map_lsb_a{a}      lsb_xor3 of the map (bit 30 of x1, y2, z3, XOR)       a in NIST_A_VALUES
	rot_raw           median-thresholded sin(n*theta_1 + phi), theta_1 = 1  (lambda = 0, aperiodic)
	rot_lsb           lsb_xor3 of sin(n*theta_k + phi_k), theta = (1, sqrt2, sqrt3)
	cheb_lsb          lsb_xor3 of three Chebyshev maps z -> 1 - 2 z^2 (conjugate to logistic r = 4)
	v3_lsb            lsb_xor3 of the map without Phi and F (weak chaos) at a = 0.97
	pcg_lsb           lsb_xor3 of PCG64 uniform doubles on [-1, 1]
	pcg_bits          PCG64 random bits (harness control)
Sequence i of a condition depends only on (seed, i), so results do not depend on how work is chunked.
For continuity with the manuscript, the pass count of the first map sequence on the original
45-p-value configuration (5 templates) is also stored.

Outputs: results/nist_protocol.json (assessment), results/nist_protocol_pvalues.json (all p-values)

Usage: python nist_protocol.py [--quick] [family ...]   families: map, rot, cheb, v3, pcg

tab-indented, snake_case, Allman braces.
"""

import sys
import time
import numpy as np
from multiprocessing import Pool

import settings as st
from gravity_map import map_step_vec, advance, sample_initial_states
from nist_fast import run_full_nist_suite_fast, nist_multi_sequence_assessment
from analysis_utils import save_json, load_json

N_BITS = 2 ** 20
N_SEQ = 100
CHUNK = 10
NIST_A_VALUES = [0.97, 2.0, 6.0, -1.0, 5.2]
ORIGINAL_TEMPLATES = ["000000001", "000000011", "000000101", "111111110", "000010011"]
BIT_POSITION = 30
TRANSIENT = 2000


def _bit(values, position=BIT_POSITION):
	return ((np.ascontiguousarray(values, dtype=np.float64).view(np.uint64) >> np.uint64(position)) & np.uint64(1)).astype(np.uint8)


def _median_bits(series):
	return (series >= np.median(series)).astype(np.uint8)


def _suite(bits):
	results, _, _ = run_full_nist_suite_fast(bits.astype(np.int64), alpha=0.01, templates="all")
	return {k: (None if v is None else float(v)) for k, v in results.items()}


def _map_chunk(a_param, indices, n_bits, map_kwargs):
	keys = sample_initial_states(N_SEQ, st.IC_RADIUS, st.SEED_NIST)[indices]
	states = advance(keys, a_param, TRANSIENT, **map_kwargs)
	m = states.shape[0]
	x1 = np.empty((n_bits, m))
	lsb = np.empty((n_bits, m), dtype=np.uint8)
	for n in range(n_bits):
		states = map_step_vec(states, a_param, **map_kwargs)
		x1[n] = states[:, 0]
		lsb[n] = _bit(states[:, 0]) ^ _bit(states[:, 4]) ^ _bit(states[:, 8])
	return x1, lsb


def run_task(task):
	"""Worker: generate one chunk of sequences for a family and run the suite on every stream."""
	family, param, indices, n_bits = task
	out = {}
	if family == "map" or family == "v3":
		kwargs = {"use_phi": False, "use_force": False} if family == "v3" else {}
		x1, lsb = _map_chunk(param, np.array(indices), n_bits, kwargs)
		if family == "map":
			out["raw"] = [_suite(_median_bits(x1[:, i])) for i in range(len(indices))]
		out["lsb"] = [_suite(lsb[:, i]) for i in range(len(indices))]
	elif family == "rot":
		thetas = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
		n_axis = np.arange(1, n_bits + 1, dtype=np.float64)
		out["raw"], out["lsb"] = [], []
		for index in indices:
			phases = np.random.default_rng([st.SEED_NIST, index]).uniform(0.0, 2.0 * np.pi, 3)
			series = [np.sin(n_axis * thetas[k] + phases[k]) for k in range(3)]
			out["raw"].append(_suite(_median_bits(series[0])))
			out["lsb"].append(_suite(_bit(series[0]) ^ _bit(series[1]) ^ _bit(series[2])))
	elif family == "cheb":
		z = np.array([np.random.default_rng([st.SEED_NIST, index]).uniform(-1.0, 1.0, 3) for index in indices])
		for _ in range(TRANSIENT):
			z = 1.0 - 2.0 * z * z
		bits = np.empty((n_bits, len(indices)), dtype=np.uint8)
		for n in range(n_bits):
			z = 1.0 - 2.0 * z * z
			bits[n] = _bit(z[:, 0]) ^ _bit(z[:, 1]) ^ _bit(z[:, 2])
		out["lsb"] = [_suite(bits[:, i]) for i in range(len(indices))]
	elif family == "pcg":
		out["lsb"], out["bits"] = [], []
		for index in indices:
			rng = np.random.default_rng([st.SEED_NIST, index])
			u = rng.uniform(-1.0, 1.0, (n_bits, 3))
			out["lsb"].append(_suite(_bit(u[:, 0]) ^ _bit(u[:, 1]) ^ _bit(u[:, 2])))
			out["bits"].append(_suite(rng.integers(0, 2, n_bits).astype(np.uint8)))
	return family, param, indices, out


def original_subset_count(pvalues):
	"""Pass count / computed count on the manuscript's 45-p-value configuration (5 templates)."""
	keys = [k for k in pvalues if not k.startswith("7_non_overlapping_template_") or k.split("_")[-1] in ORIGINAL_TEMPLATES]
	computed = [pvalues[k] for k in keys if pvalues[k] is not None]
	return {"passed": int(sum(p >= 0.01 for p in computed)), "computed": len(computed), "keys": len(keys)}


def build_tasks(families, n_seq, n_bits, chunk):
	tasks = []
	chunks = [list(range(s, min(s + chunk, n_seq))) for s in range(0, n_seq, chunk)]
	for family in families:
		params = NIST_A_VALUES if family == "map" else [st.A_CHAOTIC if family == "v3" else None]
		for param in params:
			for indices in chunks:
				tasks.append((family, param, indices, n_bits))
	return tasks


def condition_name(family, param, stream):
	if family == "map":
		return "map_%s_a%s" % (stream, param)
	return "%s_%s" % (family, stream)


def main():
	quick = "--quick" in sys.argv
	families = [a for a in sys.argv[1:] if not a.startswith("--")] or ["pcg", "rot", "cheb", "v3", "map"]
	n_seq = 4 if quick else N_SEQ
	chunk = 2 if quick else CHUNK
	n_bits = N_BITS
	suffix = "_quick" if quick else ""
	try:
		summary = load_json("nist_protocol%s.json" % suffix)
		all_pvalues = load_json("nist_protocol_pvalues%s.json" % suffix)
	except FileNotFoundError:
		summary = {"n_bits": n_bits, "n_sequences": n_seq, "alpha": 0.01, "bit_position": BIT_POSITION, "transient": TRANSIENT, "conditions": {}}
		all_pvalues = {}

	tasks = build_tasks(families, n_seq, n_bits, chunk)
	collected = {}
	t0 = time.perf_counter()
	with Pool(processes=11) as pool:
		for family, param, indices, out in pool.imap_unordered(run_task, tasks):
			for stream, pvals in out.items():
				name = condition_name(family, param, stream)
				bucket = collected.setdefault(name, {})
				for index, pv in zip(indices, pvals):
					bucket[index] = pv
				if len(bucket) == n_seq:
					ordered = [bucket[i] for i in range(n_seq)]
					assessment = nist_multi_sequence_assessment(ordered)
					assessment["first_sequence_original_45_config"] = original_subset_count(ordered[0])
					summary["conditions"][name] = assessment
					all_pvalues[name] = ordered
					save_json("nist_protocol%s.json" % suffix, summary)
					save_json("nist_protocol_pvalues%s.json" % suffix, all_pvalues)
					print("%-16s types=%3d prop_ok=%3d unif_ok=%3d/%3d all_ok=%-5s first45=%s  [%.0fs]" % (name, assessment["n_pvalue_types"], assessment["proportion_ok_types"], assessment["uniformity_ok_types"], assessment["uniformity_assessed_types"], assessment["all_ok"], assessment["first_sequence_original_45_config"], time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
	main()
