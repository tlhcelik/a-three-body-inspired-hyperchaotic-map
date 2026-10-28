"""
Reproduce every number, table and figure of the revised manuscript (reviewer item R2).

Order matters: noise_floor must precede the scans that use its threshold; precision_check precedes the
figures and the key-space analysis that overlay the double-precision floor.

Usage:
	python run_all.py            full run (several hours; NIST protocol uses 11 processes)
	python run_all.py --quick    smoke test with reduced sizes (writes *_quick results, no manuscript tables)
	python run_all.py --from X   start at step X (e.g. --from nist_protocol)
	python run_all.py --latex    also compile ../main.tex (pdflatex, bibtex, pdflatex x2)

tab-indented, snake_case, Allman braces.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
	("verify_phase1", "vectorised map and analytic Jacobian checks"),
	("noise_floor", "estimator noise floor and classification threshold"),
	("epsilon_scan", "softening scan over the a grid"),
	("canonical_reference", "reference values at a = 0.97 and 5.2"),
	("pe_rounding_sensitivity", "permutation-entropy tie-rounding precision study"),
	("spectrum", "Lyapunov spectra and Jacobian/delta0 studies"),
	("r2_boundary_analysis", "boundary parameter table and figures"),
	("robustness_a", "robustness of the chaotic-band parameters"),
	("a52_multistability", "attractors, basins and on-attractor sensitivity"),
	("symmetry_check", "288-element symmetry group"),
	("shannon_density", "invariant density and Shannon entropy"),
	("fixed_points", "fixed points with the analytic Jacobian"),
	("gravity_ablation", "component and fold ablation"),
	("precision_check", "60-digit comparison and Brent cycles"),
	("key_space_analysis", "parameter-space bound and key sensitivity"),
	("nist_protocol", "NIST SP 800-22 multi-sequence protocol and controls"),
	("bit_index_scan", "mantissa bit-index scan"),
	("figures_revision", "manuscript figures"),
	("build_tables", "LaTeX tables and number macros"),
]

QUICK_CAPABLE = {"noise_floor", "epsilon_scan", "canonical_reference", "pe_rounding_sensitivity", "spectrum", "r2_boundary_analysis", "robustness_a", "a52_multistability", "shannon_density", "fixed_points", "gravity_ablation", "precision_check", "key_space_analysis", "nist_protocol", "bit_index_scan"}


def run_step(name, quick):
	args = [sys.executable, os.path.join(HERE, name + ".py")]
	if quick:
		if name not in QUICK_CAPABLE:
			return None
		args.append("--quick")
	log_path = os.path.join(HERE, "results", "logs", name + ("_quick" if quick else "") + ".log")
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	t0 = time.perf_counter()
	with open(log_path, "w", encoding="utf-8") as log:
		code = subprocess.call(args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
	return code, time.perf_counter() - t0, log_path


def compile_latex():
	root = os.path.dirname(HERE)
	for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
		subprocess.call(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	print("latex build finished; see main.log")


def main():
	quick = "--quick" in sys.argv
	start = None
	if "--from" in sys.argv:
		start = sys.argv[sys.argv.index("--from") + 1]
	started = start is None
	for name, description in STEPS:
		if not started:
			started = name == start
			if not started:
				continue
		result = run_step(name, quick)
		if result is None:
			print("skip   %-22s (no quick mode)" % name)
			continue
		code, seconds, log_path = result
		print("%s %-22s %7.0fs  %s" % ("ok    " if code == 0 else "FAILED", name, seconds, description), flush=True)
		if code != 0:
			print("see", log_path)
			sys.exit(code)
	if "--latex" in sys.argv and not quick:
		compile_latex()


if __name__ == "__main__":
	main()
