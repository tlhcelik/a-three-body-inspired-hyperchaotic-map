"""
Canonical computation settings for the revised manuscript.
Every analysis script reads its settings from here so that one number in the paper
corresponds to one configuration (reviewer items R7, R28).

tab-indented, snake_case, Allman braces.
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TABLES_DIR = os.path.join(HERE, "tables")
FIGURES_DIR = os.path.join(HERE, "figures")

# Reference parameters
A_CHAOTIC = 0.97
A_REGULAR = 5.2
A_BAND = [-1.0, 0.97, 2.0, 6.0]

# Parameter grid for scans: a in [-6, 8] with step 0.1 -> N = 141
A_GRID = np.round(np.arange(-6.0, 8.0 + 1e-9, 0.1), 10)

# Softening values compared in the epsilon scan
EPSILON_VALUES = [1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.0e-1]

# Lyapunov estimation
N_TRANSIENT = 1000
N_ITER_LYAP = 10000
N_TRANSIENT_SCAN = 500
N_ITER_SCAN = 3000
DELTA0 = 1.0e-8
JAC_STEP = 1.0e-6

# Initial-condition ensembles: uniform box of half-width IC_RADIUS around DEFAULT_INITIAL_STATE
N_IC = 20
IC_RADIUS = 0.15

# Time-series statistics
N_SERIES = 20000
ZERO_ONE_N_C = 50
PE_EMBEDDING = 7
PE_ROUND_DECIMALS = 7
N_PE = 200000
PE_ROUND_SCAN = [4, 5, 6, 7, 8, 9, 10, 12, 14, 15]
PE_ROUND_REFERENCE = 12
PE_ROUND_TRANSIENT = 10000
PE_STABILITY_TOL = 5.0e-5
SHANNON_BINS = 256
CORR_N_POINTS = 10000

# Random seeds
SEED_IC = 2026
SEED_ZERO_ONE = 42
SEED_CORR = 7
SEED_NIST = 800
SEED_PERTURB = 11


def ensure_dirs():
	"""Create output directories if missing."""
	for path in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
		os.makedirs(path, exist_ok=True)


def result_path(name):
	"""Path of a JSON result file inside RESULTS_DIR."""
	ensure_dirs()
	return os.path.join(RESULTS_DIR, name)
