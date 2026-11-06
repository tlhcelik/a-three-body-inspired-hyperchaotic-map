# Three-body-inspired bounded hyperchaotic map — reproduction code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23806191.svg)](https://doi.org/10.5281/zenodo.23806191)

Code accompanying *"A Three-Body-Inspired Bounded Hyperchaotic Map: Mechanism, Scaling with the Number of Bodies, and Numerical Reliability"*.

## Requirements

Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`, `mpmath` (only for the 60-digit reference).

```
pip install numpy scipy matplotlib mpmath
```

## Layout

| Path | Content |
|---|---|
| `rgmap/core.py` | Vectorised map for any number of bodies, masses, softening, boundary and coupling variant; closed-form Jacobian; two-trajectory and QR Lyapunov estimators; exact decomposition of the exponent sum |
| `rgmap/controls.py` | Helpers for the coupling controls: magnitude matching by rms of F, median of \|F\|, median of \|D\| on the control's own attractor; shared evaluation of a control |
| `rgmap/hp.py` | 60-digit reference implementation (mpmath) |
| `tests_core.py` | Jacobian vs finite differences, reference exponents |
| `exp_mechanism.py` | First version of the coupling controls (rms matching), sum decomposition → `results/mechanism.json` |
| `exp_matching.py` | Controls under three matching rules and a finer set of distance exponents p → `results/matching.json` |
| `exp_strength.py` | Best strength of every coupling family (criterion-free comparison) → `results/strength.json` |
| `exp_nbody.py` | N = 1…6 bodies and mass ratios → `results/nbody.json` |
| `exp_retune.py` | Re-tuning of ε and R² for N = 3…6 → `results/retune.json` |
| `exp_structure.py` | Zero-crossing index of the spectrum for N = 2…8, positive-exponent count along a, factorisation J = diag(g) J_D → `results/structure.json` |
| `exp_twoparam.py` | (a, ε) and (a, R²) Lyapunov diagrams, basin slices → `results/twoparam.npz` |
| `exp_basin_uncert.py`, `exp_transient.py` | Basin intermingling and transient chaos at a = 5.2 |
| `exp_compare.py` | Comparison with established maps → `results/compare.json` |
| `repro.py` | Reproduction of the core analyses with figures (scan, bifurcation, phase, complexity, sensitivity, ablation, fixed points, reliability) |
| `make_figures.py`, `gen_tables.py` | Figures and LaTeX tables/macros for the manuscript (written next to makale_en.tex); `gen_tables.py` also writes Turkish tables to `../tables_tr/`, `make_figures.py` Turkish-labelled figures to `figures_tr/` (copied as `../figures/fig_*_tr.pdf` for makale.tex) |
| `python_experiments/` | Pipeline of the reliability and statistical analyses: noise floor and classification threshold, ε and R² scans, reference values at a = 0.97 and 5.2, Lyapunov spectra, multistability at a = 5.2, symmetry group, invariant density, fixed points, component and fold ablation, 60-digit comparison (Python `decimal`), parameter-space bound, NIST SP 800-22 multi-sequence protocol, mantissa bit-index scan. `run_all.py` runs the 19 steps in order; `build_tables.py` writes `tables/numbers*.tex` (number macros read by both manuscripts) and `tables/tab_*.tex`; `figures_revision.py` and three analysis scripts write the Turkish-labelled figures of makale.tex to `figures/`. Settings and seeds are in `settings.py` |
| `python_experiments/ablation_no_sin/` | Map without the fold: survival time before divergence over a ∈ [−25, 25] and [−6, 8] → `figures/fig_survival_vs_a.png` (`fig_sin_ablation_survival.png` in makale.tex) |
| `results_ilk_teslim/`, `figures_ilk_teslim/` | Results and figures of the first delivery (another platform), kept for comparison |

## Running

```
python tests_core.py
python exp_mechanism.py
python exp_matching.py 6        # argument: number of worker processes
python exp_strength.py 6
python exp_nbody.py
python exp_retune.py 6
python exp_structure.py 6       # needs results/nbody.json and results/retune.json
python exp_twoparam.py
python exp_basin_uncert.py
python exp_transient.py
python exp_compare.py           # needs results/mechanism.json
python repro.py
python make_figures.py          # needs all results above
python gen_tables.py            # writes ../tables_en/
cp figures/*.pdf ../figures_en/
```

The reliability and statistical analyses are in `python_experiments/`:

```
cd python_experiments
python run_all.py --quick            # smoke test with reduced sizes, about 18 minutes (writes *_quick results, no tables)
python run_all.py                    # full run from scratch
python run_all.py --from nist_protocol   # resume at a step
python build_tables.py               # tables and number macros from the stored results only
cd ablation_no_sin
python divergence_scan.py            # writes divergence_scan_*.npz
python generate_diagnostic_figures.py
```

The `--latex` option of `run_all.py` compiles the authors' working manuscript and has no effect here.

Measured on a 12-core Windows machine (Python 3.11.3, NumPy 2.3.4): `python_experiments/run_all.py` 4.1 hours, of which the NIST protocol takes 3.8 hours with 11 processes and the other 18 steps 17 minutes; `tests_core.py` 5 s; the first-delivery experiments (`exp_mechanism` … `exp_compare`, `repro.py`) about 20 minutes when run in parallel (`exp_twoparam.py` dominates with 20 minutes); `exp_matching` about 4 minutes, `exp_strength` about 5 minutes, `exp_retune` about 10 minutes, `exp_structure` about 1.5 minutes with 6 workers. Random seeds are fixed in each script.

## Reproducibility notes

* A one-ulp difference is amplified to order one within about ten iterations, so results that depend on individual orbits (not on ensemble statistics) differ between platforms.
* Re-running the first-delivery pipeline on a different machine reproduced the scan (chaotic ratio 0.929), the N-body table, the mass-ratio table, the comparison table, transient statistics and basin statistics within sampling noise. It did **not** reproduce the controls matched by the root-mean-square of F: the rms of the Newtonian force on the 4000 reference states changed from 1.70 to 1.93 (heavy-tailed, dominated by rare close encounters), which moved the p = 1, p = 5 and linear controls from non-hyperchaotic to hyperchaotic. This is why the controls are now compared under median-based matching rules and at each family's best strength (`exp_matching.py`, `exp_strength.py`).
* The argmax a*(N) of the median exponent (used in the N-body table) sits on a flat maximum and moved for N = 2 and 3 (0.1 → 0.2, 0.3 → 0.6); the positive-exponent counts at a*(N) did not change.
* `python_experiments/run_all.py` was re-run from an empty results folder on the machine above (28 October 2026). All 12 tables and number macros, all result files and all figures came out identical to the stored ones (tables and result files byte for byte, figures pixel for pixel); the only differences were the wall-clock timings recorded in `verify_phase1.json`. The same holds for `ablation_no_sin/`.

## Result files

`results/` holds the outputs used in the manuscript, all produced on one machine (Windows 10, Python 3.11.3, NumPy 2.3.4). `*_log.txt` are the console logs of the runs. `retune_pass1.json` and `strength_pass1.json` are first passes with narrower grids (ε ≥ 0.05, R² ≤ 16; strengths up to 2⁴ s₀); they were superseded by `retune.json` and `strength.json` because the optimum lay on the edge of the first grids. `figures_tr/` contains the same figures with Turkish labels for the Turkish manuscript.

`python_experiments/results/` holds the outputs of the reliability pipeline. `build_tables.py` reads only the `.json` files and `epsilon_scan.npz`, so the tables can be rebuilt in a second without re-running the analyses. `nist_protocol_pvalues.json` (18 MB) contains every individual NIST p-value of the protocol; `nist_protocol.json` holds the summaries used in the tables. `bifurcation.npz` and `pe_scan.npz` are written by `figures_revision.py`.

## Map

For bodies i = 1…N with coordinates q_{i,k}:

```
r~_ij = sqrt(|r_i − r_j|² + ε²)
Φ_i   = −G Σ_{j≠i} m_j / r~_ij
F_ik  =  G Σ_{j≠i} m_i m_j (q_jk − q_ik) / r~_ij³
B_i   = 1 − |r_i|² / R²
D_ik  = a q_ik (1 + Φ_i) + B_i q_ik + F_ik
q_ik ← sin(G / D_ik)
```

Defaults: N = 3, m = 1, G = 1, ε = 0.1, R² = 2; reference parameters a = 0.97 (hyperchaotic) and a = 5.2 (periodic). The reference initial state is the rounded NumPy draw `np.random.seed(42); np.random.rand(9) - 0.5` (the two equal entries −0.3440 are the roundings of −0.34398 and −0.34401).

## Citation

If you use this code, please cite the article and the archived code. The citation data are in `CITATION.cff`. The code is archived on Zenodo: [10.5281/zenodo.23806191](https://doi.org/10.5281/zenodo.23806191) always resolves to the latest version. The article uses version 0.1.1. Version 0.1.0 has the DOI [10.5281/zenodo.23806192](https://doi.org/10.5281/zenodo.23806192).

## License

Licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright and attribution notices are in `NOTICE`.
