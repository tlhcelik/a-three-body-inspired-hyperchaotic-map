"""
Fixed points by multistart Newton-type root finding with the analytic Jacobian (reviewer item R16).

Starts: half uniform in [-1, 1]^9, half near the symmetric line x = c (1,...,1). Distinct roots are
merged within 1e-7. For each root: residual, MINPACK exit flag, eigenvalue moduli, minimum raw
inter-body distance (0 means coincident bodies), whether all nine coordinates are equal, the three inter-body
distances with their relative spread (0 for an equilateral triangle) and the size of the stabiliser and of the
orbit under the 288-element symmetry group (symmetry_check.group_elements).

Output: results/fixed_points.json

tab-indented, snake_case, Allman braces.
"""

import sys
import numpy as np
from scipy.optimize import fsolve

import settings as st
from gravity_map import map_step_vec
from jacobian import analytic_jacobian
from analysis_utils import save_json, min_pair_distance
from symmetry_check import group_elements, act


def geometry_and_symmetry(x_star):
	"""Pairwise body distances, their relative spread, and stabiliser and orbit size under the 288-element group."""
	pos = x_star.reshape(3, 3)
	sides = [float(np.linalg.norm(pos[i] - pos[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
	mean_side = float(np.mean(sides))
	spread = (max(sides) - min(sides)) / mean_side if mean_side > 0.0 else None
	stabiliser = 0
	images = set()
	for element in group_elements():
		image = act(element, x_star)
		if np.max(np.abs(image - x_star)) < 1.0e-9:
			stabiliser += 1
		images.add(tuple(np.round(image, 8)))
	return {"body_distances": sides, "distance_spread": spread, "stabiliser_size": stabiliser, "orbit_size": len(images)}


def find_roots(a_param, n_starts, seed):
	rng = np.random.default_rng(seed)
	half = n_starts // 2
	starts = np.concatenate([
		np.full((1, 9), 1.0e-3),
		np.zeros((1, 9)) + rng.normal(scale=1.0e-3, size=(1, 9)),
		rng.uniform(-1.0, 1.0, size=(half, 9)),
		rng.uniform(-1.0, 1.0, size=(n_starts - half, 1)) * np.ones((1, 9)) + rng.normal(scale=0.01, size=(n_starts - half, 9)),
	])
	residual = lambda x: map_step_vec(x, a_param) - x
	jac = lambda x: analytic_jacobian(x, a_param) - np.eye(9)
	roots = []
	for x0 in starts:
		x_star, info, ier, _ = fsolve(residual, x0, fprime=jac, full_output=True, xtol=1.0e-13)
		res = float(np.max(np.abs(residual(x_star))))
		if ier != 1 or res > 1.0e-10:
			continue
		for root in roots:
			if np.max(np.abs(root["x"] - x_star)) < 1.0e-7:
				root["hits"] += 1
				break
		else:
			eig = np.abs(np.linalg.eigvals(analytic_jacobian(x_star, a_param)))
			roots.append({
				"x": x_star,
				"residual_inf": res,
				"minpack_ier": int(ier),
				"hits": 1,
				"eig_modulus_min": float(eig.min()),
				"eig_modulus_max": float(eig.max()),
				"stable": bool(eig.max() < 1.0),
				"min_body_distance": float(min_pair_distance(x_star)),
				"all_coordinates_equal": bool(np.ptp(x_star) < 1.0e-9),
				**geometry_and_symmetry(x_star),
			})
	return roots


def main(quick=False):
	n_starts = 20 if quick else 300
	out = {"n_starts": n_starts, "points": {}}
	for a_param in (st.A_CHAOTIC, st.A_REGULAR):
		roots = find_roots(a_param, n_starts, st.SEED_IC)
		out["points"][str(a_param)] = {"n_roots": len(roots), "roots": roots}
		print("a=%s roots=%d" % (a_param, len(roots)))
		for r in roots:
			print("   hits=%d |lambda| in [%.3g, %.3g] stable=%s min_dist=%.3g symmetric=%s x0=%.6f" % (r["hits"], r["eig_modulus_min"], r["eig_modulus_max"], r["stable"], r["min_body_distance"], r["all_coordinates_equal"], r["x"][0]))
	print("->", save_json("fixed_points_quick.json" if quick else "fixed_points.json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
