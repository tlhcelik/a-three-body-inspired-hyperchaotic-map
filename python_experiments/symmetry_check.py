"""
Discrete symmetries of the map (reviewer items R13, R19, R25).

Candidate group: body permutations (S3, equal masses) x axis permutations (S3) x axis reflections ({+-1}^3),
acting on positions P (3x3) as  P' = P[pi][:, sigma] * s. Softened distances, Phi and B are invariant; D_{i,k}
is odd under a reflection of axis k and sin is odd, so the map should commute with every element (288 elements).

	equivariance      max |F(g x) - g F(x)| over random states and all group elements
	orbit_classes     the period-2 attractors found at a = 5.2 grouped into symmetry classes
	global_inversion  F(-x) = -F(x) check (the element s = (-1,-1,-1), identity permutations)

Output: results/symmetry_check.json

tab-indented, snake_case, Allman braces.
"""

import itertools
import numpy as np

import settings as st
from gravity_map import map_step_vec
from analysis_utils import save_json, load_json


def group_elements():
	elements = []
	for body_perm in itertools.permutations(range(3)):
		for axis_perm in itertools.permutations(range(3)):
			for signs in itertools.product((1.0, -1.0), repeat=3):
				elements.append((body_perm, axis_perm, np.array(signs)))
	return elements


def act(element, states):
	"""Apply a group element to states of shape (9,) or (M,9)."""
	body_perm, axis_perm, signs = element
	pos = states.reshape(states.shape[:-1] + (3, 3))
	moved = pos[..., list(body_perm), :][..., :, list(axis_perm)] * signs
	return moved.reshape(states.shape)


def orbit_points(point, a_param, period):
	pts = [np.asarray(point, dtype=float)]
	for _ in range(period - 1):
		pts.append(map_step_vec(pts[-1], a_param))
	arr = np.array(pts)
	return arr[np.lexsort(arr.T[::-1])]


def main():
	rng = np.random.default_rng(st.SEED_IC)
	elements = group_elements()
	states = rng.uniform(-0.9, 0.9, size=(200, 9))
	worst = 0.0
	for element in elements:
		for a_param in (st.A_CHAOTIC, st.A_REGULAR):
			lhs = map_step_vec(act(element, states), a_param)
			rhs = act(element, map_step_vec(states, a_param))
			worst = max(worst, float(np.max(np.abs(lhs - rhs))))
	out = {"group_order": len(elements), "equivariance_max_abs_error": worst}
	print("group order", len(elements), "max equivariance error", worst)

	try:
		scan = load_json("a52_multistability.json")["basin_scan_a52"]
	except FileNotFoundError:
		scan = []
	periodic = [s for s in scan if s["period"] == 2]
	orbits = [orbit_points(s["example_point"], st.A_REGULAR, 2) for s in periodic]
	class_of = [-1] * len(orbits)
	n_classes = 0
	for i in range(len(orbits)):
		if class_of[i] >= 0:
			continue
		class_of[i] = n_classes
		images = [orbit_points(act(e, orbits[i][0]), st.A_REGULAR, 2) for e in elements]
		for j in range(i + 1, len(orbits)):
			if class_of[j] >= 0:
				continue
			if any(np.max(np.abs(img - orbits[j])) < 1.0e-6 for img in images):
				class_of[j] = n_classes
		n_classes += 1
	distinct_images = []
	if orbits:
		for e in elements:
			img = orbit_points(act(e, orbits[0][0]), st.A_REGULAR, 2)
			if not any(np.max(np.abs(img - d)) < 1.0e-6 for d in distinct_images):
				distinct_images.append(img)
	out["a52_period2_attractors_found"] = len(orbits)
	out["a52_symmetry_classes"] = n_classes
	out["a52_class_sizes_found"] = [class_of.count(c) for c in range(n_classes)]
	out["a52_distinct_images_of_first_orbit"] = len(distinct_images)
	print("a=5.2: attractors found", len(orbits), "symmetry classes", n_classes, "sizes", out["a52_class_sizes_found"], "distinct images of first orbit", len(distinct_images))
	print("->", save_json("symmetry_check.json", out))


if __name__ == "__main__":
	main()
