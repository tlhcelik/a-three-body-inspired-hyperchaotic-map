"""
Ablation variant of the bounded gravitational chaotic map: the outer sin(.)
wrapper is removed from Eq. (eq:proposed_map).

Original:
	q_{n+1} = sin[ G / ( a*q_n*(1+Phi_i) + (1 - r_i^2/R^2)*q_n + F_q ) ]

Ablated (this file):
	q_{n+1} = G / ( a*q_n*(1+Phi_i) + (1 - r_i^2/R^2)*q_n + F_q )

Everything else (softening, R^2 boundary, masses, initial state) is kept
identical to python_experiments/gravity_map.py so that any behavioral change
can be attributed solely to the removal of sin(.).

tab-indented, snake_case, Allman braces, no line-continuation splitting.
"""

import numpy as np

G_CONST = 1.0
MASS = 1.0
R2_BOUNDARY = 2.0
SOFTENING = 0.1

DEFAULT_INITIAL_STATE = np.array(
[
	-0.1255, 0.4507, 0.232,
	0.0987, -0.3440, -0.3440,
	-0.4419, 0.3662, 0.1011,
]
)

DIVERGENCE_THRESHOLD = 1.0e8


def unpack_state(state):
	return state.reshape(3, 3)


def pack_state(positions):
	return positions.reshape(9)


def pairwise_distances(positions):
	diff = positions[:, None, :] - positions[None, :, :]
	sq_dist = np.sum(diff * diff, axis=-1)
	soft_dist = np.sqrt(sq_dist + SOFTENING * SOFTENING)
	return soft_dist, diff


def gravitational_potential(positions):
	r_ij, _ = pairwise_distances(positions)
	inv_r = 1.0 / r_ij
	np.fill_diagonal(inv_r, 0.0)
	phi = -G_CONST * MASS * np.sum(inv_r, axis=1)
	return phi


def gravitational_force(positions):
	r_ij, diff = pairwise_distances(positions)
	inv_r3 = 1.0 / (r_ij ** 3)
	np.fill_diagonal(inv_r3, 0.0)
	force = G_CONST * MASS * MASS * np.einsum("ij,ijk->ik", inv_r3, -diff)
	return force


def radial_boundary_term(positions):
	r2 = np.sum(positions * positions, axis=1)
	return 1.0 - r2 / R2_BOUNDARY


def map_step(state, a_param):
	"""Advance the state by one iteration of the sin(.)-ablated map.
	No output clamping is applied here (by design) so that divergence, if
	present, is directly observable by the caller."""
	positions = unpack_state(state)
	phi = gravitational_potential(positions)
	force = gravitational_force(positions)
	boundary = radial_boundary_term(positions)

	new_positions = np.empty_like(positions)
	for body_index in range(3):
		for axis_index in range(3):
			q_n = positions[body_index, axis_index]
			self_dynamic_term = a_param * q_n * (1.0 + phi[body_index])
			boundary_term = boundary[body_index] * q_n
			force_term = force[body_index, axis_index]
			denominator = self_dynamic_term + boundary_term + force_term
			if abs(denominator) < 1.0e-12:
				denominator = 1.0e-12 if denominator >= 0 else -1.0e-12
			new_positions[body_index, axis_index] = G_CONST / denominator

	return pack_state(new_positions)


def iterate_map(initial_state, a_param, n_iterations, stop_on_divergence=True):
	"""Return the trajectory (n_iterations+1, 9). If stop_on_divergence is True,
	iteration halts early (remaining rows filled with NaN) once any coordinate
	exceeds DIVERGENCE_THRESHOLD or becomes non-finite."""
	trajectory = np.full((n_iterations + 1, 9), np.nan)
	trajectory[0] = initial_state
	state = initial_state.copy()
	diverged_at = None
	for iteration_index in range(n_iterations):
		state = map_step(state, a_param)
		if stop_on_divergence:
			if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > DIVERGENCE_THRESHOLD:
				diverged_at = iteration_index + 1
				break
		trajectory[iteration_index + 1] = state
	return trajectory, diverged_at


if __name__ == "__main__":
	for a_test in [0.97, 5.2, 0.1, -0.5, 1.0, 2.0]:
		traj, div_at = iterate_map(DEFAULT_INITIAL_STATE, a_test, 200)
		if div_at is not None:
			print(f"a={a_test:+.3f}  IRAKSADI iterasyon={div_at}  son deger buyuklugu(sonlu son satirdan onceki)={np.max(np.abs(traj[div_at-1])):.3e}")
		else:
			print(f"a={a_test:+.3f}  siniirli kaldi (200 iter)  max|x|={np.nanmax(np.abs(traj)):.3e}")
