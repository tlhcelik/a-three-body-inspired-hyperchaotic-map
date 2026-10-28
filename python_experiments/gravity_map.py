"""
Sinirli Gravitasyonel Kaotik Harita (Bounded Gravitational Chaotic Map)
------------------------------------------------------------------------
Reference implementation of the map defined in Eq. (eq:proposed_map):

    q_{i,k}^{(n+1)} = sin[ G / D_{i,k}(x_n) ]
    D_{i,k}(x_n)    = a*q_{i,k}*(1+Phi_i) + (1 - r_i^2/R^2)*q_{i,k} + F_{i,k}

with softened pair distances  r~_ij = sqrt(|p_i - p_j|^2 + eps_s^2)  used in both
Phi_i = -G sum_{j!=i} m_j / r~_ij  and  F_{i,k} = G sum_{j!=i} m_i m_j (q_{j,k}-q_{i,k}) / r~_ij^3.

Update order: SYNCHRONOUS (Jacobi). Phi, F and B are evaluated once from x_n and all
nine coordinates are updated together. A sequential (Gauss-Seidel) reading of the same
equation is a different map.

State: 3 bodies, each with 3D position -> 9-dimensional state vector.
Fixed parameters: G=1, m1=m2=m3=1, R^2=2.0, eps_s=0.1.

`map_step` is the scalar reference implementation and is kept unchanged.
`map_step_vec` is the vectorised / batched implementation used by the revision scripts;
it also exposes the ablation switches (use_phi, use_force, use_boundary, fold).

tab-indented, snake_case, Allman braces, no line-continuation splitting.
"""

import numpy as np

G_CONST = 1.0
MASS = 1.0
R2_BOUNDARY = 2.0
SOFTENING = 0.1
CLAMP_THRESHOLD = 1.0e-12

DEFAULT_INITIAL_STATE = np.array(
[
	-0.1255, 0.4507, 0.232,
	0.0987, -0.3440, -0.3440,
	-0.4419, 0.3662, 0.1011,
]
)

# Number of denominator entries that hit the |D| < CLAMP_THRESHOLD guard in map_step_vec.
clamp_counter = {"count": 0}


def unpack_state(state):
	"""Reshape a 9-vector into a (3,3) array of [body, xyz]."""
	return state.reshape(3, 3)


def pack_state(positions):
	"""Flatten a (3,3) array back into a 9-vector."""
	return positions.reshape(9)


def pairwise_distances(positions):
	"""Return softened pairwise distances r_ij as a (3,3) matrix (diagonal unused)."""
	diff = positions[:, None, :] - positions[None, :, :]
	sq_dist = np.sum(diff * diff, axis=-1)
	soft_dist = np.sqrt(sq_dist + SOFTENING * SOFTENING)
	return soft_dist, diff


def gravitational_potential(positions):
	"""Phi_i = -G * sum_{j != i} m_j / r_ij for each body i, shape (3,)."""
	r_ij, _ = pairwise_distances(positions)
	inv_r = 1.0 / r_ij
	np.fill_diagonal(inv_r, 0.0)
	phi = -G_CONST * MASS * np.sum(inv_r, axis=1)
	return phi


def gravitational_force(positions):
	"""F_i (vector, shape (3,3)) = G * sum_{j!=i} m_i*m_j*(r_j - r_i)/r_ij^3."""
	r_ij, diff = pairwise_distances(positions)
	inv_r3 = 1.0 / (r_ij ** 3)
	np.fill_diagonal(inv_r3, 0.0)
	force = G_CONST * MASS * MASS * np.einsum("ij,ijk->ik", inv_r3, -diff)
	return force


def radial_boundary_term(positions):
	"""B_i = 1 - r_i^2 / R^2 for each body, shape (3,)."""
	r2 = np.sum(positions * positions, axis=1)
	return 1.0 - r2 / R2_BOUNDARY


def map_step(state, a_param):
	"""Advance the 9-dimensional state vector by one iteration of the proposed map (scalar reference)."""
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
			new_positions[body_index, axis_index] = np.sin(G_CONST / denominator)

	return pack_state(new_positions)


def iterate_map(initial_state, a_param, n_iterations):
	"""Return the full trajectory (n_iterations+1, 9) of the map."""
	trajectory = np.empty((n_iterations + 1, 9))
	trajectory[0] = initial_state
	state = initial_state.copy()
	for iteration_index in range(n_iterations):
		state = map_step(state, a_param)
		trajectory[iteration_index + 1] = state
	return trajectory


# ---------------------------------------------------------------------------
# Vectorised / batched implementation with ablation switches
# ---------------------------------------------------------------------------

def _sawtooth(u):
	"""Piecewise-linear sawtooth with the same period (2*pi) and range [-1,1) as sin; s(0)=0."""
	phase = u / (2.0 * np.pi) + 0.5
	return 2.0 * (phase - np.floor(phase)) - 1.0


def _triangle(u):
	"""Piecewise-linear triangle wave with the same period, range and zeros as sin (no trigonometry)."""
	s = _sawtooth(u)
	return np.where(s > 0.5, 2.0 * (1.0 - s), np.where(s < -0.5, -2.0 * (1.0 + s), 2.0 * s))


FOLDS = {
	"sin": np.sin,
	"sawtooth": _sawtooth,
	"triangle": _triangle,
	"tanh": np.tanh,
	"none": lambda u: u,
}


def _broadcast_param(value, positions, trailing=2):
	"""Return a scalar or an (M,1,...) array so a per-trajectory parameter broadcasts over (M,3,3).

	`trailing` is the number of singleton axes appended: 2 for (M,3,3) targets, 1 for (M,3) targets.
	"""
	arr = np.asarray(value, dtype=float)
	if arr.ndim == 0:
		return float(arr)
	return arr.reshape((-1,) + (1,) * trailing)


def map_denominators(positions, a_param, use_phi=True, use_force=True, use_boundary=True, softening=None, r2_boundary=None):
	"""Denominators D_{i,k} for positions of shape (3,3) or (M,3,3); same operation order as map_step.

	`a_param`, `softening` and `r2_boundary` may each be a scalar or an (M,) array.
	"""
	eps_s = _broadcast_param(SOFTENING if softening is None else softening, positions, 2)
	r2_b = _broadcast_param(R2_BOUNDARY if r2_boundary is None else r2_boundary, positions, 1)
	a_b = _broadcast_param(a_param, positions, 2)
	off_diag = 1.0 - np.eye(3)

	diff = positions[..., :, None, :] - positions[..., None, :, :]
	sq_dist = np.sum(diff * diff, axis=-1)
	soft_dist = np.sqrt(sq_dist + eps_s * eps_s)

	if use_phi:
		phi = -G_CONST * MASS * np.sum(off_diag / soft_dist, axis=-1)
		self_dynamic_term = a_b * positions * (1.0 + phi[..., None])
	else:
		self_dynamic_term = a_b * positions * 1.0

	if use_boundary:
		r2 = np.sum(positions * positions, axis=-1)
		boundary = 1.0 - r2 / r2_b
		boundary_term = boundary[..., None] * positions
	else:
		boundary_term = 0.0 * positions

	if use_force:
		inv_r3 = off_diag / (soft_dist ** 3)
		force_term = G_CONST * MASS * MASS * np.einsum("...ij,...ijk->...ik", inv_r3, -diff)
	else:
		force_term = 0.0 * positions

	return self_dynamic_term + boundary_term + force_term


def map_step_vec(state, a_param, use_phi=True, use_force=True, use_boundary=True, fold="sin", softening=None, r2_boundary=None):
	"""One synchronous (Jacobi) iteration for a state of shape (9,) or (M,9).

	`a_param`, `softening` and `r2_boundary` may be scalars; `a_param` may also be an (M,) array
	so that a parameter scan runs as one batch.
	"""
	positions = state.reshape(state.shape[:-1] + (3, 3))
	denominator = map_denominators(positions, a_param, use_phi, use_force, use_boundary, softening, r2_boundary)
	small = np.abs(denominator) < CLAMP_THRESHOLD
	if small.any():
		clamp_counter["count"] += int(small.sum())
		denominator = np.where(small, np.where(denominator >= 0.0, CLAMP_THRESHOLD, -CLAMP_THRESHOLD), denominator)
	new_positions = FOLDS[fold](G_CONST / denominator)
	return new_positions.reshape(state.shape)


def iterate_map_vec(initial_state, a_param, n_iterations, record_every=1, **map_kwargs):
	"""Trajectory of shape (n_records, *state.shape) using map_step_vec; keeps every `record_every`-th state."""
	state = np.array(initial_state, dtype=float)
	n_records = n_iterations // record_every + 1
	trajectory = np.empty((n_records,) + state.shape)
	trajectory[0] = state
	record_index = 1
	for iteration_index in range(1, n_iterations + 1):
		state = map_step_vec(state, a_param, **map_kwargs)
		if iteration_index % record_every == 0:
			trajectory[record_index] = state
			record_index += 1
	return trajectory


def advance(state, a_param, n_iterations, **map_kwargs):
	"""Advance a state (9,) or batch (M,9) by n_iterations without storing the trajectory."""
	state = np.array(state, dtype=float)
	for _ in range(n_iterations):
		state = map_step_vec(state, a_param, **map_kwargs)
	return state


def sample_initial_states(n_states, radius, seed, center=None):
	"""Initial states drawn uniformly in [-radius, radius]^9 around `center` (default: DEFAULT_INITIAL_STATE)."""
	base = DEFAULT_INITIAL_STATE if center is None else np.asarray(center, dtype=float)
	rng = np.random.default_rng(seed)
	return base[None, :] + rng.uniform(-radius, radius, size=(n_states, 9))


if __name__ == "__main__":
	traj = iterate_map(DEFAULT_INITIAL_STATE, 0.97, 20)
	print(traj[:5])
	print("final state:", traj[-1])
