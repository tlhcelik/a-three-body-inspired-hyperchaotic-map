"""
Lyapunov exponent estimation for the bounded gravitational chaotic map.

Reference (unchanged): max_lyapunov_exponent_qr, max_lyapunov_exponent_two_traj (scalar map_step).
Revision additions:
	lyapunov_spectrum_qr      full 9-exponent spectrum (Benettin/QR), analytic or numerical Jacobian
	kaplan_yorke_dimension    D_KY from a spectrum
	two_traj_batch            largest exponent for a batch of initial states and/or a-values at once

tab-indented, snake_case, Allman braces.
"""

import numpy as np
from gravity_map import map_step, map_step_vec, DEFAULT_INITIAL_STATE
from jacobian import analytic_jacobian


def numerical_jacobian(state, a_param, step_size=1.0e-6):
	"""Central-difference Jacobian of the map at `state`, shape (9,9)."""
	dim = state.shape[0]
	jacobian = np.zeros((dim, dim))
	for column_index in range(dim):
		perturbation = np.zeros(dim)
		perturbation[column_index] = step_size
		forward = map_step(state + perturbation, a_param)
		backward = map_step(state - perturbation, a_param)
		jacobian[:, column_index] = (forward - backward) / (2.0 * step_size)
	return jacobian


def max_lyapunov_exponent_qr(initial_state, a_param, n_transient, n_iterations, jac_step=1.0e-6):
	"""Estimate the largest Lyapunov exponent using QR-decomposition based tangent-space evolution."""
	state = initial_state.copy()
	for _ in range(n_transient):
		state = map_step(state, a_param)

	dim = state.shape[0]
	q_matrix = np.eye(dim)
	log_sum = 0.0
	valid_steps = 0

	for _ in range(n_iterations):
		jacobian = numerical_jacobian(state, a_param, jac_step)
		state = map_step(state, a_param)
		product = jacobian @ q_matrix
		q_matrix, r_matrix = np.linalg.qr(product)
		diag_r = np.abs(np.diag(r_matrix))
		diag_r = np.clip(diag_r, 1.0e-300, None)
		log_sum += np.log(diag_r[0])
		valid_steps += 1

	return log_sum / valid_steps


def max_lyapunov_exponent_two_traj(initial_state, a_param, n_transient, n_iterations, delta0=1.0e-8):
	"""Estimate the largest Lyapunov exponent using the classical two-trajectory renormalization method
	(Wolf et al. 1985 style: perturb, iterate, measure divergence, renormalize).
	Renormalisation is applied after EVERY iteration."""
	state_a = initial_state.copy()
	for _ in range(n_transient):
		state_a = map_step(state_a, a_param)

	perturbation = np.zeros_like(state_a)
	perturbation[0] = delta0
	state_b = state_a + perturbation

	log_sum = 0.0
	for _ in range(n_iterations):
		state_a = map_step(state_a, a_param)
		state_b = map_step(state_b, a_param)
		diff = state_b - state_a
		dist = np.linalg.norm(diff)
		if dist < 1.0e-300:
			dist = 1.0e-300
		log_sum += np.log(dist / delta0)
		state_b = state_a + (diff / dist) * delta0

	return log_sum / n_iterations


def numerical_jacobian_vec(state, a_param, step_size=1.0e-6, **map_kwargs):
	"""Central-difference Jacobian using map_step_vec (all 18 evaluations in one batch)."""
	dim = state.shape[0]
	offsets = np.eye(dim) * step_size
	batch = np.concatenate([state[None, :] + offsets, state[None, :] - offsets], axis=0)
	images = map_step_vec(batch, a_param, **map_kwargs)
	return ((images[:dim] - images[dim:]) / (2.0 * step_size)).T


def lyapunov_spectrum_qr(initial_state, a_param, n_transient, n_iterations, jacobian="analytic", jac_step=1.0e-6, return_history=False, **map_kwargs):
	"""Full Lyapunov spectrum (descending) by QR re-orthonormalisation of the tangent map.

	jacobian = "analytic" uses jacobian.analytic_jacobian (fold must be sin);
	jacobian = "numerical" uses a central difference with step `jac_step`.
	With return_history=True also returns the running estimate of the spectrum every 100 steps.
	"""
	state = np.array(initial_state, dtype=float)
	for _ in range(n_transient):
		state = map_step_vec(state, a_param, **map_kwargs)

	dim = state.shape[0]
	q_matrix = np.eye(dim)
	log_sums = np.zeros(dim)
	history = []
	jac_kwargs = {k: v for k, v in map_kwargs.items() if k in ("use_phi", "use_force", "use_boundary", "softening", "r2_boundary")}

	for step_index in range(1, n_iterations + 1):
		if jacobian == "analytic":
			jac = analytic_jacobian(state, a_param, **jac_kwargs)
		else:
			jac = numerical_jacobian_vec(state, a_param, jac_step, **map_kwargs)
		state = map_step_vec(state, a_param, **map_kwargs)
		q_matrix, r_matrix = np.linalg.qr(jac @ q_matrix)
		log_sums += np.log(np.clip(np.abs(np.diag(r_matrix)), 1.0e-300, None))
		if return_history and step_index % 100 == 0:
			history.append(np.sort(log_sums / step_index)[::-1])

	spectrum = np.sort(log_sums / n_iterations)[::-1]
	if return_history:
		return spectrum, np.array(history)
	return spectrum


def kaplan_yorke_dimension(spectrum):
	"""Kaplan-Yorke dimension; returns len(spectrum) when the cumulative sum never turns negative."""
	lam = np.sort(np.asarray(spectrum, dtype=float))[::-1]
	cumulative = np.cumsum(lam)
	if cumulative[0] < 0.0:
		return 0.0
	non_negative = np.where(cumulative >= 0.0)[0]
	j = int(non_negative[-1])
	if j == len(lam) - 1:
		return float(len(lam))
	return float(j + 1 + cumulative[j] / abs(lam[j + 1]))


def two_traj_batch(initial_states, a_param, n_transient, n_iterations, delta0=1.0e-8, perturb_seed=None, **map_kwargs):
	"""Largest Lyapunov exponent for a batch of trajectories, shape (M,9) -> (M,).

	`a_param` may be a scalar or an (M,) array. The initial tangent direction is the first
	coordinate (as in the reference method) unless `perturb_seed` is given, in which case a
	random unit direction per trajectory is used. Renormalisation after every iteration.
	"""
	state_a = np.array(initial_states, dtype=float)
	for _ in range(n_transient):
		state_a = map_step_vec(state_a, a_param, **map_kwargs)

	n_traj, dim = state_a.shape
	if perturb_seed is None:
		direction = np.zeros((n_traj, dim))
		direction[:, 0] = 1.0
	else:
		rng = np.random.default_rng(perturb_seed)
		direction = rng.normal(size=(n_traj, dim))
		direction /= np.linalg.norm(direction, axis=1, keepdims=True)
	state_b = state_a + delta0 * direction

	log_sum = np.zeros(n_traj)
	for _ in range(n_iterations):
		state_a = map_step_vec(state_a, a_param, **map_kwargs)
		state_b = map_step_vec(state_b, a_param, **map_kwargs)
		diff = state_b - state_a
		dist = np.maximum(np.linalg.norm(diff, axis=1), 1.0e-300)
		log_sum += np.log(dist / delta0)
		state_b = state_a + diff * (delta0 / dist)[:, None]
		finite = np.all(np.isfinite(state_a), axis=1)
		if not finite.all():
			log_sum[~finite] = np.nan

	return log_sum / n_iterations


if __name__ == "__main__":
	lam_qr = max_lyapunov_exponent_qr(DEFAULT_INITIAL_STATE, 0.97, n_transient=500, n_iterations=2000)
	lam_two = max_lyapunov_exponent_two_traj(DEFAULT_INITIAL_STATE, 0.97, n_transient=500, n_iterations=2000)
	print("QR-based lambda_max (a=0.97):", lam_qr)
	print("Two-trajectory lambda_max (a=0.97):", lam_two)
