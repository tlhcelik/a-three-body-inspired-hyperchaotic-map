"""
Closed-form Jacobian of the bounded gravitational chaotic map (fold = sin).

For q'_{i,k} = sin(G / D_{i,k}) with
    D_{i,k} = a q_{i,k} (1 + Phi_i) + B_i q_{i,k} + F_{i,k},
    r~_ij   = sqrt(|p_i - p_j|^2 + eps_s^2),   d_ij = p_j - p_i,
the derivatives used are
    dPhi_i/dp_{j,c} = G m d_{ij,c} / r~_ij^3                    (j != i),  dPhi_i/dp_{i,c} = -sum_j of those
    dF_{i,k}/dp_{j,c} = G m^2 (delta_kc / r~^3 - 3 d_k d_c / r~^5) (j != i),  dF_{i,k}/dp_{i,c} = -sum_j of those
    dB_i/dp_{i,c} = -2 p_{i,c} / R^2
    dq'_{i,k}/dp_{l,c} = -cos(G/D) G / D^2 * dD_{i,k}/dp_{l,c}
Entries whose denominator is clamped (|D| < CLAMP_THRESHOLD) get zero derivative, matching the map.

tab-indented, snake_case, Allman braces.
"""

import numpy as np
import gravity_map as gm


def analytic_jacobian(state, a_param, use_phi=True, use_force=True, use_boundary=True, softening=None, r2_boundary=None):
	"""Exact 9x9 Jacobian of map_step_vec(fold='sin') at `state` (shape (9,))."""
	eps_s = gm.SOFTENING if softening is None else softening
	r2_b = gm.R2_BOUNDARY if r2_boundary is None else r2_boundary
	p = state.reshape(3, 3)
	eye3 = np.eye(3)
	off_diag = 1.0 - eye3

	d = p[None, :, :] - p[:, None, :]
	soft_dist = np.sqrt(np.sum(d * d, axis=-1) + eps_s * eps_s)
	inv_r = off_diag / soft_dist
	inv_r3 = off_diag / soft_dist ** 3
	inv_r5 = off_diag / soft_dist ** 5

	d_dens = np.zeros((3, 3, 3, 3))
	coef_self = np.full(3, a_param)

	if use_phi:
		phi = -gm.G_CONST * gm.MASS * np.sum(inv_r, axis=1)
		dphi = gm.G_CONST * gm.MASS * d * inv_r3[:, :, None]
		for i in range(3):
			dphi[i, i, :] = -np.sum(dphi[i, :, :], axis=0)
		coef_self = a_param * (1.0 + phi)
		d_dens += a_param * p[:, :, None, None] * dphi[:, None, :, :]
	else:
		phi = np.zeros(3)

	if use_boundary:
		boundary = 1.0 - np.sum(p * p, axis=1) / r2_b
		coef_self = coef_self + boundary
		dboundary = np.zeros((3, 3, 3))
		for i in range(3):
			dboundary[i, i, :] = -2.0 * p[i, :] / r2_b
		d_dens += p[:, :, None, None] * dboundary[:, None, :, :]
	else:
		boundary = np.zeros(3)

	if use_force:
		force = gm.G_CONST * gm.MASS * gm.MASS * np.einsum("ij,ijk->ik", inv_r3, d)
		pair = gm.G_CONST * gm.MASS * gm.MASS * (eye3[None, None, :, :] * inv_r3[:, :, None, None] - 3.0 * d[:, :, :, None] * d[:, :, None, :] * inv_r5[:, :, None, None])
		dforce = pair.copy()
		for i in range(3):
			dforce[i, i, :, :] = -np.sum(pair[i, :, :, :], axis=0)
		d_dens += np.transpose(dforce, (0, 2, 1, 3))
	else:
		force = np.zeros((3, 3))

	for i in range(3):
		for k in range(3):
			d_dens[i, k, i, k] += coef_self[i]

	self_term = (a_param * (1.0 + phi))[:, None] * p if use_phi else a_param * p
	dens = self_term + boundary[:, None] * p + force
	outer = -np.cos(gm.G_CONST / dens) * gm.G_CONST / (dens * dens)
	outer = np.where(np.abs(dens) < gm.CLAMP_THRESHOLD, 0.0, outer)

	return (outer[:, :, None, None] * d_dens).reshape(9, 9)
