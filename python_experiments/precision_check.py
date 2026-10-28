"""
Finite-precision checks (reviewer items R10, R16-adjacent).

	validity_horizon   max |x_double - x_hp| per step from the default state, hp = 60-digit decimal
	lambda_precision   two-trajectory lambda_max in 60-digit arithmetic vs double, same states/lengths
	cycle_length       Brent cycle detection in float64 for several keys, up to 2^CYCLE_POWER iterations

The high-precision map uses Python's decimal module (no third-party dependency): softened distances via
Decimal.sqrt and sin via argument reduction modulo 2*pi plus a Taylor series.

Output: results/precision_check.json

tab-indented, snake_case, Allman braces.
"""

import sys
from decimal import Decimal, getcontext
import numpy as np

import settings as st
import gravity_map as gm
from gravity_map import map_step_vec, advance, sample_initial_states, DEFAULT_INITIAL_STATE
from lyapunov import two_traj_batch
from analysis_utils import save_json, summary_stats

DIGITS = 60
CYCLE_POWER = 22


def decimal_pi():
	"""pi to the current decimal precision (recipe from the Python decimal documentation)."""
	getcontext().prec += 2
	three = Decimal(3)
	lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
	while s != lasts:
		lasts = s
		n, na = n + na, na + 8
		d, da = d + da, da + 32
		t = (t * n) / d
		s += t
	getcontext().prec -= 2
	return +s


def make_decimal_map(a_param, softening, r2_boundary):
	getcontext().prec = DIGITS
	pi = decimal_pi()
	two_pi = 2 * pi
	half_pi = pi / 2
	eps2 = Decimal(repr(softening)) ** 2
	r2_b = Decimal(repr(r2_boundary))
	a_d = Decimal(repr(a_param))
	one = Decimal(1)
	tiny = Decimal("1e-%d" % (DIGITS - 2))

	def dsin(x):
		k = (x / two_pi).to_integral_value()
		r = x - k * two_pi
		if r > pi:
			r -= two_pi
		if r < -pi:
			r += two_pi
		if r > half_pi:
			r = pi - r
		elif r < -half_pi:
			r = -pi - r
		term = r
		total = r
		r2 = r * r
		n = 1
		while abs(term) > tiny:
			term = -term * r2 / ((2 * n) * (2 * n + 1))
			total += term
			n += 1
		return total

	def step(state):
		p = [state[0:3], state[3:6], state[6:9]]
		dist = [[None] * 3 for _ in range(3)]
		for i in range(3):
			for j in range(3):
				if i != j:
					dx = [p[i][k] - p[j][k] for k in range(3)]
					dist[i][j] = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + eps2).sqrt()
		new = []
		for i in range(3):
			phi = -sum(one / dist[i][j] for j in range(3) if j != i)
			r_sq = p[i][0] * p[i][0] + p[i][1] * p[i][1] + p[i][2] * p[i][2]
			boundary = one - r_sq / r2_b
			for k in range(3):
				force = sum((p[j][k] - p[i][k]) / (dist[i][j] ** 3) for j in range(3) if j != i)
				denom = a_d * p[i][k] * (one + phi) + boundary * p[i][k] + force
				new.append(dsin(one / denom))
		return new

	return step


def to_decimal(state):
	return [Decimal(repr(float(v))) for v in state]


def validity_horizon(a_param, n_steps):
	step = make_decimal_map(a_param, gm.SOFTENING, gm.R2_BOUNDARY)
	hp = to_decimal(DEFAULT_INITIAL_STATE)
	dp = DEFAULT_INITIAL_STATE.copy()
	errors = []
	for _ in range(n_steps):
		hp = step(hp)
		dp = map_step_vec(dp, a_param)
		errors.append(float(np.max(np.abs(dp - np.array([float(v) for v in hp])))))
	return errors


def lambda_high_precision(a_param, initial_state, n_transient, n_iter, delta0="1e-25"):
	step = make_decimal_map(a_param, gm.SOFTENING, gm.R2_BOUNDARY)
	state_a = to_decimal(initial_state)
	for _ in range(n_transient):
		state_a = step(state_a)
	d0 = Decimal(delta0)
	state_b = list(state_a)
	state_b[0] = state_b[0] + d0
	log_sum = 0.0
	for _ in range(n_iter):
		state_a = step(state_a)
		state_b = step(state_b)
		diff = [b - a for a, b in zip(state_a, state_b)]
		dist = sum(d * d for d in diff).sqrt()
		log_sum += float((dist / d0).ln())
		state_b = [a + d * d0 / dist for a, d in zip(state_a, diff)]
	return log_sum / n_iter


def brent_cycle_lengths(states, a_param, max_power):
	"""Vectorised Brent: returns cycle length per trajectory or 0 if none found within 2**max_power steps."""
	n_traj = states.shape[0]
	power = np.ones(n_traj, dtype=np.int64)
	lam = np.ones(n_traj, dtype=np.int64)
	found = np.zeros(n_traj, dtype=np.int64)
	tortoise = states.copy()
	hare = map_step_vec(states, a_param)
	for _ in range(2 ** max_power):
		equal = np.all(hare == tortoise, axis=1) & (found == 0)
		found[equal] = lam[equal]
		if (found > 0).all():
			break
		reset = (power == lam) & (found == 0)
		tortoise[reset] = hare[reset]
		power[reset] *= 2
		lam[reset] = 0
		hare = map_step_vec(hare, a_param)
		lam += 1
	return found


def main(quick=False):
	out = {"digits": DIGITS}
	horizon_steps = 15 if quick else 40
	out["validity_horizon"] = {}
	for a_param in (st.A_CHAOTIC, -1.0, 2.0):
		errs = validity_horizon(a_param, horizon_steps)
		first_large = next((i + 1 for i, e in enumerate(errs) if e > 1.0e-2), None)
		out["validity_horizon"][str(a_param)] = {"max_abs_error_per_step": errs, "first_step_error_above_1e-2": first_large}
		print("a=%s horizon (err>1e-2) at n=%s" % (a_param, first_large))

	n_ic = 2 if quick else 5
	n_tr = 50 if quick else 200
	n_it = 150 if quick else 1000
	ics = sample_initial_states(n_ic, st.IC_RADIUS, st.SEED_IC)
	out["lambda_precision"] = {"n_transient": n_tr, "n_iter": n_it}
	for a_param in (st.A_CHAOTIC, -1.0):
		hp = [lambda_high_precision(a_param, ics[i], n_tr, n_it) for i in range(n_ic)]
		dp = two_traj_batch(ics, a_param, n_tr, n_it, st.DELTA0)
		out["lambda_precision"][str(a_param)] = {"high_precision": summary_stats(hp), "double": summary_stats(dp), "high_precision_per_ic": hp, "double_per_ic": dp}
		print("a=%s lambda hp=%.3f+-%.3f double=%.3f+-%.3f" % (a_param, np.mean(hp), np.std(hp), np.mean(dp), np.std(dp)))

	n_keys = 4 if quick else 16
	power = 12 if quick else CYCLE_POWER
	keys = advance(sample_initial_states(n_keys, st.IC_RADIUS, st.SEED_NIST), st.A_CHAOTIC, st.N_TRANSIENT)
	cycles = brent_cycle_lengths(keys, st.A_CHAOTIC, power)
	out["cycle_length"] = {"a": st.A_CHAOTIC, "n_keys": n_keys, "max_iterations": 2 ** power, "cycle_found_per_key": cycles, "keys_without_cycle": int((cycles == 0).sum())}
	print("cycles found:", cycles, "within", 2 ** power)
	print("->", save_json("precision_check_quick.json" if quick else "precision_check.json", out))


if __name__ == "__main__":
	main(quick="--quick" in sys.argv)
