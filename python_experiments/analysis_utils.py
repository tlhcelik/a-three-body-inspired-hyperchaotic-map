"""
Shared analysis utilities for the revision scripts.

	zero_one_test_gm        0-1 test with the standard Gottwald-Melbourne (2009) layout:
	                        M_c(n) for n <= n_cut = N/10, averaged over N - n_cut points,
	                        c drawn from (pi/5, 4*pi/5). (The original zero_one_test.py swapped
	                        the roles of n and the averaging window.)
	permutation_entropy_vec vectorised Bandt-Pompe entropy with an explicit tie policy
	correlation_integral    Grassberger-Procaccia C(r) via a KD-tree, with local slopes
	detect_period           exact/near-exact period of the tail of a trajectory
	wilson_interval         binomial confidence interval
	save_json               numpy-safe JSON writer into results/

tab-indented, snake_case, Allman braces.
"""

import json
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

import settings as st


def _to_builtin(obj):
	"""Recursively convert numpy types for JSON."""
	if isinstance(obj, dict):
		return {str(k): _to_builtin(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_to_builtin(v) for v in obj]
	if isinstance(obj, np.ndarray):
		return _to_builtin(obj.tolist())
	if isinstance(obj, (np.floating,)):
		value = float(obj)
		return None if not math.isfinite(value) else value
	if isinstance(obj, float):
		return None if not math.isfinite(obj) else obj
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.bool_,)):
		return bool(obj)
	return obj


def save_json(name, payload):
	"""Write `payload` to results/<name> and return the path."""
	path = st.result_path(name)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(_to_builtin(payload), f, indent=2, ensure_ascii=False)
	return path


def load_json(name):
	"""Read results/<name>."""
	with open(st.result_path(name), encoding="utf-8") as f:
		return json.load(f)


def summary_stats(values):
	"""Mean, sd, median, quartiles, min, max of finite values."""
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return {"n": 0}
	return {
		"n": int(arr.size),
		"mean": float(arr.mean()),
		"sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
		"median": float(np.median(arr)),
		"q1": float(np.percentile(arr, 25)),
		"q3": float(np.percentile(arr, 75)),
		"min": float(arr.min()),
		"max": float(arr.max()),
	}


def zero_one_test_gm(series, n_c=50, seed=42, n_cut_fraction=0.1):
	"""Median K and all K_c of the 0-1 test (correlation method, oscillatory term removed)."""
	x = np.asarray(series, dtype=float)
	rng = np.random.default_rng(seed)
	n_total = x.shape[0]
	n_cut = max(10, int(n_total * n_cut_fraction))
	window = n_total - n_cut
	j = np.arange(1, n_total + 1)
	n_axis = np.arange(1, n_cut + 1)
	mean_x = x.mean()
	k_values = np.empty(n_c)
	c_values = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0, size=n_c)
	for c_index, c in enumerate(c_values):
		p = np.concatenate(([0.0], np.cumsum(x * np.cos(j * c))))
		q = np.concatenate(([0.0], np.cumsum(x * np.sin(j * c))))
		msd = np.empty(n_cut)
		for i, n in enumerate(n_axis):
			dp = p[n:n + window] - p[:window]
			dq = q[n:n + window] - q[:window]
			msd[i] = np.mean(dp * dp + dq * dq)
		d_c = msd - mean_x * mean_x * (1.0 - np.cos(n_axis * c)) / (1.0 - np.cos(c))
		if np.std(d_c) < 1.0e-15:
			k_values[c_index] = 0.0
		else:
			k_values[c_index] = np.corrcoef(n_axis, d_c)[0, 1]
	return float(np.median(k_values)), k_values, c_values


def zero_one_pq(series, c):
	"""Translation variables p(n), q(n) for one c (for the (p,q) plane figure)."""
	x = np.asarray(series, dtype=float)
	j = np.arange(1, x.shape[0] + 1)
	return np.cumsum(x * np.cos(j * c)), np.cumsum(x * np.sin(j * c))


def permutation_entropy_vec(series, m=7, delay=1, round_decimals=None):
	"""Bandt-Pompe permutation entropy.

	Tie policy: values are rounded to `round_decimals` (if given) and ranked with a stable sort,
	so equal values always produce the same ordinal pattern. Returns a dict with the normalised
	entropy, the entropy rate in nats/iteration (H/(m-1)), its ceiling ln(m!)/(m-1), and the number
	of distinct patterns observed.
	"""
	x = np.asarray(series, dtype=float)
	if round_decimals is not None:
		x = np.round(x, round_decimals)
	windows = sliding_window_view(x, (m - 1) * delay + 1)[:, ::delay]
	ranks = np.argsort(windows, axis=1, kind="stable")
	codes = np.zeros(ranks.shape[0], dtype=np.int64)
	for col in range(m):
		codes = codes * m + ranks[:, col]
	_, counts = np.unique(codes, return_counts=True)
	prob = counts / counts.sum()
	raw = float(-np.sum(prob * np.log(prob)))
	log_mfact = math.log(math.factorial(m))
	return {
		"normalized": raw / log_mfact,
		"entropy_nats": raw,
		"rate_nats_per_iter": raw / (m - 1),
		"rate_ceiling": log_mfact / (m - 1),
		"patterns_observed": int(counts.size),
		"patterns_possible": math.factorial(m),
		"n_windows": int(ranks.shape[0]),
	}


def shannon_entropy_bits(series, n_bins=256, value_range=(-1.0, 1.0)):
	"""Shannon entropy (bits) of a binned series and its normalised value."""
	hist, _ = np.histogram(series, bins=n_bins, range=value_range)
	prob = hist / hist.sum()
	prob = prob[prob > 0]
	bits = float(-np.sum(prob * np.log2(prob)))
	return bits, bits / math.log2(n_bins)


def correlation_integral(points, n_radii=40, r_min_quantile=1.0e-4, r_max_quantile=0.5, seed=7, n_ref_pairs=200000):
	"""C(r) on log-spaced radii and local slopes d ln C / d ln r.

	Radii span from the r_min_quantile to the r_max_quantile of a random sample of pair distances.
	Pair counts exclude i == j.
	"""
	pts = np.asarray(points, dtype=float)
	n_points = pts.shape[0]
	rng = np.random.default_rng(seed)
	i_idx = rng.integers(0, n_points, n_ref_pairs)
	j_idx = rng.integers(0, n_points, n_ref_pairs)
	keep = i_idx != j_idx
	sample_d = np.linalg.norm(pts[i_idx[keep]] - pts[j_idx[keep]], axis=1)
	r_lo = max(np.quantile(sample_d, r_min_quantile), 1.0e-12)
	r_hi = np.quantile(sample_d, r_max_quantile)
	radii = np.logspace(np.log10(r_lo), np.log10(r_hi), n_radii)
	tree = cKDTree(pts)
	counts = tree.count_neighbors(tree, radii) - n_points
	n_pairs = n_points * (n_points - 1)
	c_values = counts / n_pairs
	valid = c_values > 0
	log_r = np.log(radii[valid])
	log_c = np.log(c_values[valid])
	slopes = np.gradient(log_c, log_r) if log_r.size > 2 else np.array([])
	return {"radii": radii[valid], "c": c_values[valid], "log_r": log_r, "log_c": log_c, "local_slopes": slopes, "n_points": n_points}


def plateau_estimate(local_slopes, c_values, width=6, c_min=1.0e-5, c_max=1.0e-1):
	"""D2 estimate as the mean local slope over the flattest run of `width` radii with c_min <= C <= c_max."""
	slopes = np.asarray(local_slopes)
	cvals = np.asarray(c_values)
	allowed = (cvals >= c_min) & (cvals <= c_max)
	best = None
	for start in range(0, slopes.size - width + 1):
		sl = slice(start, start + width)
		if not allowed[sl].all():
			continue
		spread = float(np.std(slopes[sl]))
		if best is None or spread < best[0]:
			best = (spread, start)
	if best is None:
		return {"d2": None, "spread": None, "window": None}
	spread, start = best
	return {"d2": float(np.mean(slopes[start:start + width])), "spread": spread, "window": [int(start), int(start + width)]}


def detect_period(tail, max_period=64, tol=1.0e-9):
	"""Smallest p <= max_period with max|x_{t+p} - x_t| < tol over the tail (shape (T, d)); None if aperiodic."""
	arr = np.asarray(tail, dtype=float)
	for p in range(1, max_period + 1):
		if arr.shape[0] <= p:
			break
		if np.max(np.abs(arr[p:] - arr[:-p])) < tol:
			return p
	return None


def min_pair_distance(states):
	"""Minimum raw (unsoftened) inter-body distance for a state (9,) or batch (M,9)."""
	pos = np.asarray(states).reshape(np.shape(states)[:-1] + (3, 3))
	d01 = np.linalg.norm(pos[..., 0, :] - pos[..., 1, :], axis=-1)
	d02 = np.linalg.norm(pos[..., 0, :] - pos[..., 2, :], axis=-1)
	d12 = np.linalg.norm(pos[..., 1, :] - pos[..., 2, :], axis=-1)
	return np.minimum(np.minimum(d01, d02), d12)


def wilson_interval(k, n, z=1.96):
	"""Wilson score interval for a binomial proportion k/n."""
	if n == 0:
		return (float("nan"), float("nan"))
	phat = k / n
	denom = 1.0 + z * z / n
	centre = (phat + z * z / (2.0 * n)) / denom
	half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
	return (centre - half, centre + half)
