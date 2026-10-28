"""
Post-processing (whitening) methods for the raw scalar output of the bounded
gravitational chaotic map, aimed at removing the short-range sequential
correlation identified by the full NIST SP 800-22 battery (see nist_full_suite.py).

Two candidate methods are implemented:
	- "xor3": median-threshold each of 3 independent state coordinates
	  (one per body) and XOR the resulting bits.
	- "lsb_xor3": extract a low-order IEEE-754 mantissa bit from each of the
	  same 3 coordinates and XOR them (targets the fact that the map's
	  most-significant digits vary smoothly under sin(.), while low-order
	  mantissa bits are far more sensitive to the chaotic dynamics).

tab-indented, snake_case, Allman braces.
"""

import numpy as np


def _median_threshold_bit(column):
	median_value = np.median(column)
	return (column >= median_value).astype(np.int64)


def _mantissa_bit(column, bit_position=30):
	"""Extract bit `bit_position` (0 = LSB) of the IEEE-754 mantissa of each value."""
	raw = column.astype(np.float64).view(np.uint64)
	return ((raw >> np.uint64(bit_position)) & np.uint64(1)).astype(np.int64)


def postprocess_bitstream(trajectory, method="xor3", coord_indices=(0, 4, 8), bit_position=30):
	"""trajectory: (n, 9) array from gravity_map.iterate_map. Returns a 1D bit array."""
	columns = [trajectory[:, idx] for idx in coord_indices]

	if method == "xor3":
		bits = [_median_threshold_bit(col) for col in columns]
	elif method == "lsb_xor3":
		bits = [_mantissa_bit(col, bit_position) for col in columns]
	else:
		raise ValueError(f"unknown method: {method}")

	combined = bits[0]
	for b in bits[1:]:
		combined = combined ^ b
	return combined
