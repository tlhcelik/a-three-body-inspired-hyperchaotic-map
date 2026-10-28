"""
Systematic divergence scan for the sin(.)-ablated map: for a broad range of
control parameter a (matching the original paper's initial exploratory range
a in [-25,25] and the finer Lyapunov-scan range a in [-6,8]), record how many
iterations the trajectory survives before diverging, starting from the
paper's DEFAULT_INITIAL_STATE. A second scan repeats this at the two paper
reference parameters (a=0.97, a=5.2) over many random nearby initial
conditions, to check whether divergence is specific to the default initial
state or a generic property of the ablated map.

tab-indented, snake_case, Allman braces.
"""

import numpy as np
from gravity_map_nosin import iterate_map, DEFAULT_INITIAL_STATE, R2_BOUNDARY

n_cap = 5000


def survival_iterations(a_param, initial_state, n_cap=n_cap):
	_, div_at = iterate_map(initial_state, a_param, n_cap)
	if div_at is None:
		return n_cap
	return div_at


if __name__ == "__main__":
	print("=== Genis tarama: a in [-25,25], adim=0.5, varsayilan baslangic durumu ===")
	a_values_wide = np.arange(-25, 25.01, 0.5)
	survival_wide = np.array([survival_iterations(a, DEFAULT_INITIAL_STATE) for a in a_values_wide])

	n_survived_cap = np.sum(survival_wide >= n_cap)
	print(f"Toplam a degeri: {len(a_values_wide)}")
	print(f"Cap'e ({n_cap} iter) ulasan (sinirli kalan) a sayisi: {n_survived_cap}")
	print(f"Ortalama hayatta kalma iterasyonu: {np.mean(survival_wide[survival_wide < n_cap]):.1f}")
	print(f"Medyan hayatta kalma iterasyonu: {np.median(survival_wide[survival_wide < n_cap]):.1f}")
	print(f"Min hayatta kalma: {np.min(survival_wide)}   Max (cap haric): {np.max(survival_wide[survival_wide < n_cap]) if np.any(survival_wide < n_cap) else 'N/A'}")

	if n_survived_cap > 0:
		print("Sinirli kalan a degerleri:")
		for a, s in zip(a_values_wide, survival_wide):
			if s >= n_cap:
				print(f"  a={a:+.2f}  survived>={n_cap}")

	np.savez("divergence_scan_wide.npz", a_values=a_values_wide, survival=survival_wide)

	print()
	print("=== Ince tarama: a in [-6,8], adim=0.1 (orijinal Lyapunov tarama araligi) ===")
	a_values_fine = np.arange(-6, 8.01, 0.1)
	survival_fine = np.array([survival_iterations(a, DEFAULT_INITIAL_STATE) for a in a_values_fine])
	n_survived_cap_fine = np.sum(survival_fine >= n_cap)
	print(f"Toplam a degeri: {len(a_values_fine)}  Cap'e ulasan: {n_survived_cap_fine}")
	print(f"Ortalama hayatta kalma (cap haric): {np.mean(survival_fine[survival_fine < n_cap]):.1f}")
	np.savez("divergence_scan_fine.npz", a_values=a_values_fine, survival=survival_fine)

	print()
	print("=== Referans noktalarinda rastgele baslangic kosullariyla dogrulama (n=100 deneme) ===")
	rng = np.random.default_rng(42)
	for a_ref in [0.97, 5.2]:
		survivals = []
		for _ in range(100):
			perturb = rng.uniform(-0.3, 0.3, 9)
			init = DEFAULT_INITIAL_STATE + perturb
			survivals.append(survival_iterations(a_ref, init))
		survivals = np.array(survivals)
		n_bounded = np.sum(survivals >= n_cap)
		print(f"a={a_ref}: {n_bounded}/100 baslangic kosulu sinirli kaldi; ortalama hayatta kalma (digerleri)={np.mean(survivals[survivals<n_cap]) if n_bounded<100 else float('nan'):.1f}  min={np.min(survivals)}  max={np.max(survivals)}")
