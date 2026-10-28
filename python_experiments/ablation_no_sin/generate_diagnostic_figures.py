"""
Diagnostic figures for the sin(.)-ablation experiment.

tab-indented, snake_case, Allman braces.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravity_map_nosin import iterate_map, DEFAULT_INITIAL_STATE

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 150})


def fig_survival_vs_a():
	data_wide = np.load("divergence_scan_wide.npz")
	data_fine = np.load("divergence_scan_fine.npz")

	fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

	axes[0].plot(data_wide["a_values"], data_wide["survival"], marker="o", markersize=3, linewidth=0.8, color="#c0392b")
	axes[0].set_yscale("log")
	axes[0].set_xlabel("Kontrol parametresi $a$")
	axes[0].set_ylabel("Iraksama oncesi hayatta kalan iterasyon sayisi (log)")
	axes[0].set_title(r"Genis tarama $a\in[-25,25]$ (sin olmadan)")
	axes[0].axhline(5000, color="gray", linestyle="--", linewidth=0.8, label="cap (5000 iter)")
	axes[0].legend(fontsize=9)

	axes[1].plot(data_fine["a_values"], data_fine["survival"], marker="o", markersize=3, linewidth=0.8, color="#2c3e50")
	axes[1].set_yscale("log")
	axes[1].set_xlabel("Kontrol parametresi $a$")
	axes[1].set_ylabel("Iraksama oncesi hayatta kalan iterasyon sayisi (log)")
	axes[1].set_title(r"Ince tarama $a\in[-6,8]$ (orijinal Lyapunov araligi, sin olmadan)")
	axes[1].axhline(5000, color="gray", linestyle="--", linewidth=0.8)

	fig.tight_layout()
	fig.savefig("figures/fig_survival_vs_a.png")
	plt.close(fig)
	print("saved fig_survival_vs_a.png")


def fig_trajectory_examples():
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

	traj, div_at = iterate_map(DEFAULT_INITIAL_STATE, 0.97, 30)
	n_valid = div_at if div_at is not None else 30
	axes[0].plot(np.arange(n_valid), traj[:n_valid, 0], marker="o", markersize=3, color="#c0392b")
	axes[0].set_title(f"$a=0.97$ (sin olmadan): iraksama, iter={div_at}")
	axes[0].set_xlabel("Iterasyon $n$")
	axes[0].set_ylabel("$x_1$")

	traj2, div_at2 = iterate_map(DEFAULT_INITIAL_STATE, 5.2, 100)
	n_valid2 = div_at2 if div_at2 is not None else 100
	axes[1].plot(np.arange(n_valid2), traj2[:n_valid2, 0], marker="o", markersize=3, color="#8e44ad")
	axes[1].set_title(f"$a=5.2$ (sin olmadan): iraksama, iter={div_at2}")
	axes[1].set_xlabel("Iterasyon $n$")

	traj3, div_at3 = iterate_map(DEFAULT_INITIAL_STATE, 7.5, 40)
	axes[2].plot(np.arange(traj3.shape[0]), traj3[:, 0], marker="o", markersize=3, color="#1b7a3d")
	axes[2].set_title("$a=7.5$ (sin olmadan): periyot-2'ye yakinsama")
	axes[2].set_xlabel("Iterasyon $n$")

	fig.tight_layout()
	fig.savefig("figures/fig_trajectory_examples.png")
	plt.close(fig)
	print("saved fig_trajectory_examples.png")


if __name__ == "__main__":
	fig_survival_vs_a()
	fig_trajectory_examples()
