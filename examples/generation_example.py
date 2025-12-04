#!/usr/bin/env python
"""
Example: Random Field Generation

Demonstrates how to generate periodic Gaussian random fields with different
spectral characteristics:
1. Self-affine spectrum (power-law)
2. Matérn covariance spectrum
3. Filtered white noise vs. ideal spectrum methods (noise parameter)

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL
License: BSD-3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt

from rfgen import selfaffine_field, matern_field


def plot_field(field, title, ax=None, cmap="RdYlBu_r"):
    """Helper function to plot a 2D field."""
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(field, cmap=cmap, interpolation="bicubic")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def main():
    # Parameters
    N = 512  # Grid size
    k_low = 8 / N  # Lower cutoff wavenumber
    k_high = 128 / N  # Upper cutoff wavenumber
    Hurst = 0.7  # Hurst exponent for self-affine
    seed = 42  # Random seed for reproducibility

    # Create random number generator
    rng = np.random.default_rng(seed)

    print("Generating random fields...")
    print(f"  Grid size: {N}x{N}")
    print(f"  k_low = {k_low:.4f}, k_high = {k_high:.4f}")
    print(f"  Hurst exponent: {Hurst}")

    # --- Self-affine fields ---
    # Method 1: Filtered white noise (noise=True, default)
    field_noise = selfaffine_field(
        dim=2, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, noise=True, rng=rng
    )
    field_noise /= np.std(field_noise)  # Normalize

    # Method 2: Ideal spectrum with random phase (noise=False)
    rng2 = np.random.default_rng(seed)  # Reset RNG
    field_ideal = selfaffine_field(
        dim=2, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, noise=False, rng=rng2
    )
    field_ideal /= np.std(field_ideal)

    # With plateau (roll-off at low frequencies)
    rng3 = np.random.default_rng(seed)
    field_plateau = selfaffine_field(
        dim=2, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, plateau=True, rng=rng3
    )
    field_plateau /= np.std(field_plateau)

    # --- Matérn fields ---
    rng4 = np.random.default_rng(seed)
    field_matern_noise = matern_field(
        dim=2,
        N=N,
        nu=1.5,  # Smoothness parameter
        correlation_length=0.05,
        sigma=1.0,
        k_low=k_low,
        k_high=k_high,
        noise=True,  # With spectral noise
        rng=rng4,
    )
    field_matern_noise /= np.std(field_matern_noise)

    rng5 = np.random.default_rng(seed)
    field_matern_ideal = matern_field(
        dim=2,
        N=N,
        nu=1.5,
        correlation_length=0.05,
        sigma=1.0,
        k_low=k_low,
        k_high=k_high,
        noise=False,  # Ideal spectrum
        rng=rng5,
    )
    field_matern_ideal /= np.std(field_matern_ideal)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Self-affine fields
    im1 = plot_field(field_noise, f"Self-affine (noise=True)\nH = {Hurst}", axes[0, 0])
    im2 = plot_field(field_ideal, f"Self-affine (noise=False)\nH = {Hurst}", axes[0, 1])
    im3 = plot_field(field_plateau, f"Self-affine (with plateau)\nH = {Hurst}", axes[0, 2])

    # Matérn fields
    im4 = plot_field(field_matern_noise, "Matérn (noise=True)\nν = 1.5", axes[1, 0])
    im5 = plot_field(field_matern_ideal, "Matérn (noise=False)\nν = 1.5", axes[1, 1])

    # Hide empty subplot
    axes[1, 2].axis("off")

    # Add colorbars
    for ax, im in zip(axes.flat[:5], [im1, im2, im3, im4, im5]):
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    plt.suptitle("Periodic Gaussian Random Fields", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("random_fields_comparison.png", dpi=150)
    plt.show()

    print("\nFields generated successfully!")
    print("Figure saved as 'random_fields_comparison.png'")


if __name__ == "__main__":
    main()
