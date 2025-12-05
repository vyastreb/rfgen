#!/usr/bin/env python
"""
Example: Random Field Analysis

Demonstrates the analysis tools available in the rfgen package:
1. Autocorrelation function computation
2. Power spectral density (PSD) analysis
3. Spectral moments and derived quantities

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL
License: BSD-3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt

from rfgen import (
    # Generation
    selfaffine_field,
    # Autocorrelation
    autocorrelation_1d,
    autocorrelation_2d,
    correlation_length,
    integral_correlation_length,
    # Spectrum
    psd_1d,
    psd_radial_average,
    fit_power_law,
    estimate_hurst_exponent,
    # Moments
    compute_standard_moments,
    nayak_parameter,
    rms_quantities,
)


def main():
    # Parameters
    N = 1024
    # Wavenumber bounds in normalized units (0 to 0.5 = Nyquist)
    k_low_norm = 4 / N      # Normalized k_low
    k_high_norm = 256 / N   # Normalized k_high
    Hurst = 0.7
    seed = 42

    rng = np.random.default_rng(seed)
    spacing = 1.0 / N  # Physical spacing (domain size = 1)

    # Convert to physical wavenumbers for PSD analysis
    # Physical k = normalized_k * N (when spacing = 1/N)
    k_low_phys = k_low_norm * N   # = 4
    k_high_phys = k_high_norm * N  # = 256

    print("=" * 60)
    print("Random Field Analysis Example")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Grid size: {N}x{N}")
    print(f"  Hurst exponent (input): {Hurst}")
    print(f"  k_low = {k_low_norm:.4f} (normalized), {k_low_phys:.0f} (physical)")
    print(f"  k_high = {k_high_norm:.4f} (normalized), {k_high_phys:.0f} (physical)")

    # Generate field (uses normalized k values)
    print("\nGenerating random field...")
    field = selfaffine_field(
        dim=2, N=N, Hurst=Hurst, k_low=k_low_norm, k_high=k_high_norm, rng=rng
    )
    field -= np.mean(field)
    field /= np.std(field)

    # Extract a 1D profile
    profile = field[N // 2, :]

    # --- Autocorrelation Analysis ---
    print("\n--- Autocorrelation Analysis ---")

    # 1D autocorrelation
    R_1d = autocorrelation_1d(profile)
    l_corr = correlation_length(R_1d, threshold=0.0, spacing=spacing)
    l_int = integral_correlation_length(R_1d, spacing=spacing)

    print(f"  Correlation length (1D profile): {l_corr:.4f}")
    print(f"  Integral correlation length: {l_int:.4f}")

    # 2D autocorrelation
    R_2d = autocorrelation_2d(field)

    # --- Power Spectrum Analysis ---
    print("\n--- Power Spectrum Analysis ---")

    # 1D PSD
    k_1d, psd_1d_vals = psd_1d(profile, spacing=spacing)

    # Radially averaged 2D PSD (returns physical k values)
    k_radial, psd_radial = psd_radial_average(field, spacing=spacing)

    # Fit power law to estimate Hurst exponent (use physical k values)
    H_est, r_sq = estimate_hurst_exponent(field, k_low=k_low_phys, k_high=k_high_phys, spacing=spacing)
    print(f"  Estimated Hurst exponent: {H_est:.3f}")
    print(f"  R² of power-law fit: {r_sq:.4f}")

    # Fit PSD directly (use physical k values)
    A, beta, r_sq2 = fit_power_law(k_radial, psd_radial, k_min=k_low_phys, k_max=k_high_phys)
    print(f"  PSD power-law exponent β: {beta:.3f}")
    print(f"  (Expected: β = dim + 2H = {2 + 2*Hurst:.3f})")

    # --- Spectral Moments ---
    print("\n--- Spectral Moments ---")

    moments = compute_standard_moments(field, spacing=spacing)
    print("  Standard moments:")
    for name, value in moments.items():
        print(f"    {name}: {value:.6e}")

    alpha = nayak_parameter(field, spacing=spacing)
    print(f"\n  Nayak's bandwidth parameter α: {alpha:.3f}")

    rms = rms_quantities(field, spacing=spacing)
    print("\n  RMS quantities:")
    for name, value in rms.items():
        print(f"    {name}: {value:.6e}")

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 10))

    # 1. Field
    ax1 = fig.add_subplot(2, 3, 1)
    im = ax1.imshow(field, cmap="RdYlBu_r", interpolation="bicubic")
    ax1.set_title(f"Random Field (H = {Hurst})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 2. 1D Profile
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.linspace(0, 1, N)
    ax2.plot(x, profile, "b-", lw=0.5)
    ax2.set_xlabel("x / L")
    ax2.set_ylabel("z / σ")
    ax2.set_title("1D Profile")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    # 3. 1D Autocorrelation
    ax3 = fig.add_subplot(2, 3, 3)
    dx = np.linspace(0, 0.5, N // 2)
    ax3.plot(dx, R_1d[: N // 2], "b-", lw=1.5)
    ax3.axhline(0, color="k", ls="--", lw=0.5)
    ax3.axvline(l_corr, color="r", ls="--", lw=1, label=f"l* = {l_corr:.3f}")
    ax3.set_xlabel("Δx / L")
    ax3.set_ylabel("R(Δx)")
    ax3.set_title("Autocorrelation Function (1D)")
    ax3.set_xlim(0, 0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 2D Autocorrelation
    ax4 = fig.add_subplot(2, 3, 4)
    extent = [0, 0.3, 0, 0.3]
    n_show = int(0.3 * N)
    im4 = ax4.imshow(
        R_2d[:n_show, :n_show],
        cmap="RdYlBu_r",
        interpolation="bicubic",
        extent=extent,
        origin="lower",
    )
    ax4.set_xlabel("Δx / L")
    ax4.set_ylabel("Δy / L")
    ax4.set_title("2D Autocorrelation")
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # 5. Radial PSD (log-log)
    ax5 = fig.add_subplot(2, 3, 5)
    # Filter to fitting range for plotting
    fit_mask = (k_radial >= k_low_phys) & (k_radial <= k_high_phys) & (psd_radial > 0)
    psd_in_range = psd_radial[fit_mask]
    # Set y-limits based on actual PSD values in the fitting range
    ymin = 0.5 * psd_in_range.min()
    ymax = 2.0 * psd_in_range.max()
    ax5.set_ylim(ymin, ymax)
    ax5.loglog(k_radial[fit_mask], psd_radial[fit_mask], "b.", ms=3, alpha=0.5, label="Data")
    # Plot fit
    k_fit = np.logspace(np.log10(k_low_phys), np.log10(k_high_phys), 100)
    psd_fit = A * k_fit ** (-beta)
    ax5.loglog(k_fit, psd_fit, "r-", lw=2, label=f"Fit: k^{{-{beta:.2f}}}",alpha=0.5)
    ax5.axvline(k_low_phys, color="g", ls=":", label=f"k_low = {k_low_phys:.0f}")
    ax5.axvline(k_high_phys, color="g", ls=":")
    ax5.set_xlabel("k")
    ax5.set_ylabel("PSD(k)")
    ax5.set_title("Radially Averaged PSD")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, which="both")
    # Set x-axis limits to fitting range with some margin
    ax5.set_xlim(k_low_phys * 0.8, k_high_phys * 1.2)

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary = [
        f"Input Hurst: {Hurst:.2f}",
        f"Estimated Hurst: {H_est:.3f}",
        f"PSD exponent β: {beta:.3f}",
        f"Correlation length: {l_corr:.4f}",
        f"Bandwidth α: {alpha:.3f}",
        "",
        f"RMS height: {rms['rms_height']:.4f}",
        f"RMS slope: {rms['rms_slope']:.4f}",
        f"RMS curvature: {rms['rms_curvature']:.4f}",
    ]
    ax6.text(
        0.1,
        0.9,
        "\n".join(summary),
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax6.set_title("Summary Statistics")

    plt.tight_layout()
    plt.savefig("random_field_analysis.png", dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("Analysis complete! Figure saved as 'random_field_analysis.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
