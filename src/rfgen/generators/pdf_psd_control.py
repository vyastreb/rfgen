"""
Random field generator with prescribed PSD and PDF (IAAFT-like algorithm).

This module implements an iterative amplitude adjusted Fourier transform (IAAFT)
scheme to generate 1D/2D/3D periodic random fields that match simultaneously:

    (i)  a prescribed isotropic power spectral density (PSD) Φ(k),
    (ii) a prescribed one-point probability distribution (PDF) of the field.

The algorithm works by alternating two projections:

    1. Projection onto the set of fields with the target amplitude distribution
       (rank-order mapping to a fixed sorted sample drawn from the target PDF).
    2. Projection onto the set of fields with the target power spectrum
       (replacement of Fourier amplitudes by the target ones, keeping phases).

This is an extension of the IAAFT surrogate data method (Schreiber & Schmitz, 1996)
to arbitrary PSD given analytically as a function of |k|.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.fft import fftfreq, fftn, ifftn


def arbitrary_pdf_psd_field(
    dim: int,
    N: int,
    psd_func: Callable[[np.ndarray], np.ndarray],
    pdf_func: Callable[[np.ndarray], np.ndarray] | None = None,
    icdf_func: Callable[[np.ndarray], np.ndarray] | None = None,
    n_iters: int = 50,
    z_min: float = -5.0,
    z_max: float = 5.0,
    n_z: int = 10000,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Generate a periodic random field with prescribed PSD and PDF (IAAFT).

    Parameters
    ----------
    dim : int
        Spatial dimension (1, 2, or 3).
    N : int
        Number of grid points per dimension. The total number of points is N**dim.
    psd_func : callable
        Function Φ(k) giving the *radial* power spectral density as a function of
        the wavenumber magnitude k (0 <= k <= 0.5 in non-dimensional units
        consistent with :func:`numpy.fft.fftfreq`). It must accept a NumPy array
        and return an array of the same shape.
    pdf_func : callable, optional
        Probability density function p(z) of the target marginal distribution of
        the field values. Used only if ``icdf_func`` is not provided. It must
        accept a NumPy array of z-values and return p(z) ≥ 0 of the same shape.
    icdf_func : callable, optional
        Inverse cumulative distribution function (quantile function) of the
        target distribution. If provided, this is used directly to construct the
        target amplitude distribution. It must map u in [0, 1] to z-values.
        If ``icdf_func`` is None, it is built numerically from ``pdf_func`` via
        integration and interpolation.
    n_iters : int, optional
        Number of IAAFT iterations (alternating PSD and PDF projections).
        Typical values: 20–100. Default is 50.
    z_min, z_max : float, optional
        Bounds of the z-interval used to tabulate the PDF/CDF if ``pdf_func`` is
        provided. Default is [-5, 5]. These should cover essentially all the
        probability mass of the desired distribution.
    n_z : int, optional
        Number of points in the z-grid used to tabulate the PDF/CDF if
        ``pdf_func`` is provided. Default is 10000.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If None, uses NumPy's
        default RNG.
    verbose : bool, optional
        If True, print basic diagnostic information (iteration count, etc.).

    Returns
    -------
    z : ndarray
        Real-valued random field with shape (N,), (N, N), or (N, N, N),
        depending on ``dim``.

    Notes
    -----
    - The PSD and PDF are matched in an approximate fixed-point sense by the
      IAAFT iterations. In practice, a few tens of iterations are sufficient
      for good convergence.
    - The target PSD is assumed isotropic, Φ = Φ(|k|). If anisotropy is needed,
      the construction of the amplitude array must be modified accordingly.
    - The algorithm preserves zero mean if the k=0 amplitude of the PSD is zero.
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
    if N <= 0:
        raise ValueError("N must be positive")
    if psd_func is None:
        raise ValueError("psd_func must be provided")
    if icdf_func is None and pdf_func is None:
        raise ValueError("Either icdf_func or pdf_func must be provided")

    if rng is None:
        rng = np.random.default_rng()

    shape = (N,) if dim == 1 else (N,) * dim
    M = N**dim  # total number of points

    # -------------------------------------------------------------------------
    # 1) Build the Fourier-space radial wavenumber grid and target amplitudes
    # -------------------------------------------------------------------------
    k1 = fftfreq(N)  # in [-0.5, 0.5), consistent with numpy.fft
    if dim == 1:
        k = np.abs(k1)
    elif dim == 2:
        kx, ky = np.meshgrid(k1, k1, indexing="ij")
        k = np.sqrt(kx**2 + ky**2)
    else:  # dim == 3
        kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
        k = np.sqrt(kx**2 + ky**2 + kz**2)

    psd_vals = np.asarray(psd_func(k), dtype=float)
    if psd_vals.shape != k.shape:
        raise ValueError("psd_func(k) must return an array with the same shape as k")
    psd_vals = np.clip(psd_vals, a_min=0.0, a_max=None)

    # We work with amplitude sqrt(PSD)
    target_amp = np.sqrt(psd_vals)
    # Enforce real-valued field (no constant offset) if desired: zero out k=0
    target_amp[k == 0] = 0.0

    # -------------------------------------------------------------------------
    # 2) Build target amplitude distribution (sorted values) from PDF/ICDF
    # -------------------------------------------------------------------------
    if icdf_func is None:
        # Build ICDF numerically from the PDF via simple Riemann integration
        z_grid = np.linspace(z_min, z_max, n_z)
        pdf_vals = np.maximum(pdf_func(z_grid), 0.0)
        dz = z_grid[1] - z_grid[0]
        cdf_vals = np.cumsum(pdf_vals) * dz
        total = cdf_vals[-1]
        if not np.isfinite(total) or total <= 0:
            raise ValueError("PDF appears to have zero or infinite integral over [z_min, z_max]")
        cdf_vals /= total

        # Ensure strictly increasing CDF for interpolation
        eps = 1e-12
        cdf_vals = np.clip(cdf_vals, eps, 1.0 - eps)
        # Monotonic enforcement
        cdf_vals = np.maximum.accumulate(cdf_vals)

        def icdf_func(u: np.ndarray) -> np.ndarray:
            u = np.asarray(u, dtype=float)
            u = np.clip(u, eps, 1.0 - eps)
            return np.interp(u, cdf_vals, z_grid)

    # Fixed target sorted values according to the desired PDF
    u = (np.arange(M, dtype=float) + 0.5) / M  # mid-quantiles
    target_sorted = np.sort(icdf_func(u))

    # -------------------------------------------------------------------------
    # 3) Initialise field with target PSD (Gaussian, random phases)
    # -------------------------------------------------------------------------
    # Start from real Gaussian white noise
    w = rng.standard_normal(shape)
    W = fftn(w)
    phase = W / (np.abs(W) + 1e-30)  # complex unit-modulus phases
    F = target_amp * phase
    z = np.real(ifftn(F))

    # -------------------------------------------------------------------------
    # 4) IAAFT iterations: alternate PDF and PSD projections
    # -------------------------------------------------------------------------
    flat = z.reshape(M)

    for it in range(n_iters):
        # --- (a) Impose target amplitude distribution (PDF) ---
        # Rank-order mapping: sort current field, assign target_sorted
        order = np.argsort(flat)
        new_flat = np.empty_like(flat)
        new_flat[order] = target_sorted
        flat = new_flat

        # --- (b) Impose target PSD ---
        z = flat.reshape(shape)
        Z = fftn(z)
        phase = Z / (np.abs(Z) + 1e-30)
        F = target_amp * phase
        z = np.real(ifftn(F))
        flat = z.reshape(M)

        if verbose and ((it + 1) % 10 == 0 or it == n_iters - 1):
            var_current = float(np.var(flat))
            mean_current = float(np.mean(flat))
            print(f"IAAFT iter {it+1}/{n_iters}: mean = {mean_current:.3e}, var = {var_current:.3e}")

    # Final projection onto the target PDF (ensures marginal distribution)
    order = np.argsort(flat)
    new_flat = np.empty_like(flat)
    new_flat[order] = target_sorted
    flat = new_flat

    return flat.reshape(shape)
