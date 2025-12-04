"""
Power spectral density computation for random fields.

Provides fast FFT-based computation of power spectral density (PSD)
for 1D profiles and 2D/3D fields, including radially averaged spectra.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

import numpy as np
from numpy.fft import fft, fft2, fftn, fftfreq, fftshift


def psd_1d(signal: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectral density of a 1D signal.

    Parameters
    ----------
    signal : ndarray
        1D input signal.
    spacing : float, optional
        Grid spacing (determines frequency units). Default is 1.0.

    Returns
    -------
    k : ndarray
        Wavenumber array (positive frequencies only).
    psd : ndarray
        Power spectral density at each wavenumber.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import psd_1d
    >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1024))
    >>> k, psd = psd_1d(signal, spacing=1/1024)
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {signal.shape}")

    N = len(signal)
    zhat = fft(signal)
    psd_full = np.abs(zhat) ** 2 / N**2

    # Get frequencies
    k_full = fftfreq(N, d=spacing)

    # Return only positive frequencies (excluding DC)
    pos_mask = k_full > 0
    k = k_full[pos_mask]
    psd = psd_full[pos_mask]

    return k, psd


def psd_2d(field: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectral density of a field.

    Parameters
    ----------
    field : ndarray
        2D input field with shape (Ny, Nx).
    spacing : float, optional
        Grid spacing (same in both directions). Default is 1.0.

    Returns
    -------
    k : ndarray
        2D wavenumber magnitude array.
    psd : ndarray
        2D power spectral density.

    Notes
    -----
    The returned arrays are centered (zero frequency at center).
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {field.shape}")

    Ny, Nx = field.shape
    zhat = fft2(field)
    psd = np.abs(zhat) ** 2 / (Nx * Ny) ** 2

    # Frequency arrays
    kx = fftfreq(Nx, d=spacing)
    ky = fftfreq(Ny, d=spacing)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid**2 + ky_grid**2)

    # Center the spectrum
    k = fftshift(k)
    psd = fftshift(psd)

    return k, psd


def psd_radial_average(
    field: np.ndarray,
    spacing: float = 1.0,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the radially averaged power spectral density.

    This is useful for isotropic fields where the 2D PSD is azimuthally symmetric.

    Parameters
    ----------
    field : ndarray
        2D input field.
    spacing : float, optional
        Grid spacing. Default is 1.0.
    n_bins : int, optional
        Number of radial bins. If None, uses N//4 bins.

    Returns
    -------
    k_bins : ndarray
        Wavenumber bin centers.
    psd_avg : ndarray
        Radially averaged PSD.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import psd_radial_average
    >>> from randomfield.generators import periodic_gaussian_random_field
    >>> field = periodic_gaussian_random_field(N=256, Hurst=0.8)
    >>> k, psd = psd_radial_average(field, spacing=1/256)
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {field.shape}")

    Ny, Nx = field.shape
    N = min(Nx, Ny)

    if n_bins is None:
        n_bins = N // 4

    # Compute 2D PSD
    k, psd = psd_2d(field, spacing)

    # Define radial bins
    k_max = 0.5 / spacing  # Nyquist
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

    # Average in each bin
    psd_avg = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k >= k_edges[i]) & (k < k_edges[i + 1])
        if np.any(mask):
            psd_avg[i] = np.mean(psd[mask])
            counts[i] = np.sum(mask)

    # Exclude empty bins
    valid = counts > 0
    return k_bins[valid], psd_avg[valid]


def psd_along_axis(
    field: np.ndarray,
    axis: int = 0,
    spacing: float = 1.0,
    average: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD along a specified axis of a 2D field.

    Parameters
    ----------
    field : ndarray
        2D input field.
    axis : int, optional
        Axis along which to compute PSD (0 or 1). Default is 0.
    spacing : float, optional
        Grid spacing. Default is 1.0.
    average : bool, optional
        If True, average over the other axis. If False, return PSD for
        each slice. Default is True.

    Returns
    -------
    k : ndarray
        Wavenumber array.
    psd : ndarray
        Power spectral density. Shape is (N,) if average=True, else (M, N)
        where M is the size of the other axis.
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {field.shape}")

    N = field.shape[axis]
    M = field.shape[1 - axis]

    # FFT along the specified axis
    zhat = np.fft.fft(field, axis=axis)
    psd_full = np.abs(zhat) ** 2 / N**2

    # Frequencies
    k_full = fftfreq(N, d=spacing)
    pos_mask = k_full > 0
    k = k_full[pos_mask]

    # Extract positive frequencies
    if axis == 0:
        psd_full = psd_full[pos_mask, :]
    else:
        psd_full = psd_full[:, pos_mask]

    if average:
        psd = np.mean(psd_full, axis=1 - axis)
    else:
        psd = psd_full

    return k, psd


def fit_power_law(
    k: np.ndarray,
    psd: np.ndarray,
    k_min: float | None = None,
    k_max: float | None = None,
) -> tuple[float, float, float]:
    """
    Fit a power law to the PSD: PSD(k) = A * k^(-β).

    Parameters
    ----------
    k : ndarray
        Wavenumber array.
    psd : ndarray
        Power spectral density.
    k_min : float, optional
        Minimum wavenumber for fitting. Default: min(k).
    k_max : float, optional
        Maximum wavenumber for fitting. Default: max(k).

    Returns
    -------
    A : float
        Amplitude (prefactor).
    beta : float
        Power law exponent.
    r_squared : float
        R² goodness of fit.

    Notes
    -----
    For self-affine surfaces, β = dim + 2H where H is the Hurst exponent.
    """
    k = np.asarray(k)
    psd = np.asarray(psd)

    # Filter range
    if k_min is None:
        k_min = k.min()
    if k_max is None:
        k_max = k.max()

    mask = (k >= k_min) & (k <= k_max) & (psd > 0) & (k > 0)
    k_fit = k[mask]
    psd_fit = psd[mask]

    if len(k_fit) < 2:
        raise ValueError("Not enough points for fitting")

    # Linear fit in log-log space
    log_k = np.log(k_fit)
    log_psd = np.log(psd_fit)

    # Fit: log(PSD) = log(A) - β * log(k)
    coeffs = np.polyfit(log_k, log_psd, 1)
    beta = -coeffs[0]
    A = np.exp(coeffs[1])

    # R² calculation
    log_psd_pred = coeffs[0] * log_k + coeffs[1]
    ss_res = np.sum((log_psd - log_psd_pred) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return A, beta, r_squared


def estimate_hurst_exponent(
    field: np.ndarray,
    k_low: float | None = None,
    k_high: float | None = None,
    spacing: float = 1.0,
) -> tuple[float, float]:
    """
    Estimate the Hurst exponent from a random field's PSD.

    For self-affine fields, PSD(k) ∝ k^(-dim-2H), so H = (β - dim) / 2.

    Parameters
    ----------
    field : ndarray
        Input field (1D or 2D).
    k_low : float, optional
        Minimum wavenumber for fitting.
    k_high : float, optional
        Maximum wavenumber for fitting.
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    H : float
        Estimated Hurst exponent.
    r_squared : float
        R² goodness of fit.
    """
    field = np.asarray(field)
    dim = field.ndim

    if dim == 1:
        k, psd = psd_1d(field, spacing)
    elif dim == 2:
        k, psd = psd_radial_average(field, spacing)
    else:
        raise ValueError("Only 1D and 2D fields supported")

    _, beta, r_squared = fit_power_law(k, psd, k_low, k_high)

    # H = (β - dim) / 2
    H = (beta - dim) / 2

    return H, r_squared

