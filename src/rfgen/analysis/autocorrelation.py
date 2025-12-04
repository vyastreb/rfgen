"""
Autocorrelation function computation for random fields.

Provides fast FFT-based computation of autocorrelation functions
for 1D profiles and 2D/3D fields.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn


def autocorrelation_1d(signal: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the autocorrelation function of a 1D signal using FFT.

    Uses the Wiener-Khinchin theorem: the autocorrelation is the inverse
    Fourier transform of the power spectral density.

    Parameters
    ----------
    signal : ndarray
        1D input signal.
    normalize : bool, optional
        If True, normalize so R(0) = 1. Default is True.

    Returns
    -------
    R : ndarray
        Autocorrelation function. Same length as input.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import autocorrelation_1d
    >>> signal = np.random.randn(1024)
    >>> R = autocorrelation_1d(signal)
    >>> R[0]  # Should be 1.0 if normalized
    1.0
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {signal.shape}")

    N = len(signal)
    zhat = fft(signal)
    psd = zhat * np.conj(zhat) / N**2
    R = np.real(ifft(psd))

    if normalize and R[0] != 0:
        R = R / R[0]

    return R


def autocorrelation_2d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the 2D autocorrelation function of a field using FFT.

    Uses the Wiener-Khinchin theorem: the autocorrelation is the inverse
    Fourier transform of the power spectral density.

    Parameters
    ----------
    field : ndarray
        2D input field with shape (Ny, Nx).
    normalize : bool, optional
        If True, normalize so R(0,0) = 1. Default is True.

    Returns
    -------
    R : ndarray
        2D autocorrelation function. Same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import autocorrelation_2d
    >>> field = np.random.randn(128, 128)
    >>> R = autocorrelation_2d(field)
    >>> R[0, 0]  # Should be 1.0 if normalized
    1.0
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {field.shape}")

    Ny, Nx = field.shape
    zhat = fft2(field)
    psd = zhat * np.conj(zhat) / (Nx * Ny) ** 2
    R = np.real(ifft2(psd))

    if normalize and R[0, 0] != 0:
        R = R / R[0, 0]

    return R


def autocorrelation_nd(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the N-dimensional autocorrelation function using FFT.

    Parameters
    ----------
    field : ndarray
        Input field of any dimension.
    normalize : bool, optional
        If True, normalize so R at origin = 1. Default is True.

    Returns
    -------
    R : ndarray
        Autocorrelation function. Same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import autocorrelation_nd
    >>> field = np.random.randn(32, 32, 32)
    >>> R = autocorrelation_nd(field)
    >>> R[0, 0, 0]  # Should be 1.0 if normalized
    1.0
    """
    field = np.asarray(field)
    N_total = field.size
    zhat = fftn(field)
    psd = zhat * np.conj(zhat) / N_total**2
    R = np.real(ifftn(psd))

    if normalize:
        origin = tuple(0 for _ in range(field.ndim))
        if R[origin] != 0:
            R = R / R[origin]

    return R


def correlation_length(
    R: np.ndarray,
    threshold: float = 0.0,
    spacing: float = 1.0,
) -> float:
    """
    Estimate the correlation length from an autocorrelation function.

    The correlation length is defined as the distance at which the
    autocorrelation first drops below the threshold value.

    Parameters
    ----------
    R : ndarray
        1D autocorrelation function (assumed normalized, R[0] = 1).
    threshold : float, optional
        Threshold value for determining correlation length. Default is 0.0
        (first zero crossing).
    spacing : float, optional
        Grid spacing for converting index to physical distance. Default is 1.0.

    Returns
    -------
    l_corr : float
        Estimated correlation length.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import autocorrelation_1d, correlation_length
    >>> from randomfield.generators import periodic_gaussian_random_field
    >>> field = periodic_gaussian_random_field(dim=1, N=1024, k_low=0.01, k_high=0.3)
    >>> R = autocorrelation_1d(field)
    >>> l_corr = correlation_length(R, spacing=1.0/1024)
    """
    R = np.asarray(R).flatten()

    # Find first crossing below threshold
    for i, r in enumerate(R):
        if r < threshold:
            # Linear interpolation for better estimate
            if i > 0:
                r_prev = R[i - 1]
                # Interpolate: find x where R(x) = threshold
                frac = (r_prev - threshold) / (r_prev - r + 1e-30)
                return (i - 1 + frac) * spacing
            return i * spacing

    # If never crosses, return half the domain
    return len(R) * spacing / 2


def integral_correlation_length(
    R: np.ndarray,
    spacing: float = 1.0,
) -> float:
    """
    Compute the integral correlation length.

    The integral correlation length is defined as:
        L = ∫₀^∞ R(r) dr

    For discrete data, we integrate up to the first zero crossing
    to avoid issues with oscillating tails.

    Parameters
    ----------
    R : ndarray
        1D autocorrelation function (assumed normalized, R[0] = 1).
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    L : float
        Integral correlation length.
    """
    R = np.asarray(R).flatten()

    # Find first zero crossing
    n_integrate = len(R)
    for i, r in enumerate(R):
        if r < 0:
            n_integrate = i
            break

    # Integrate using trapezoidal rule
    L = np.trapz(R[:n_integrate], dx=spacing)

    return L

