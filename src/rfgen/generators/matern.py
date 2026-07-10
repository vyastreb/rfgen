"""
Matérn covariance Gaussian random field generators.

Generate 1D/2D/3D periodic Gaussian random fields with Matérn covariance structure.
The Matérn covariance is widely used in geostatistics and spatial statistics.

Two generation modes are available via the `noise` parameter:

1. **Filtered white noise** (`noise=True`):
   White noise filtered with the Matérn spectral density.

2. **Ideal spectrum with random phase** (`noise=False`):
   Directly constructs Fourier coefficients with exact Matérn magnitudes
   and random phases.

Reference:
    Rasmussen, C.E. and Williams, C.K.I., 2006. Gaussian Processes for Machine Learning.
    MIT Press. Chapter 4.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

import math

import numpy as np
from scipy.special import gamma

from ._fft import real_fft_radial_frequency_grid


def matern_spectrum(
    k: np.ndarray | float,
    sigma: float,
    dim: int,
    nu: float,
    cor_length: float,
) -> np.ndarray | float:
    """
    Compute the Matérn power spectral density.

    The Matérn spectral density in dimension d is:

        S(k) = σ² · (2^d · π^(d/2) · Γ(ν + d/2) · (2ν)^ν) /
               (Γ(ν) · ℓ^(2ν)) · (2ν/ℓ² + 4π²k²)^(-(ν + d/2))

    Parameters
    ----------
    k : array_like or float
        Wavenumber magnitude(s).
    sigma : float
        Standard deviation of the field.
    dim : int
        Spatial dimension (1, 2, or 3).
    nu : float
        Smoothness parameter (ν > 0). As ν → ∞, approaches squared exponential.
        Common values: 0.5 (Ornstein-Uhlenbeck), 1.5, 2.5.
    cor_length : float
        Correlation length.

    Returns
    -------
    S : array_like or float
        Power spectral density at the given wavenumber(s).

    Notes
    -----
    Special cases for ν:
    - ν = 0.5: Exponential covariance (Ornstein-Uhlenbeck process)
    - ν = 1.5: Once differentiable
    - ν = 2.5: Twice differentiable
    - ν → ∞: Squared exponential (infinitely differentiable)
    """
    # Numerator
    numerator = (
        (sigma**2)
        * (2**dim)
        * (math.pi ** (dim / 2))
        * gamma(nu + dim / 2)
        * ((2 * nu) ** nu)
    )

    # Denominator
    denominator = gamma(nu) * (cor_length ** (2 * nu))

    # Power term
    power_base = (2 * nu) / (cor_length**2) + (4 * (math.pi**2) * (k**2))
    power_exponent = -(nu + dim / 2)

    return (numerator / denominator) * (power_base**power_exponent)


def matern_field(
    dim: int = 2,
    N: int = 256,
    nu: float = 0.5,
    correlation_length: float = 0.1,
    sigma: float = 1.0,
    k_low: float = 0.03,
    k_high: float = 0.3,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a periodic Gaussian random field with Matérn covariance.

    Parameters
    ----------
    dim : int, optional
        Dimension of the field (1, 2, or 3). Default is 2.
    N : int, optional
        Size of the field along each dimension. Default is 256.
    nu : float, optional
        Smoothness parameter (ν > 0). Default is 0.5.
    correlation_length : float, optional
        Correlation length of the field. Default is 0.1.
    sigma : float, optional
        Standard deviation of the field. Default is 1.0.
    k_low : float, optional
        Lower bound of the wavenumber range. Default is 0.03.
    k_high : float, optional
        Upper bound of the wavenumber range (≤ 0.5). Default is 0.3.
    noise : bool, optional
        If True, generate field by filtering white noise (introduces spectral noise).
        If False, generate field with ideal spectrum and random phases (exact PSD).
        Default is True.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    verbose : bool, optional
        If True, print generation parameters. Default is False.

    Returns
    -------
    z : ndarray
        The generated random field with shape (N,), (N, N), or (N, N, N).

    Raises
    ------
    ValueError
        If parameters are outside valid ranges.

    Notes
    -----
    For `noise=True`, the PSD has random fluctuations around the Matérn form.
    For `noise=False`, the PSD follows the Matérn form exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from rfgen import matern_field
    >>> rng = np.random.default_rng(42)

    # Field with spectral noise (default)
    >>> field = matern_field(dim=2, N=128, nu=1.5, rng=rng)

    # Field with ideal (exact) Matérn spectrum
    >>> field_ideal = matern_field(dim=2, N=128, nu=1.5, noise=False, rng=rng)
    """
    # Validate parameters
    if not (0 < k_low <= k_high <= 0.5):
        raise ValueError("Require 0 < k_low <= k_high <= 0.5 (Nyquist)")
    if nu <= 0:
        raise ValueError("Smoothness parameter nu must be > 0")
    if correlation_length <= 0:
        raise ValueError("Correlation length must be > 0")
    if dim not in (1, 2, 3):
        raise ValueError(f"Dimension must be 1, 2, or 3, got {dim}")

    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        mode = "filtered noise" if noise else "ideal spectrum"
        print(f"Matérn Random Field ({mode}):")
        print(f"    dim = {dim}")
        print(f"    N = {N}")
        print(f"    nu = {nu}")
        print(f"    correlation_length = {correlation_length}")
        print(f"    sigma = {sigma}")
        print(f"    k_low = {k_low}")
        print(f"    k_high = {k_high}")

    if noise:
        return _matern_filtered_noise(dim, N, nu, correlation_length, sigma, k_low, k_high, rng)
    else:
        return _matern_ideal_spectrum(dim, N, nu, correlation_length, sigma, k_low, k_high, rng)


def _matern_filtered_noise(
    dim: int,
    N: int,
    nu: float,
    correlation_length: float,
    sigma: float,
    k_low: float,
    k_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Matérn field by filtering white noise."""
    shape = (N,) * dim
    amplitude = _matern_filter(dim, N, nu, correlation_length, sigma, k_low, k_high)
    spectrum = np.fft.rfftn(rng.standard_normal(shape))
    spectrum *= amplitude
    return np.fft.irfftn(spectrum, s=shape)


def _matern_ideal_spectrum(
    dim: int,
    N: int,
    nu: float,
    correlation_length: float,
    sigma: float,
    k_low: float,
    k_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Matérn field with ideal spectrum and random phases."""
    shape = (N,) * dim
    amplitude = _matern_filter(dim, N, nu, correlation_length, sigma, k_low, k_high)

    # ``rfftn`` retains the complex phase of every independent Fourier mode.
    spectrum = np.fft.rfftn(rng.standard_normal(shape))
    spectrum /= np.abs(spectrum) + 1e-30
    spectrum *= amplitude
    return np.fft.irfftn(spectrum, s=shape)


def _matern_filter(
    dim: int,
    n: int,
    nu: float,
    correlation_length: float,
    sigma: float,
    k_low: float,
    k_high: float,
) -> np.ndarray:
    """Build a Matérn amplitude filter on an ``rfftn`` grid."""
    k = real_fft_radial_frequency_grid(dim, n)
    amplitude = np.zeros_like(k)
    mask = (k >= k_low) & (k <= k_high)
    amplitude[mask] = np.sqrt(matern_spectrum(k[mask], sigma, dim, nu, correlation_length))
    return amplitude
