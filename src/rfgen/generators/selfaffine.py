"""
Self-affine Gaussian random field generators.

Generate 1D/2D/3D periodic Gaussian random fields with self-affine (power-law) spectra.

Two generation modes are available via the `noise` parameter:

1. **Filtered white noise** (`noise=True`):
   White noise in real space is filtered in Fourier space with a power-law filter.
   Reference: Hu, Y.Z. and Tonder, K., 1992. International Journal of Machine Tools
   and Manufacture, 32(1-2), pp.83-90. DOI: 10.1016/0890-6955(92)90064-N

2. **Ideal spectrum with random phase** (`noise=False`):
   Directly constructs Fourier coefficients with prescribed power-law magnitudes
   and random phases that respect Hermitian symmetry.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

import numpy as np

from ._fft import selfaffine_filter


def selfaffine_field(
    dim: int = 2,
    N: int = 256,
    Hurst: float = 0.5,
    k_low: float = 0.03,
    k_high: float = 0.3,
    plateau: bool = False,
    noise: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a periodic Gaussian random field with a self-affine spectrum.

    Parameters
    ----------
    dim : int, optional
        Dimension of the field (1, 2, or 3). Default is 2.
    N : int, optional
        Size of the field along each dimension. Default is 256.
    Hurst : float, optional
        Hurst exponent, in the range [0, 1]. Controls the roughness/smoothness.
        H → 0: rough field, H → 1: smooth field. Default is 0.5.
    k_low : float, optional
        Lower bound of the wavenumber range (k_low > 0). Default is 0.03.
    k_high : float, optional
        Upper bound of the wavenumber range (k_high ≤ 0.5, Nyquist). Default is 0.3.
    plateau : bool, optional
        If True, the power spectrum is flat for k < k_low (roll-off).
        If False, wavenumbers below k_low are set to zero. Default is False.
    noise : bool, optional
        If True, generate field by filtering white noise (introduces spectral noise).
        If False, generate field with ideal spectrum and random phases (exact PSD).
        Default is True.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If None, uses numpy's
        default RNG. Default is None.
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
    The power spectral density follows:
        Φ(k) ∝ |k|^{-(dim + 2H)}

    where H is the Hurst exponent. For `noise=True`, the PSD has random
    fluctuations around this power law. For `noise=False`, the PSD follows
    the power law exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from rfgen import selfaffine_field
    >>> rng = np.random.default_rng(42)

    # Field with spectral noise (default)
    >>> field = selfaffine_field(dim=2, N=128, Hurst=0.8, rng=rng)

    # Field with ideal (exact) spectrum
    >>> field_ideal = selfaffine_field(dim=2, N=128, Hurst=0.8, noise=False, rng=rng)
    """
    # Validate (and adjust) parameters
    if k_low is None or k_low == 0:
        k_low = 1 / (2 * N)
    if k_high is None or k_high == 0:
        k_high = 0.5
    if not (0 < k_low <= k_high <= 0.5):
        raise ValueError("Require 0 < k_low <= k_high <= 0.5 (Nyquist)")
    if not 0 <= Hurst <= 1:
        raise ValueError("Hurst exponent must be in [0, 1]")
    if dim not in (1, 2, 3):
        raise ValueError(f"Dimension must be 1, 2, or 3, got {dim}")

    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        mode = "filtered noise" if noise else "ideal spectrum"
        print(f"Self-affine Random Field ({mode}):")
        print(f"    dim = {dim}")
        print(f"    N = {N}")
        print(f"    Hurst = {Hurst}")
        print(f"    k_low = {k_low}")
        print(f"    k_high = {k_high}")
        print(f"    plateau = {plateau}")

    if noise:
        return _selfaffine_filtered_noise(dim, N, Hurst, k_low, k_high, plateau, rng)
    else:
        return _selfaffine_ideal_spectrum(dim, N, Hurst, k_low, k_high, plateau, rng)


def _selfaffine_filtered_noise(
    dim: int,
    N: int,
    Hurst: float,
    k_low: float,
    k_high: float,
    plateau: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate self-affine field by filtering white noise."""
    shape = (N,) * dim
    amplitude = selfaffine_filter(dim, N, Hurst, k_low, k_high, plateau)
    spectrum = np.fft.rfftn(rng.standard_normal(shape))
    spectrum *= amplitude
    return np.fft.irfftn(spectrum, s=shape)


def _selfaffine_ideal_spectrum(
    dim: int,
    N: int,
    Hurst: float,
    k_low: float,
    k_high: float,
    plateau: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate self-affine field with ideal spectrum and random phases."""
    shape = (N,) * dim
    amplitude = selfaffine_filter(dim, N, Hurst, k_low, k_high, plateau)

    # ``rfftn`` retains the complex phase of every independent Fourier mode.
    spectrum = np.fft.rfftn(rng.standard_normal(shape))
    spectrum /= np.abs(spectrum) + 1e-30
    spectrum *= amplitude
    return np.fft.irfftn(spectrum, s=shape)
