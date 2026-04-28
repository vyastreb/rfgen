"""
Spectral moments computation for random fields.

Spectral moments are important statistical descriptors of random surfaces
and fields, used in tribology, contact mechanics, and surface metrology.

The spectral moments m_{ij} are defined as:
    m_{ij} = ∫∫ ωx^i * ωy^j * Φ(ωx, ωy) dωx dωy

where ω = 2π k are **angular** wavenumbers and Φ is the power spectral density.

Using angular wavenumbers means physical derivatives are obtained directly:

    var(∂z/∂x) = m_{20}      →  rms slope in x  = √m_{20}
    var(∂²z/∂x²) = m_{40}    →  rms curvature in x = √m_{40}
    rms |∇z| = √(2 m_2)  where m_2 = (m_{20} + m_{02}) / 2

References:
    Nayak, P.R., 1971. Random process model of rough surfaces. 
    Journal of Lubrication Technology, 93(3), pp.398-407.
    
    Greenwood, J.A., 1984. A unified theory of surface roughness. 
    Proceedings of the Royal Society A, 393(1804), pp.133-157.

Author: Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux
License: BSD-3-Clause
"""

import numpy as np
from numpy.fft import fft2, fftfreq


def spectral_moment(
    field: np.ndarray,
    i: int,
    j: int,
    spacing: float = 1.0,
) -> float:
    """
    Compute the spectral moment m_{ij} of a 2D field.

    The spectral moment is defined as:
        m_{ij} = ∫∫ |ωx|^i * |ωy|^j * Φ(ωx, ωy) dωx dωy

    where ω = 2π k are angular wavenumbers and Φ is the power spectral density.
    Physical derivatives are obtained directly: var(∂z/∂x) = m_{20}, etc.

    Parameters
    ----------
    field : ndarray
        2D input field.
    i : int
        Power of kx (non-negative integer).
    j : int
        Power of ky (non-negative integer).
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    m_ij : float
        The spectral moment m_{ij}.

    Examples
    --------
    >>> import numpy as np
    >>> from randomfield.analysis import spectral_moment
    >>> from randomfield.generators import periodic_gaussian_random_field
    >>> field = periodic_gaussian_random_field(N=256, Hurst=0.8)
    >>> m00 = spectral_moment(field, 0, 0)  # Variance of heights
    >>> m20 = spectral_moment(field, 2, 0)  # var(dz/dx)
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {field.shape}")
    if i < 0 or j < 0:
        raise ValueError("Moment indices must be non-negative")

    Ny, Nx = field.shape
    dk = 1.0 / spacing  # Cycle-frequency resolution

    # Compute PSD (in cycle-frequency space)
    zhat = fft2(field)
    psd = np.abs(zhat) ** 2 / (Nx * Ny) ** 2

    # Cycle-frequency arrays
    kx = fftfreq(Nx, d=spacing)
    ky = fftfreq(Ny, d=spacing)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Use absolute values for moments (symmetric about origin)
    kx_power = np.abs(kx_grid) ** i if i > 0 else np.ones_like(kx_grid)
    ky_power = np.abs(ky_grid) ** j if j > 0 else np.ones_like(ky_grid)

    # Integrate in cycle-frequency space, then convert to angular-frequency moments:
    # m_{ij}^angular = (2π)^(i+j) * m_{ij}^cycle
    m_ij = np.sum(kx_power * ky_power * psd) * (dk / Nx) * (dk / Ny)

    return float(m_ij * (2 * np.pi) ** (i + j))


def spectral_moment_1d(
    signal: np.ndarray,
    n: int,
    spacing: float = 1.0,
) -> float:
    """
    Compute the n-th spectral moment of a 1D signal.

    The spectral moment is defined as:
        m_n = ∫ |ω|^n * Φ(ω) dω

    where ω = 2π k is the angular wavenumber.

    Parameters
    ----------
    signal : ndarray
        1D input signal.
    n : int
        Order of the moment (non-negative integer).
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    m_n : float
        The n-th spectral moment.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {signal.shape}")
    if n < 0:
        raise ValueError("Moment order must be non-negative")

    N = len(signal)
    dk = 1.0 / (N * spacing)

    # Compute PSD
    zhat = np.fft.fft(signal)
    psd = np.abs(zhat) ** 2 / N**2

    # Frequencies
    k = fftfreq(N, d=spacing)

    # Use absolute values
    k_power = np.abs(k) ** n if n > 0 else np.ones_like(k)

    # Integrate in cycle-frequency space, then convert to angular-frequency moment:
    # m_n^angular = (2π)^n * m_n^cycle
    m_n = np.sum(k_power * psd) * dk

    return float(m_n * (2 * np.pi) ** n)


def compute_standard_moments(
    field: np.ndarray,
    spacing: float = 1.0,
) -> dict[str, float]:
    """
    Compute standard spectral moments for a 2D surface.

    Computes m00, m10, m01, m20, m02, m11, m40, m04, m22 which are
    commonly used in surface roughness characterization.

    Parameters
    ----------
    field : ndarray
        2D input field.
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    moments : dict
        Dictionary with moment names as keys and values.

    Notes
    -----
    Physical interpretations:
    - m00: Variance of the surface heights
    - m20, m02: Related to mean square slopes
    - m40, m04: Related to mean square curvatures
    - m11: Cross-correlation of slopes

    For isotropic surfaces: m20 ≈ m02, m40 ≈ m04.
    """
    moments = {}

    # Zero-order moment (variance)
    moments["m00"] = spectral_moment(field, 0, 0, spacing)

    # First-order moments
    moments["m10"] = spectral_moment(field, 1, 0, spacing)
    moments["m01"] = spectral_moment(field, 0, 1, spacing)

    # Second-order moments
    moments["m20"] = spectral_moment(field, 2, 0, spacing)
    moments["m02"] = spectral_moment(field, 0, 2, spacing)
    moments["m11"] = spectral_moment(field, 1, 1, spacing)

    # Fourth-order moments
    moments["m40"] = spectral_moment(field, 4, 0, spacing)
    moments["m04"] = spectral_moment(field, 0, 4, spacing)
    moments["m22"] = spectral_moment(field, 2, 2, spacing)

    return moments


def nayak_parameter(field: np.ndarray, spacing: float = 1.0) -> float:
    """
    Compute Nayak's bandwidth parameter α.

    The bandwidth parameter is defined as:
        α = m0 * m4 / m2²

    where m0, m2, m4 are isotropic spectral moments.

    For Gaussian surfaces: α ≥ 1, with α = 1 for a narrow-band surface.

    Parameters
    ----------
    field : ndarray
        2D input field.
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    alpha : float
        Nayak's bandwidth parameter.

    References
    ----------
    Nayak, P.R., 1971. Random process model of rough surfaces.
    Journal of Lubrication Technology, 93(3), pp.398-407.
    """
    m00 = spectral_moment(field, 0, 0, spacing)
    m20 = spectral_moment(field, 2, 0, spacing)
    m02 = spectral_moment(field, 0, 2, spacing)
    m40 = spectral_moment(field, 4, 0, spacing)
    m04 = spectral_moment(field, 0, 4, spacing)

    # Isotropic averages
    m2 = 0.5 * (m20 + m02)
    m4 = 0.5 * (m40 + m04)

    if m2 == 0:
        return np.inf

    alpha = m00 * m4 / m2**2

    return alpha


def rms_quantities(field: np.ndarray, spacing: float = 1.0) -> dict[str, float]:
    """
    Compute RMS surface quantities from spectral moments.

    Parameters
    ----------
    field : ndarray
        2D input field.
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    quantities : dict
        Dictionary containing:
        - 'rms_height': RMS height (σ)
        - 'rms_slope_x': RMS slope in x direction
        - 'rms_slope_y': RMS slope in y direction
        - 'rms_slope': Isotropic RMS slope
        - 'rms_curvature_x': RMS curvature in x direction
        - 'rms_curvature_y': RMS curvature in y direction
        - 'rms_curvature': Isotropic RMS curvature
    """
    m00 = spectral_moment(field, 0, 0, spacing)
    m20 = spectral_moment(field, 2, 0, spacing)
    m02 = spectral_moment(field, 0, 2, spacing)
    m40 = spectral_moment(field, 4, 0, spacing)
    m04 = spectral_moment(field, 0, 4, spacing)

    # spectral_moment returns angular-frequency moments, so physical derivatives
    # are obtained directly: var(∂z/∂x) = m20, var(∂²z/∂x²) = m40, etc.
    quantities = {
        "rms_height": np.sqrt(m00),
        "rms_slope_x": np.sqrt(m20),
        "rms_slope_y": np.sqrt(m02),
        "rms_slope": np.sqrt(0.5 * (m20 + m02)),
        "rms_curvature_x": np.sqrt(m40),
        "rms_curvature_y": np.sqrt(m04),
        "rms_curvature": np.sqrt(0.5 * (m40 + m04)),
    }

    return quantities


def summit_density_estimate(field: np.ndarray, spacing: float = 1.0) -> float:
    """
    Estimate the density of summits (local maxima) per unit area.

    Based on random process theory for Gaussian surfaces:
        D_s = (1 / 6π√3) * √(m4/m2)

    Parameters
    ----------
    field : ndarray
        2D input field.
    spacing : float, optional
        Grid spacing. Default is 1.0.

    Returns
    -------
    D_s : float
        Summit density (number per unit area).

    Notes
    -----
    This is a statistical estimate valid for isotropic Gaussian surfaces.
    The actual count of local maxima may differ.
    """
    m20 = spectral_moment(field, 2, 0, spacing)
    m02 = spectral_moment(field, 0, 2, spacing)
    m40 = spectral_moment(field, 4, 0, spacing)
    m04 = spectral_moment(field, 0, 4, spacing)

    m2 = 0.5 * (m20 + m02)
    m4 = 0.5 * (m40 + m04)

    if m2 <= 0:
        return 0.0

    D_s = (1.0 / (6 * np.pi * np.sqrt(3))) * np.sqrt(m4 / m2)

    return D_s

