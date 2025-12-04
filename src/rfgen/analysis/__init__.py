"""
Analysis tools for random fields.

This module provides functions for analyzing random fields:

- Autocorrelation functions (1D, 2D, N-D)
- Power spectral density computation
- Spectral moments and derived quantities
"""

from .autocorrelation import (
    autocorrelation_1d,
    autocorrelation_2d,
    autocorrelation_nd,
    correlation_length,
    integral_correlation_length,
)
from .spectrum import (
    psd_1d,
    psd_2d,
    psd_radial_average,
    psd_along_axis,
    fit_power_law,
    estimate_hurst_exponent,
)
from .moments import (
    spectral_moment,
    spectral_moment_1d,
    compute_standard_moments,
    nayak_parameter,
    rms_quantities,
    summit_density_estimate,
)

__all__ = [
    # Autocorrelation
    "autocorrelation_1d",
    "autocorrelation_2d",
    "autocorrelation_nd",
    "correlation_length",
    "integral_correlation_length",
    # Spectrum
    "psd_1d",
    "psd_2d",
    "psd_radial_average",
    "psd_along_axis",
    "fit_power_law",
    "estimate_hurst_exponent",
    # Moments
    "spectral_moment",
    "spectral_moment_1d",
    "compute_standard_moments",
    "nayak_parameter",
    "rms_quantities",
    "summit_density_estimate",
]
