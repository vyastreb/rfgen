"""
rfgen - Periodic Gaussian Random Field Generation and Analysis.

A Python package for generating and analyzing periodic Gaussian random fields
using spectral (Fourier) methods.

    Features
    --------
    - Self-affine (power-law) spectrum generation
    - Matérn covariance spectrum generation
    - Arbitrary PSD and PDF control (IAAFT)
    - Both filtered white noise and ideal spectrum methods
- Fast autocorrelation and PSD computation
- Spectral moment analysis

Quick Start
-----------
>>> import numpy as np
>>> from rfgen import selfaffine_field
>>> rng = np.random.default_rng(42)
>>> field = selfaffine_field(dim=2, N=256, Hurst=0.8, rng=rng)

References
----------
Hu, Y.Z. and Tonder, K., 1992. Simulation of 3-D random rough surface by 2-D
digital filter and Fourier analysis. International Journal of Machine Tools
and Manufacture, 32(1-2), pp.83-90. DOI: 10.1016/0890-6955(92)90064-N

Author
------
Vladislav Yastrebov, CNRS, Mines Paris - PSL, Centre des matériaux

License
-------
BSD-3-Clause
"""

__version__ = "0.1.5"
__author__ = "Vladislav Yastrebov"

# Main generators
from .generators import (
    selfaffine_field,
    matern_field,
    matern_spectrum,
    arbitrary_pdf_psd_field,
)

# Analysis tools
from .analysis import (
    # Autocorrelation
    autocorrelation_1d,
    autocorrelation_2d,
    autocorrelation_nd,
    correlation_length,
    integral_correlation_length,
    # Spectrum
    psd_1d,
    psd_2d,
    psd_radial_average,
    psd_along_axis,
    fit_power_law,
    estimate_hurst_exponent,
    # Moments
    spectral_moment,
    spectral_moment_1d,
    compute_standard_moments,
    nayak_parameter,
    rms_quantities,
    summit_density_estimate,
)

__all__ = [
    # Version
    "__version__",
    # Generators
    "selfaffine_field",
    "matern_field",
    "matern_spectrum",
    "arbitrary_pdf_psd_field",
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
