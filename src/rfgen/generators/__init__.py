"""
Random field generators.

This module provides functions to generate periodic Gaussian random fields
with different spectral characteristics:

- Self-affine (power-law) spectrum
- Matérn covariance spectrum

Each generator has a `noise` parameter to choose between:
- `noise=True`: Filter white noise (introduces spectral fluctuations)
- `noise=False`: Ideal spectrum with random phases (exact PSD)
"""

from .selfaffine import selfaffine_field
from .matern import matern_spectrum, matern_field

__all__ = [
    "selfaffine_field",
    "matern_spectrum",
    "matern_field",
]
