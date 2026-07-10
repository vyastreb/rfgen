"""Internal helpers for memory-efficient real Fourier transforms."""

import numpy as np


def real_fft_radial_frequency_grid(dim: int, n: int) -> np.ndarray:
    """Return ``|k|`` on the non-redundant grid used by ``rfftn``.

    The last axis of a real FFT contains only non-negative frequencies.  The
    other axes retain the usual ``fftfreq`` ordering.  The grid is assembled
    with broadcasting into one output array, rather than materialising one
    full coordinate array per axis.
    """
    shape = (n,) * (dim - 1) + (n // 2 + 1,)
    k_squared = np.zeros(shape, dtype=float)

    for axis in range(dim):
        frequencies = np.fft.rfftfreq(n) if axis == dim - 1 else np.fft.fftfreq(n)
        axis_shape = [1] * dim
        axis_shape[axis] = frequencies.size
        k_squared += frequencies.reshape(axis_shape) ** 2

    return np.sqrt(k_squared, out=k_squared)


def selfaffine_filter(
    dim: int,
    n: int,
    hurst: float,
    k_low: float,
    k_high: float,
    plateau: bool,
) -> np.ndarray:
    """Build a self-affine amplitude filter on an ``rfftn`` grid."""
    amplitude = real_fft_radial_frequency_grid(dim, n)
    in_band = (amplitude >= k_low) & (amplitude <= k_high)
    low_frequency = amplitude < k_low if plateau else None

    np.divide(amplitude, k_low, out=amplitude, where=in_band)
    np.power(amplitude, -(0.5 * dim + hurst), out=amplitude, where=in_band)
    amplitude[~in_band] = 0.0

    if low_frequency is not None:
        amplitude[low_frequency] = 1.0

    return amplitude
