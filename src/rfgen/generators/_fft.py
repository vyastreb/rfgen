"""Internal helpers for memory-efficient real Fourier transforms."""

import numpy as np
from scipy import fft as scipy_fft


def real_dtype(dtype: object) -> np.dtype:
    """Return a supported real floating-point dtype."""
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("dtype must be numpy.float32 or numpy.float64") from error

    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be numpy.float32 or numpy.float64")
    return resolved


def rfftn(values: np.ndarray) -> np.ndarray:
    """Transform real values without promoting single-precision input."""
    if values.dtype == np.float32:
        return scipy_fft.rfftn(values)
    return np.fft.rfftn(values)


def irfftn(spectrum: np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Inverse-transform a half-spectrum in the requested real precision."""
    if dtype == np.dtype(np.float32):
        return scipy_fft.irfftn(spectrum, s=shape).astype(dtype, copy=False)
    return np.fft.irfftn(spectrum, s=shape)


def real_fft_radial_frequency_grid(
    dim: int,
    n: int,
    dtype: object = np.float64,
) -> np.ndarray:
    """Return ``|k|`` on the non-redundant grid used by ``rfftn``.

    The last axis of a real FFT contains only non-negative frequencies.  The
    other axes retain the usual ``fftfreq`` ordering.  The grid is assembled
    with broadcasting into one output array, rather than materialising one
    full coordinate array per axis.
    """
    dtype = real_dtype(dtype)
    shape = (n,) * (dim - 1) + (n // 2 + 1,)
    k_squared = np.zeros(shape, dtype=dtype)

    for axis in range(dim):
        frequencies = np.fft.rfftfreq(n) if axis == dim - 1 else np.fft.fftfreq(n)
        frequencies = frequencies.astype(dtype, copy=False)
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
    dtype: object = np.float64,
) -> np.ndarray:
    """Build a self-affine amplitude filter on an ``rfftn`` grid."""
    amplitude = real_fft_radial_frequency_grid(dim, n, dtype=dtype)
    in_band = (amplitude >= k_low) & (amplitude <= k_high)
    low_frequency = amplitude < k_low if plateau else None

    np.divide(amplitude, k_low, out=amplitude, where=in_band)
    np.power(amplitude, -(0.5 * dim + hurst), out=amplitude, where=in_band)
    amplitude[~in_band] = 0.0

    if low_frequency is not None:
        amplitude[low_frequency] = 1.0

    return amplitude
