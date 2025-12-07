"""
  Several characteristic roughness surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from rfgen import selfaffine_field

def build_inverse_cdf_from_pdf(pdf, x_min, x_max, n_grid=10_000):
    """
    Construct a numerical inverse CDF F_T^{-1} from a target PDF.

    Parameters
    ----------
    pdf : callable
        Function pdf(x) returning the (unnormalized or normalized) density.
    x_min, x_max : float
        Bounds of the support where pdf(x) is significant.
    n_grid : int
        Number of grid points for tabulation.

    Returns
    -------
    F_T_inv : callable
        Numerical inverse CDF: takes u in [0, 1] and returns x.
    """
    x = np.linspace(x_min, x_max, n_grid)
    p = np.maximum(pdf(x), 0.0)      # enforce non-negativity
    if not np.any(p > 0):
        raise ValueError("PDF is zero on the whole grid; check x_min/x_max.")

    # CDF via cumulative sum, then normalize to [0,1]
    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]
    cdf[0] = 0.0
    cdf[-1] = 1.0

    def F_T_inv(u):
        u = np.asarray(u)
        # clip to [0,1] to avoid numerical issues
        u_clipped = np.clip(u, 0.0, 1.0)
        return np.interp(u_clipped, cdf, x, left=x_min, right=x_max)

    return F_T_inv

def remap_distribution(z: np.ndarray, probability_distribution: Callable[[np.ndarray], np.ndarray], x_min: float = 0.0, x_max: float = 10.0):
    """
    Remap field z to have target marginal via inverse CDF.
    
    Parameters
    ----------
    z : np.ndarray
        Input field to remap.
    probability_distribution : Callable
        Target PDF function.
    x_min, x_max : float
        Bounds of the support where the PDF is significant.
    """
    F_T_inv = build_inverse_cdf_from_pdf(probability_distribution, x_min, x_max)
    z_flat = z.ravel()
    # empirical CDF values of z via ranks
    ranks = np.argsort(np.argsort(z_flat))
    u = (ranks + 0.5) / len(z_flat)   # in (0,1)

    z_new_flat = F_T_inv(u)
    return z_new_flat.reshape(z.shape)


def asphalt_concrete_surface(L: float, N: int, aggregate_size: float, high_distribution_probability: Callable[[np.ndarray], np.ndarray], height_bounds: tuple[float, float], Hurst_aggregate: float, k_high: float, Hurst_roughness: float, rms_roughness: float, k_low: float = None, rng: np.random.Generator = None, seed: int = 42):
    """
    Asphalt concrete surface with given aggregate size and high distribution probability.

    Parameters
    ----------
    L : float
        Length of the surface in length units.
    N : int
        Number of points on the surface.
    aggregate_size : float
        Size of the aggregate in length units.
    high_distribution_probability : Callable[[np.ndarray], np.ndarray]
        Function that returns the probability of a high distribution of the asphalt concrete.
    height_bounds : tuple[float, float]
        (min, max) bounds for the height distribution support.
    k_low : float
        Lower bound of the wavenumber range (k_low > 0) for the superposed roughness.
    k_high : float
        Upper bound of the wavenumber range (k_high ≤ 0.5, Nyquist) for the superposed roughness.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """
    # Generate self-affine field with lambda_s = 2\pi/k_s corresponding to the aggregate size
    k_s = 2 * np.pi / aggregate_size
    k_s *= L / N
    k_l = 0.9*k_s
    if rng is None:
        rng = np.random.default_rng(seed)
    field_aggregate = selfaffine_field(dim=2, N=N, Hurst=Hurst_aggregate, k_low=k_l, k_high=k_s, rng=rng, noise=True)

    # Adjust the hights to fit the high distribution probability
    field_aggregate = remap_distribution(field_aggregate, high_distribution_probability, x_min=height_bounds[0], x_max=height_bounds[1])

    #Generate self-affine roughness to superpose on the aggregate
    if k_low is None:
        k_low = k_s
    field_roughness = selfaffine_field(dim=2, N=N, Hurst=Hurst_roughness, k_low=k_low, k_high=k_high, rng=rng)
    # Normalize the roughness to have the given rms roughness
    field_roughness = field_roughness * rms_roughness / np.std(field_roughness)
    # Superpose the two fields
    field = field_aggregate + field_roughness
    return field

def main():
    # Parameters
    L = 100.0 # mm
    N = 1024
    aggregate_size = 50.0 # mm
    def Weibull_distribution(x: float, k: float, lambd: float) -> float:
        return k / lambd * (x / lambd)**(k - 1) * np.exp(-(x / lambd)**k)
    high_distribution_probability = lambda x: Weibull_distribution(x, k=2, lambd=3)

    height_bounds = (0.0, 15.0)  # bounds for Weibull distribution support
    
    z = np.linspace(height_bounds[0], height_bounds[1], 100)
    plt.plot(z, high_distribution_probability(z))
    plt.show()

    Hurst_aggregate = 0.5
    k_high = 128 / N
    Hurst_roughness = 0.8
    rms_roughness = .3 # mm

    # Generate the surface
    field = asphalt_concrete_surface(L, N, aggregate_size, high_distribution_probability, height_bounds, Hurst_aggregate, k_high, Hurst_roughness, rms_roughness)

    pdf,bins = np.histogram(field.ravel(), bins=100, density=True)
    plt.plot(bins[:-1], pdf, label="PDF")
    plt.plot(z, high_distribution_probability(z), label="Target PDF")
    plt.legend()
    plt.show()

    # Plot the surface
    plt.imshow(field, cmap="RdYlBu_r", interpolation="bicubic")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()