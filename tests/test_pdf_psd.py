"""Tests for arbitrary PDF/PSD random field generator."""

import numpy as np
import pytest
from scipy import stats

from rfgen import arbitrary_pdf_psd_field


class TestArbitraryPDFPSDGenerator:
    """Tests for arbitrary PDF and PSD random field generator."""

    def test_shape_1d(self):
        """Test that 1D field has correct shape."""
        N = 64

        def psd_func(k):
            return np.ones_like(k)

        def pdf_func(z):
            return stats.norm.pdf(z)

        field = arbitrary_pdf_psd_field(dim=1, N=N, psd_func=psd_func, pdf_func=pdf_func)
        assert field.shape == (N,)

    def test_shape_2d(self):
        """Test that 2D field has correct shape."""
        N = 32

        def psd_func(k):
            return np.exp(-(k**2))

        def pdf_func(z):
            return stats.norm.pdf(z)

        field = arbitrary_pdf_psd_field(dim=2, N=N, psd_func=psd_func, pdf_func=pdf_func)
        assert field.shape == (N, N)

    def test_shape_3d(self):
        """Test that 3D field has correct shape."""
        N = 16

        def psd_func(k):
            return np.exp(-k)

        def pdf_func(z):
            return stats.norm.pdf(z)

        field = arbitrary_pdf_psd_field(dim=3, N=N, psd_func=psd_func, pdf_func=pdf_func)
        assert field.shape == (N, N, N)

    def test_pdf_matching(self):
        """Test that the generated field follows the target PDF roughly."""
        N = 128
        dim = 2
        rng = np.random.default_rng(42)

        # Target: Exponential distribution
        # pdf(z) = exp(-z) for z >= 0
        def psd_func(k):
            return 1.0 / (1 + k**2)

        def pdf_func(z):
            val = np.zeros_like(z)
            mask = z >= 0
            val[mask] = np.exp(-z[mask])
            return val

        field = arbitrary_pdf_psd_field(
            dim=dim,
            N=N,
            psd_func=psd_func,
            pdf_func=pdf_func,
            z_min=-1,
            z_max=10,
            rng=rng,
            n_iters=20,
        )

        # Check basic stats
        # Mean of Exp(1) is 1
        # Var of Exp(1) is 1

        # Note: IAAFT might not converge perfectly in small number of iters or small grid
        # But we expect mean to be positive and close to 1

        assert np.mean(field) > 0.8 and np.mean(field) < 1.2
        assert np.var(field) > 0.8 and np.var(field) < 1.2
        # Minimum should be close to 0
        assert np.min(field) >= -0.1

    def test_icdf_input(self):
        """Test providing ICDF directly."""
        N = 64
        dim = 2
        rng = np.random.default_rng(42)

        def psd_func(k):
            return np.ones_like(k)

        # Target: Uniform [0, 1]
        def icdf_func(u):
            return u

        field = arbitrary_pdf_psd_field(
            dim=dim,
            N=N,
            psd_func=psd_func,
            icdf_func=icdf_func,
            rng=rng,
            n_iters=10,
        )

        assert np.min(field) >= 0.0
        assert np.max(field) <= 1.0
        assert np.abs(np.mean(field) - 0.5) < 0.1

    def test_reproducibility(self):
        """Test that RNG produces reproducible results."""
        N = 32
        dim = 2

        def psd_func(k):
            return np.exp(-k)

        def pdf_func(z):
            return stats.norm.pdf(z)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field1 = arbitrary_pdf_psd_field(
            dim=dim, N=N, psd_func=psd_func, pdf_func=pdf_func, rng=rng1
        )
        field2 = arbitrary_pdf_psd_field(
            dim=dim, N=N, psd_func=psd_func, pdf_func=pdf_func, rng=rng2
        )

        np.testing.assert_array_equal(field1, field2)

    def test_invalid_inputs(self):
        """Test error handling."""

        def psd_func(k):
            return k

        def pdf_func(z):
            return z

        # Invalid dim
        with pytest.raises(ValueError):
            arbitrary_pdf_psd_field(dim=4, N=10, psd_func=psd_func, pdf_func=pdf_func)

        # Invalid N
        with pytest.raises(ValueError):
            arbitrary_pdf_psd_field(dim=2, N=0, psd_func=psd_func, pdf_func=pdf_func)

        # Missing PDF/ICDF
        with pytest.raises(ValueError):
            arbitrary_pdf_psd_field(dim=2, N=10, psd_func=psd_func)

        # Missing PSD
        with pytest.raises(ValueError):
            arbitrary_pdf_psd_field(dim=2, N=10, psd_func=None, pdf_func=pdf_func)

