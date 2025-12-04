"""Tests for random field generators."""

import numpy as np
import pytest

from rfgen import selfaffine_field, matern_field


class TestSelfAffineGenerator:
    """Tests for self-affine random field generator."""

    def test_shape_1d(self):
        """Test that 1D field has correct shape."""
        N = 64
        field = selfaffine_field(dim=1, N=N)
        assert field.shape == (N,)

    def test_shape_2d(self):
        """Test that 2D field has correct shape."""
        N = 64
        field = selfaffine_field(dim=2, N=N)
        assert field.shape == (N, N)

    def test_shape_3d(self):
        """Test that 3D field has correct shape."""
        N = 32
        field = selfaffine_field(dim=3, N=N)
        assert field.shape == (N, N, N)

    def test_reproducibility_noise_true(self):
        """Test that RNG produces reproducible results with noise=True."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field1 = selfaffine_field(dim=2, N=64, noise=True, rng=rng1)
        field2 = selfaffine_field(dim=2, N=64, noise=True, rng=rng2)

        np.testing.assert_array_equal(field1, field2)

    def test_reproducibility_noise_false(self):
        """Test that RNG produces reproducible results with noise=False."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field1 = selfaffine_field(dim=2, N=64, noise=False, rng=rng1)
        field2 = selfaffine_field(dim=2, N=64, noise=False, rng=rng2)

        np.testing.assert_array_equal(field1, field2)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        field1 = selfaffine_field(dim=2, N=64, rng=rng1)
        field2 = selfaffine_field(dim=2, N=64, rng=rng2)

        assert not np.allclose(field1, field2)

    def test_noise_parameter_difference(self):
        """Test that noise=True and noise=False produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field_noise = selfaffine_field(dim=2, N=64, noise=True, rng=rng1)
        field_ideal = selfaffine_field(dim=2, N=64, noise=False, rng=rng2)

        # They should be different (different generation methods)
        assert not np.allclose(field_noise, field_ideal)

    def test_real_valued(self):
        """Test that output is real-valued."""
        field = selfaffine_field(dim=2, N=64)
        assert np.isreal(field).all()

    def test_real_valued_noise_false(self):
        """Test that output is real-valued with noise=False."""
        field = selfaffine_field(dim=2, N=64, noise=False)
        assert np.isreal(field).all()

    def test_invalid_k_range(self):
        """Test that invalid k range raises error."""
        with pytest.raises(ValueError):
            selfaffine_field(k_low=0.3, k_high=0.1)

    def test_invalid_k_high(self):
        """Test that k_high > 0.5 raises error."""
        with pytest.raises(ValueError):
            selfaffine_field(k_high=0.6)

    def test_invalid_dimension(self):
        """Test that invalid dimension raises error."""
        with pytest.raises(ValueError):
            selfaffine_field(dim=4)

    def test_plateau_option(self):
        """Test that plateau option works."""
        field = selfaffine_field(dim=2, N=64, plateau=True)
        assert field.shape == (64, 64)

    def test_plateau_with_noise_false(self):
        """Test that plateau works with noise=False."""
        field = selfaffine_field(dim=2, N=64, plateau=True, noise=False)
        assert field.shape == (64, 64)


class TestMaternGenerator:
    """Tests for Matérn random field generator."""

    def test_shape_2d(self):
        """Test that 2D field has correct shape."""
        N = 64
        field = matern_field(dim=2, N=N)
        assert field.shape == (N, N)

    def test_real_valued(self):
        """Test that output is real-valued."""
        field = matern_field(dim=2, N=64)
        assert np.isreal(field).all()

    def test_real_valued_noise_false(self):
        """Test that output is real-valued with noise=False."""
        field = matern_field(dim=2, N=64, noise=False)
        assert np.isreal(field).all()

    def test_reproducibility_noise_true(self):
        """Test reproducibility with noise=True."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field1 = matern_field(dim=2, N=64, noise=True, rng=rng1)
        field2 = matern_field(dim=2, N=64, noise=True, rng=rng2)

        np.testing.assert_array_equal(field1, field2)

    def test_reproducibility_noise_false(self):
        """Test reproducibility with noise=False."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field1 = matern_field(dim=2, N=64, noise=False, rng=rng1)
        field2 = matern_field(dim=2, N=64, noise=False, rng=rng2)

        np.testing.assert_array_equal(field1, field2)

    def test_noise_parameter_difference(self):
        """Test that noise=True and noise=False produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        field_noise = matern_field(dim=2, N=64, noise=True, rng=rng1)
        field_ideal = matern_field(dim=2, N=64, noise=False, rng=rng2)

        assert not np.allclose(field_noise, field_ideal)

    def test_invalid_nu(self):
        """Test that non-positive nu raises error."""
        with pytest.raises(ValueError):
            matern_field(nu=0)
        with pytest.raises(ValueError):
            matern_field(nu=-1)

    def test_invalid_correlation_length(self):
        """Test that non-positive correlation length raises error."""
        with pytest.raises(ValueError):
            matern_field(correlation_length=0)

    def test_invalid_k_range(self):
        """Test that invalid k range raises error."""
        with pytest.raises(ValueError):
            matern_field(k_low=0.3, k_high=0.1)
