"""Tests for random field analysis tools."""

import numpy as np
import pytest

from rfgen import (
    selfaffine_field,
    autocorrelation_1d,
    autocorrelation_2d,
    autocorrelation_nd,
    correlation_length,
    psd_1d,
    psd_2d,
    psd_radial_average,
    spectral_moment,
    spectral_moment_1d,
    compute_standard_moments,
)


class TestAutocorrelation:
    """Tests for autocorrelation functions."""

    def test_autocorrelation_1d_shape(self):
        """Test that 1D autocorrelation has correct shape."""
        signal = np.random.randn(256)
        R = autocorrelation_1d(signal)
        assert R.shape == signal.shape

    def test_autocorrelation_1d_normalized(self):
        """Test that normalized autocorrelation starts at 1."""
        signal = np.random.randn(256)
        R = autocorrelation_1d(signal, normalize=True)
        np.testing.assert_almost_equal(R[0], 1.0)

    def test_autocorrelation_1d_unnormalized(self):
        """Test unnormalized autocorrelation."""
        signal = np.random.randn(256)
        R = autocorrelation_1d(signal, normalize=False)
        # R[0] should be approximately variance * N / N^2 = variance / N
        # Actually for FFT-based: R[0] = sum(signal^2) / N^2
        expected = np.sum(signal**2) / len(signal)**2
        np.testing.assert_almost_equal(R[0], expected, decimal=10)

    def test_autocorrelation_2d_shape(self):
        """Test that 2D autocorrelation has correct shape."""
        field = np.random.randn(64, 64)
        R = autocorrelation_2d(field)
        assert R.shape == field.shape

    def test_autocorrelation_2d_normalized(self):
        """Test that normalized 2D autocorrelation starts at 1."""
        field = np.random.randn(64, 64)
        R = autocorrelation_2d(field, normalize=True)
        np.testing.assert_almost_equal(R[0, 0], 1.0)

    def test_autocorrelation_nd_3d(self):
        """Test 3D autocorrelation."""
        field = np.random.randn(16, 16, 16)
        R = autocorrelation_nd(field)
        assert R.shape == field.shape
        np.testing.assert_almost_equal(R[0, 0, 0], 1.0)

    def test_correlation_length_finite(self):
        """Test that correlation length is finite for random field."""
        rng = np.random.default_rng(42)
        field = selfaffine_field(dim=1, N=512, rng=rng)
        R = autocorrelation_1d(field)
        l_corr = correlation_length(R, threshold=0.0, spacing=1/512)
        assert 0 < l_corr < 1


class TestPSD:
    """Tests for power spectral density functions."""

    def test_psd_1d_shape(self):
        """Test that 1D PSD has correct shape."""
        signal = np.random.randn(256)
        k, psd = psd_1d(signal)
        # Should return positive frequencies only (excluding DC)
        assert len(k) == 127  # (N-1)//2 for N=256
        assert len(psd) == len(k)

    def test_psd_1d_positive(self):
        """Test that PSD is non-negative."""
        signal = np.random.randn(256)
        k, psd = psd_1d(signal)
        assert np.all(psd >= 0)

    def test_psd_2d_shape(self):
        """Test that 2D PSD has correct shape."""
        field = np.random.randn(64, 64)
        k, psd = psd_2d(field)
        assert k.shape == field.shape
        assert psd.shape == field.shape

    def test_psd_radial_average(self):
        """Test radially averaged PSD."""
        field = np.random.randn(64, 64)
        k, psd = psd_radial_average(field)
        assert len(k) == len(psd)
        assert np.all(psd >= 0)


class TestSpectralMoments:
    """Tests for spectral moment computation."""

    def test_spectral_moment_m00(self):
        """Test that m00 is related to variance."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        m00 = spectral_moment(field, 0, 0, spacing=1.0)
        # m00 should be close to variance for a white noise field
        assert m00 > 0

    def test_spectral_moment_1d(self):
        """Test 1D spectral moment."""
        signal = np.random.randn(256)
        m0 = spectral_moment_1d(signal, 0)
        assert m0 > 0

    def test_spectral_moment_invalid_indices(self):
        """Test that negative indices raise error."""
        field = np.random.randn(64, 64)
        with pytest.raises(ValueError):
            spectral_moment(field, -1, 0)

    def test_compute_standard_moments(self):
        """Test that all standard moments are computed."""
        field = np.random.randn(64, 64)
        moments = compute_standard_moments(field)
        expected_keys = ["m00", "m10", "m01", "m20", "m02", "m11", "m40", "m04", "m22"]
        for key in expected_keys:
            assert key in moments
            assert isinstance(moments[key], float)

    def test_isotropy_check(self):
        """Test that isotropic field has similar moments in x and y."""
        # Use a larger field for better statistics
        rng = np.random.default_rng(42)
        field = selfaffine_field(dim=2, N=256, rng=rng)
        moments = compute_standard_moments(field, spacing=1/256)

        # For isotropic field, m20 ≈ m02 and m40 ≈ m04
        # Allow some tolerance due to finite size effects
        ratio_m2 = moments["m20"] / moments["m02"]
        ratio_m4 = moments["m40"] / moments["m04"]

        assert 0.5 < ratio_m2 < 2.0
        assert 0.5 < ratio_m4 < 2.0
