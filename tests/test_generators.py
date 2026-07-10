"""Tests for random field generators."""

import numpy as np
import pytest

from rfgen import compute_standard_moments, matern_field, matern_spectrum, rms_quantities, selfaffine_field
from rfgen.generators._fft import real_fft_radial_frequency_grid, selfaffine_filter


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

    def test_ideal_spectrum_matches_target_amplitudes(self):
        """The real FFT retains the requested amplitude of each mode."""
        n = 32
        field = selfaffine_field(dim=2, N=n, Hurst=0.7, noise=False, rng=np.random.default_rng(4))
        expected = selfaffine_filter(2, n, 0.7, 0.03, 0.3, False)

        np.testing.assert_allclose(np.abs(np.fft.rfftn(field)), expected, rtol=1e-12, atol=1e-12)

    def test_ideal_spectrum_recovers_hurst_exponent(self):
        """The exact spectrum has the requested self-affine power-law exponent."""
        n = 64
        hurst = 0.7
        field = selfaffine_field(
            dim=2, N=n, Hurst=hurst, k_low=4 / n, k_high=20 / n, noise=False, rng=np.random.default_rng(5)
        )
        k = real_fft_radial_frequency_grid(2, n)
        power = np.abs(np.fft.rfftn(field)) ** 2
        mask = (k >= 4 / n) & (k <= 20 / n)
        slope, _ = np.polyfit(np.log(k[mask]), np.log(power[mask]), 1)

        np.testing.assert_allclose(-slope, 2 + 2 * hurst, rtol=1e-12, atol=1e-12)

    def test_ideal_spectrum_has_correct_rms_moments(self):
        """Height, slope, and curvature moments agree with spectral derivatives."""
        n = 64
        field = selfaffine_field(
            dim=2, N=n, Hurst=0.5, k_low=4 / n, k_high=20 / n, noise=False, rng=np.random.default_rng(6)
        )
        spectrum = np.fft.fftn(field)
        angular_frequency = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
        derivative = np.fft.ifftn(1j * angular_frequency[:, None] * spectrum).real
        curvature = np.fft.ifftn(-(angular_frequency[:, None] ** 2) * spectrum).real

        moments = compute_standard_moments(field, spacing=1 / n)
        rms = rms_quantities(field, spacing=1 / n)

        np.testing.assert_allclose(moments["m00"], np.var(field), rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(moments["m20"], np.mean(derivative**2), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(moments["m40"], np.mean(curvature**2), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(moments["m20"], moments["m02"], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(rms["rms_slope_x"], np.sqrt(np.mean(derivative**2)), rtol=1e-12)

    def test_filtered_noise_power_is_unbiased_in_ensemble(self):
        """Filtered white noise has the requested modal power in expectation."""
        n = 32
        amplitude = selfaffine_filter(2, n, 0.0, 4 / n, 12 / n, False)
        observed_power = np.zeros_like(amplitude)

        for seed in range(48):
            field = selfaffine_field(
                dim=2, N=n, Hurst=0.0, k_low=4 / n, k_high=12 / n, rng=np.random.default_rng(seed)
            )
            observed_power += np.abs(np.fft.rfftn(field)) ** 2 / n**2

        observed_power /= 48
        mask = amplitude > 0
        relative_rms_error = np.linalg.norm(observed_power[mask] - amplitude[mask] ** 2) / np.linalg.norm(
            amplitude[mask] ** 2
        )
        assert relative_rms_error < 0.3


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

    @pytest.mark.parametrize(("dim", "n"), [(1, 32), (2, 24), (3, 12)])
    def test_ideal_spectrum_matches_target_amplitudes(self, dim, n):
        """The real FFT retains every requested independent Matérn amplitude."""
        nu = 1.5
        correlation_length = 0.1
        sigma = 1.7
        k_low = 3 / n
        k_high = 0.3
        field = matern_field(
            dim=dim,
            N=n,
            nu=nu,
            correlation_length=correlation_length,
            sigma=sigma,
            k_low=k_low,
            k_high=k_high,
            noise=False,
            rng=np.random.default_rng(7),
        )
        k = real_fft_radial_frequency_grid(dim, n)
        expected = np.zeros_like(k)
        mask = (k >= k_low) & (k <= k_high)
        expected[mask] = np.sqrt(matern_spectrum(k[mask], sigma, dim, nu, correlation_length))

        np.testing.assert_allclose(np.abs(np.fft.rfftn(field)), expected, rtol=1e-12, atol=1e-12)

    def test_sigma_scales_the_discrete_matern_field(self):
        """Sigma is a linear spectrum scale, even after finite-band truncation."""
        kwargs = dict(dim=2, N=32, nu=1.5, correlation_length=0.1, k_low=3 / 32, k_high=0.3, noise=False)
        field_sigma_1 = matern_field(sigma=1.0, rng=np.random.default_rng(8), **kwargs)
        field_sigma_2 = matern_field(sigma=2.0, rng=np.random.default_rng(8), **kwargs)

        np.testing.assert_allclose(field_sigma_2, 2.0 * field_sigma_1, rtol=1e-12, atol=1e-15)

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
