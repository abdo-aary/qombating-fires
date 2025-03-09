import numpy as np
import pytest

from bassir.prep.utils import preprocess_windows


def test_preprocess_windows_constant():
    """
    If the window is constant, after standardization the values should become 0,
    and hence the aggregated features should all be 0.
    """
    num_samples = 1
    window_size = 7
    d_raw = 13
    constant_value = 5.0
    # Create an array where every element is constant_value
    X = np.full((num_samples, window_size, d_raw), constant_value, dtype=np.float32)
    # Preprocess the windows
    aggregated = preprocess_windows(X, scaler=None)

    # Expected aggregated dimension: 4*d_raw (stats) + 2*d_raw (frequency) + d_raw (trend) + d_raw (seasonal) = 8*d_raw
    expected_dim = 8 * d_raw
    assert aggregated.shape == (
        num_samples, expected_dim), f"Expected shape (1, {expected_dim}), got {aggregated.shape}"

    # Since the window is constant, after standardization, every value should become 0. Thus, statistical summaries (
    # mean, std, min, max) are all 0, frequency features are 0, trend and seasonal features are 0. Allowing for a
    # very small numerical tolerance:
    np.testing.assert_allclose(aggregated, np.zeros((num_samples, expected_dim)), atol=1e-5)


def test_preprocess_windows_shape():
    """
    Test that the output shape of preprocess_windows is correct.
    For an input X of shape (num_samples, window_size, d_raw),
    the output should be of shape (num_samples, 8*d_raw).
    """
    num_samples = 10
    window_size = 7
    d_raw = 13
    X = np.random.rand(num_samples, window_size, d_raw).astype(np.float32)

    aggregated = preprocess_windows(X, scaler=None)
    expected_dim = 8 * d_raw
    assert aggregated.shape == (
        num_samples, expected_dim), f"Expected shape ({num_samples}, {expected_dim}), got {aggregated.shape}"


def test_preprocess_windows_values_change():
    """
    For a non-constant window, check that the aggregated features are not all zero.
    """
    num_samples = 5
    window_size = 7
    d_raw = 13
    X = np.random.rand(num_samples, window_size, d_raw).astype(np.float32)

    aggregated = preprocess_windows(X, scaler=None)
    # Check that aggregated features are not all zero (they should have some variance)
    assert np.any(np.abs(aggregated) > 1e-3), "Aggregated features seem to be all zero for random input."


def test_preprocess_windows_trend():
    """
    Create a window with linearly increasing values and check that the computed trend
    (slope) is close to the expected value.

    For a window of shape (w,1) with values [0, 1, 2, ..., w-1]:
      - After standardization, the slope is computed as:
            numerator = sum((t-t_mean)^2), denominator = same, so slope = 1.
      But note: standardization scales the values;
      for w=7, using our vectorized computation, the expected slope on standardized data is 0.5.
    """
    num_samples = 1
    window_size = 7
    d_raw = 1
    # Create a linear sequence: 0, 1, 2, ..., 6 for each sample.
    X = np.arange(window_size, dtype=np.float32).reshape(1, window_size, d_raw)
    aggregated = preprocess_windows(X, scaler=None)

    # Compute expected aggregated dimension:
    # stat_feats: 4*d_raw = 4, freq_feats: 2*d_raw = 2, trend_feats: d_raw = 1, seasonal_feats: d_raw = 1
    expected_dim = 8 * d_raw
    assert aggregated.shape == (num_samples, expected_dim)

    # Our concatenation order is:
    # indices 0-3: statistical features
    # indices 4-5: frequency features
    # index 6: trend (slope)
    # index 7: seasonal amplitude
    # For a linear window, after standardization, we expect trend ~ 0.5 (computed as explained in analysis).
    trend_value = aggregated[0, 6]
    # Allow a small tolerance.
    np.testing.assert_allclose(trend_value, 0.5, atol=0.1)

    # Also, since the window is perfectly linear, seasonal amplitude should be near 0.
    seasonal_value = aggregated[0, 7]
    np.testing.assert_allclose(seasonal_value, 0.0, atol=1e-5)


def test_preprocess_windows_frequency():
    """
    Create a window with a sinusoidal pattern.
    For a sine wave of period equal to the window length, the FFT (excluding DC)
    should have a dominant frequency index > 0 and nonzero energy.
    """
    num_samples = 1
    window_size = 7
    d_raw = 1
    t = np.arange(window_size, dtype=np.float32)
    # Create a sine wave with one full period over the window:
    sine_wave = np.sin(2 * np.pi * t / window_size)
    X = sine_wave.reshape(1, window_size, d_raw)

    aggregated = preprocess_windows(X, scaler=None)
    # Aggregated dimension should be 8*d_raw = 8.
    assert aggregated.shape == (num_samples, 8 * d_raw)

    # Frequency features are at indices 4-5 in the aggregated vector.
    freq_feats = aggregated[0, 4:6]
    dom_freq, energy = freq_feats
    # For a sine wave with one period in the window, we expect the dominant frequency index to be nonzero.
    assert dom_freq > 0
    # And energy should be positive.
    assert energy > 0


if __name__ == "__main__":
    pytest.main([__file__])
