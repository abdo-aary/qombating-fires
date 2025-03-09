import numpy as np
import pytest

from bassir.prep.utils import get_augmented_agg_train_data


def test_no_minority():
    """
    Test that if there are no minority samples (y == 1), the function returns the original data.
    """
    X = np.random.rand(10, 5)  # 10 samples, 5 features each
    y = np.zeros(10)  # all majority (0) labels
    X_aug, y_aug = get_augmented_agg_train_data(X, y, augmentation_factor=3)
    np.testing.assert_array_equal(X_aug, X)
    np.testing.assert_array_equal(y_aug, y)


def test_shape_augmentation():
    """
    Test that the augmented data has the expected shape.
    For example, if there are 20 original samples with 4 minority samples, and augmentation_factor=3,
    the augmented data should have 20 + (4 * 3) = 32 samples.
    """
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.zeros(20)
    minority_indices = [3, 7, 12, 18]  # 4 minority samples
    y[minority_indices] = 1
    augmentation_factor = 3
    X_aug, y_aug = get_augmented_agg_train_data(X, y, augmentation_factor=augmentation_factor)
    expected_samples = 20 + len(minority_indices) * augmentation_factor
    assert X_aug.shape[0] == expected_samples
    assert y_aug.shape[0] == expected_samples


def test_synthetic_labels():
    """
    Test that all synthetic samples (appended at the end) have label 1.
    """
    np.random.seed(42)
    X = np.random.rand(15, 4)
    y = np.zeros(15)
    minority_indices = [2, 5, 10]  # 3 minority samples
    y[minority_indices] = 1
    augmentation_factor = 2
    X_aug, y_aug = get_augmented_agg_train_data(X, y, augmentation_factor=augmentation_factor)
    n_min = len(minority_indices)
    # The synthetic samples are appended at the end.
    synthetic_labels = y_aug[-(n_min * augmentation_factor):]
    np.testing.assert_array_equal(synthetic_labels, np.ones(n_min * augmentation_factor))


def test_synthetic_not_duplicate():
    """
    Test that the synthetic samples differ (are not identical) from the original minority samples.
    This ensures that the interpolation is actually modifying the features.
    """
    np.random.seed(42)
    X = np.random.rand(10, 3)
    y = np.zeros(10)
    minority_indices = [0, 1]  # two minority samples
    y[minority_indices] = 1
    augmentation_factor = 1
    X_aug, y_aug = get_augmented_agg_train_data(X, y, augmentation_factor=augmentation_factor)
    # Synthetic samples are the last two rows
    synthetic_samples = X_aug[-len(minority_indices):]
    for i, idx in enumerate(minority_indices):
        diff = np.linalg.norm(synthetic_samples[i] - X[idx])
        assert diff > 1e-5, f"Synthetic sample for original index {idx} is too similar to the original."


def test_synthetic_within_bounds():
    """
    For each synthetic sample (the ones appended at the end),
    verify that each feature value lies between the min and max of the original minority samples
    (i.e. the synthetic samples are convex combinations of minority samples).
    """
    np.random.seed(42)
    # Create a simple dataset: 10 samples, 3 features
    X = np.random.rand(10, 3)
    y = np.zeros(10)
    # Mark a couple of samples as minority (label 1)
    minority_indices = [0, 1]
    y[minority_indices] = 1
    augmentation_factor = 2

    X_aug, y_aug = get_augmented_agg_train_data(X, y, augmentation_factor=augmentation_factor)
    n_min = len(minority_indices)
    # Synthetic samples are the last n_min * augmentation_factor samples
    synthetic_samples = X_aug[-(n_min * augmentation_factor):]

    # Compute componentwise min and max from the original minority samples.
    X_min_orig = X[minority_indices]
    lower_bound = X_min_orig.min(axis=0)
    upper_bound = X_min_orig.max(axis=0)

    # Check for each synthetic sample that each feature value lies within these bounds.
    for sample in synthetic_samples:
        assert np.all(sample >= lower_bound - 1e-6), "A synthetic sample is below the lower bound."
        assert np.all(sample <= upper_bound + 1e-6), "A synthetic sample is above the upper bound."


if __name__ == "__main__":
    pytest.main([__file__])
