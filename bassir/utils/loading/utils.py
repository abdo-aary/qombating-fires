import torch
import math


def generate_complex_synthetic_data(n_samples: int = 1000, dim: int = 12, noise_std: float = 0.1):
    """
    Generates a challenging synthetic dataset for binary classification.

    The inputs are sampled uniformly in [0,1]^dim. The decision function is a sum of several nonlinear
    and interaction terms:

        f1 = sin(pi * x1) * cos(pi * x2)
        f2 = x3^2 - x4
        f3 = tanh(x5 + x6)
        f4 = sin(2*pi * x7 * x8)
        f5 = cos(3*pi * x9) * sin(3*pi * x10)
        f6 = x11 - x12

    A small amount of Gaussian noise is added to f, and the median value is used as a threshold.
    The binary label y is 1 if f > threshold, and 0 otherwise.

    Args:
        n_samples: Number of samples to generate.
        dim: Dimensionality of the input (must be at least 12 for this function).
        noise_std: Standard deviation of the additive Gaussian noise.

    Returns:
        x: Tensor of shape (n_samples, dim) with input features.
        y: Tensor of shape (n_samples,) with binary labels (0 or 1).
    """
    assert dim >= 12, "This data generator requires at least 12 dimensions."
    # Sample inputs uniformly in [0, 1]
    x = torch.rand(n_samples, dim)

    # Compute several nonlinear and interaction features.
    f1 = torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1])
    f2 = x[:, 2] ** 2 - x[:, 3]
    f3 = torch.tanh(x[:, 4] + x[:, 5])
    f4 = torch.sin(2 * math.pi * x[:, 6] * x[:, 7])
    f5 = torch.cos(3 * math.pi * x[:, 8]) * torch.sin(3 * math.pi * x[:, 9])
    f6 = x[:, 10] - x[:, 11]

    # Combine the terms into a single decision function
    f = f1 + f2 + f3 + f4 + f5 + f6

    # Add noise
    f = f + noise_std * torch.randn(n_samples)

    # Use the median of f as the threshold to roughly balance classes.
    threshold = f.median()
    y = (f > threshold).float()

    return x, y
