import numpy as np
from typing import List, Tuple
import torch as tr


# --- Levenshtein Distance for String Kernel ---
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# --- Chamfer Distance for Point Cloud Kernel ---
def chamfer_distance(arrangements1: List[Tuple[float, float]], arrangements2: List[Tuple[float, float]]) -> float:
    """Compute Chamfer distance between two point clouds."""

    def nearest_neighbor_dist(point, points):
        return min(np.linalg.norm(np.array(point) - np.array(p)) for p in points) if points else 0

    d1 = sum(nearest_neighbor_dist(r, arrangements2) for r in arrangements1) / (len(arrangements1) or 1)
    d2 = sum(nearest_neighbor_dist(arrangements2, arrangements1) for arrangements2 in arrangements2) / (
                len(arrangements2) or 1)
    return (d1 + d2) / 2


def binary_activation(logits: tr.Tensor, tau: float = 1.0, threshold: float = 0.5) -> tr.Tensor:
    """
    Applies a relaxed binary (Bernoulli) activation on logits using a relaxed Bernoulli (Binary Concrete)
    distribution, thresholds the values, and ensures that at least one entry is 1.

    :param logits: The input logits tensor.
    :param tau: Temperature for the relaxed Bernoulli distribution.
    :param threshold: Threshold value to convert relaxed values into hard binary outputs.
    :return: A tensor of the same shape as logits with binary values (0s and 1s) that is differentiable.
    """
    # Define the relaxed Bernoulli distribution and sample using the reparameterization trick.
    dist = tr.distributions.RelaxedBernoulli(temperature=tau, logits=logits)
    y = dist.rsample()

    # Hard threshold to obtain binary outputs.
    y_hard = (y > threshold).float()

    # Check if all entries in the sample are zero.
    all_zero = (y_hard.sum(dim=-1, keepdim=True) == 0)
    if all_zero.any():
        # Force at least one active element by setting the index with the maximum logit to 1.
        max_indices = tr.argmax(logits, dim=-1)
        one_hot = tr.nn.functional.one_hot(max_indices, num_classes=logits.size(-1)).float()
        y_hard = y_hard + all_zero.float() * one_hot
        y_hard = (y_hard > threshold).float()

    # Use the straight-through estimator: use hard values for forward pass while keeping gradients from y.
    return (y_hard - y).detach() + y
