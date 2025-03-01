import numpy as np
from typing import List, Tuple
import torch


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


def binary_activation_with_min_activation(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Applies a deterministic hard threshold to logits to produce a binary mask,
    ensuring that at least one element is active in each row.
    In the forward pass, the output is hard (0 or 1), but the backward pass uses
    the gradient of a sigmoid (via a straight-through estimator).

    Args:
        logits: Input logits tensor.
        threshold: Threshold value for hard binarization.

    Returns:
        A tensor of the same shape as logits with binary values, with at least one 1 per row.
    """
    # Compute soft activations using the sigmoid.
    soft = torch.sigmoid(logits)

    # Hard threshold to obtain binary outputs.
    hard = (soft >= threshold).float()

    # Check for rows that are entirely zero.
    row_sum = hard.sum(dim=-1, keepdim=True)
    condition = row_sum == 0.0  # Boolean tensor: True for rows with all zeros.

    # For rows that are all zeros, set the entry corresponding to the maximum soft value to 1.
    if condition.any():
        max_indices = torch.argmax(soft, dim=-1)  # shape: (batch_size,)
        one_hot = torch.nn.functional.one_hot(max_indices, num_classes=logits.size(-1)).float()
        # Replace rows where condition is True with the one_hot vector.
        hard = torch.where(condition, one_hot, hard)

    # Use the straight-through estimator: forward pass uses hard values, backward pass uses gradients from soft.
    return hard + (soft - soft.detach())
