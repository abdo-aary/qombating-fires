from typing import Any, List, Tuple

import numpy as np
from omegaconf import DictConfig
from qadence import RydbergDevice, IdealDevice, Register
from networkx import Graph
from torch import Tensor
import torch


def get_default_register_topology(topology: str, **kwargs: Any) -> Graph:
    """
    Instantiates a register topology using an ideal Rydberg Device

    :param topology: the desired topology
    :param kwargs: arguments of necessary arguments to pass to subsequent loaders
    :return: a Register
    """
    rb_device: RydbergDevice = IdealDevice()
    if topology == 'triangular_lattice':
        reg = Register.triangular_lattice(device_specs=rb_device, **kwargs)
    elif topology == 'honeycomb_lattice':
        reg = Register.honeycomb_lattice(device_specs=rb_device, **kwargs)
    elif topology == 'square':
        reg = Register.square(device_specs=rb_device, **kwargs)
    elif topology == 'line':
        reg = Register.line(device_specs=rb_device, **kwargs)
    elif topology == 'circle':
        reg = Register.circle(device_specs=rb_device, **kwargs)
    elif topology == 'all_to_all':
        reg = Register.all_to_all(device_specs=rb_device, **kwargs)
    else:
        raise Exception(f"Unknown topology: {topology}")
    return reg.graph


def get_topology(cfg: DictConfig) -> Graph:
    """
    Initializes a topology using Qadence.Register

    :param cfg: the configuration of the topology
    :return: the graph of the topology
    """
    if cfg.type == 'default':
        topology = cfg.name
        if topology == ['triangular_lattice', 'honeycomb_lattice']:
            return get_default_register_topology(topology, n_cells_row=int(cfg.n_cells_row),
                                                 n_cells_col=int(cfg.n_cells_col))
        elif topology == 'square':
            return get_default_register_topology(topology, qubits_side=int(cfg.qubits_side))
        elif topology in ['line', 'circle', 'all_to_all']:
            return get_default_register_topology(topology, n_qubits=int(cfg.n_qubits))
        else:
            raise Exception(f"Unknown topology: {topology}")
    elif cfg.type == 'custom':
        raise NotImplementedError("Custom topologies aren't yet handled")

    raise Exception(f"Unknown topology type: {cfg.type}")


def get_binary_representation(num_qubits: int) -> Tensor:
    """
    Returns a tensor of shape (2**num_qubits, num_qubits) where each row is the binary representation
    of the integers 0 to 2**num_qubits - 1, in increasing order.

    The binary representation is in little-endian format:
      - Index 0 is the least significant bit (2^0).
      - Index num_qubits - 1 is the most significant bit (2^(num_qubits-1)).

    Thus, the decimal value computed from each row equals its row index.

    :param num_qubits: The number of qubits (bits) in each representation.
    :return: A tensor containing the binary representations.
    """
    num_values = 2 ** num_qubits
    indices = torch.arange(num_values, dtype=torch.long)
    binary_representation = ((indices.unsqueeze(1) & (1 << torch.arange(num_qubits))) > 0).long()
    return binary_representation  # Shape: (2**num_qubits, num_qubits)


def get_distances_traps(traps: Graph, reference_distance_d_0: float = 3.0) -> Tensor:
    """
    Prepares the pairwise distances between the atom traps

    :param traps: atom traps
    :param reference_distance_d_0: distances are scaled with the used reference distance "d_0". Default = 3.0 [μm]
    :return:
    """
    n_qubits = len(traps)
    distances = torch.zeros(n_qubits, n_qubits)

    if n_qubits > 1:
        reg = Register(traps)
        dict_distances = reg.distances

        # Vectorize the assignment: get row, col indices and corresponding values
        # Convert dict keys and values to tensors
        keys = torch.tensor(list(dict_distances.keys()))
        vals = torch.tensor(list(dict_distances.values()))

        # Split keys into row and column indices
        i, j = keys[:, 0], keys[:, 1]

        # Create the square matrix and fill in the values symmetrically
        distances[i, j] = vals
        distances[j, i] = vals

    return reference_distance_d_0 * distances


def chamfer_kernel(mask1: Tensor, mask2: Tensor, distances: Tensor) -> Tensor:
    """
    Compute the pairwise Chamfer kernel between two batches of binary masks.

    :param mask1: Tensor of shape (N, n_qubits) with binary values.
    :param mask2: Tensor of shape (M, n_qubits) with binary values.
    :param distances: Precomputed distance matrix of shape (n_qubits, n_qubits).
    :return: chamfer_mat: Tensor of shape (N, M) with kernel similarities.
    """
    device = mask1.device

    batch_size1, n_qubits = mask1.shape
    batch_size2, _ = mask2.shape

    # Expand D for broadcasting: shape (1, 1, n, n)
    dist_exp = distances.unsqueeze(0).unsqueeze(0)

    # Expand masks to broadcast over n_qubits indices.
    # mask1_exp: (batch_size1, 1, n_qubits, 1)
    # mask2_exp: (1, batch_size2, 1, n_qubits)
    mask1_exp = mask1.view(batch_size1, 1, n_qubits, 1)
    mask2_exp = mask2.view(1, batch_size2, 1, n_qubits)

    # We want to "mask out" inactive traps. Define a large constant.
    large_const = 1e6

    # Build a tensor of shape (N, M, n, n) that equals D_exp if both mask entries are 1,
    # and large_const otherwise.

    condition = mask1_exp.bool() & mask2_exp.bool()
    active_distances = torch.where(condition, dist_exp, large_const * torch.ones_like(dist_exp, device=device))

    # For each (i, j) and for each active index k in mask1, compute the minimum distance to an active index in
    # mask2. min_over_mask2 has shape (N, M, n).
    min_over_mask2, _ = active_distances.min(dim=-1)

    # For each sample i in mask1, count the number of active traps.
    m1 = mask1.sum(dim=1).clamp(min=1)  # shape (N,)
    # Average the minima over active indices for each sample i, for each j.
    # We want to compute, for each i,j: term1[i,j] = (1/(2*m1[i])) * sum_{k active in mask1[i]} min_over_mask2[i,j,k].
    term1 = (min_over_mask2 * mask1.unsqueeze(1)).sum(dim=2) / (2 * m1.unsqueeze(1))  # shape (N, M)

    # Similarly, compute the reverse direction.
    min_over_mask1, _ = active_distances.min(dim=-2)  # shape (N, M, n)
    m2 = mask2.sum(dim=1).clamp(min=1)  # shape (M,)
    term2 = (min_over_mask1 * mask2.unsqueeze(0)).sum(dim=2) / (2 * m2.unsqueeze(0))  # shape (N, M)

    # Chamfer distance is the sum of both terms.
    d_chamfer = term1 + term2  # shape (N, M)

    # Finally, define the kernel similarity as an RBF on the Chamfer distance.
    chamfer_mat = torch.exp(-d_chamfer)
    return chamfer_mat


def precompute_string_kernel(binary_representation: torch.Tensor) -> torch.Tensor:
    """
    Precomputes the string kernel Gram matrix K_B for the global outcome space.

    Args:
        binary_representation: Tensor of shape (K, n) where K = 2^n, with each row being the binary representation of an
        outcome.

    Returns:
        kernel_b: Tensor of shape (K, K) with
             kernel_b[b, b'] = exp(-HammingDistance(B[b], B[b'])).
    """
    # Compute pairwise Hamming distances (since all bitstrings have equal length, Levenshtein = Hamming)
    # Here we use broadcasting: difference is 1 if bits differ.
    diff = (binary_representation.unsqueeze(0) != binary_representation.unsqueeze(1))
    hamming_dist = diff.sum(dim=-1)  # shape (K, K)
    kernel_b = torch.exp(-hamming_dist)
    return kernel_b


def compute_intra_mmd_expectation(dist: torch.Tensor, kernel_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the intra-sample string kernel expectation for each sample.

    Args:
        dist: Tensor of shape (N, K) representing marginalized probability distributions.
        kernel_b: Precomputed string kernel Gram matrix of shape (K, K).

    Returns:
        intra_expect: Tensor of shape (N,) where E[i] = sum_{b,b'} dist[i,b] * dist[i,b'] * KB[b,b'].
    """
    # Use Einstein summation to compute the double sum for each sample.
    intra_expect = torch.einsum("ib,ic,bc->i", dist, dist, kernel_b)
    return intra_expect


def compute_cross_mmd_expectation(dist1: torch.Tensor, dist2: torch.Tensor, kernel_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross (inter-sample) expectation between two batches.

    Args:
        dist1: Tensor of shape (N, K) (probability distribution for batch 1).
        dist2: Tensor of shape (M, K) (for batch 2).
        kernel_b: Precomputed string kernel Gram matrix of shape (K, K).

    Returns:
        E_cross: Tensor of shape (N, M) with
                 E_cross[i,j] = sum_{b,b'} dist1[i,b] * dist2[j,b'] * kernel_b[b,b'].
    """
    cross_expect = torch.einsum("ib,jc,bc->ij", dist1, dist2, kernel_b)
    return cross_expect


def mmd_kernel(expect_intra1: torch.Tensor, expect_intra2: torch.Tensor,
               cross_exp: torch.Tensor, chamfer_mat: torch.Tensor) -> torch.Tensor:
    """
    Computes the end MMD kernel value between two batches using:

    d_M^2 = E(intra)[x] + E(intra)[x'] - 2 * (K_C(x,x') * E_cross(x,x'))

    and returns the kernel as:
      k(x,x') = exp( - sqrt(d_M^2) ).

    Args:
        expect_intra1: Tensor of shape (N,) with intra-sample expectations for batch 1.
        expect_intra2: Tensor of shape (M,) for batch 2.
        cross_exp: Tensor of shape (N, M) with cross expectations.
        chamfer_mat: Tensor of shape (N, M) with the spatial (Chamfer) kernel values.

    Returns:
        Kernel matrix of shape (N, M).
    """
    d2 = expect_intra1.unsqueeze(1) + expect_intra2.unsqueeze(0) - 2 * (chamfer_mat * cross_exp)

    return torch.exp(-torch.sqrt(d2 + 1e-10))  # Add small terms for numerical stability


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
    d2 = sum(nearest_neighbor_dist(arrangements2, arrangements1) for
             arrangements2 in arrangements2) / (len(arrangements2) or 1)
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
