import math
import pytest

from bassir.utils.qutils import *


def test_binary_representation_order():
    num_qubits = 5
    bin_rep = get_binary_representation(num_qubits)
    num_values = 2 ** num_qubits

    # Check the shape is as expected.
    assert bin_rep.shape == (num_values, num_qubits)

    # For each row, convert the binary representation (little-endian) to its decimal value.
    def bitstring_to_int(bit_row):
        return sum(bit * (2 ** i) for i, bit in enumerate(bit_row.tolist()))

    values = [bitstring_to_int(bin_rep[i]) for i in range(num_values)]

    # Check that the values are in strictly increasing order.
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1], f"Row {i} ({values[i]}) is not less than row {i + 1} ({values[i + 1]})"


def test_chamfer_kernel_complex():
    """
    Test the chamfer_kernel function for a 4-qubit system using non-trivial binary masks.
    We verify the shape, range, and a specific computed value.
    """
    # Define two batches of masks.
    # For a 4-qubit system, each mask is of length 4.
    # Batch 1: 3 samples.
    mask1 = torch.tensor([
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ])
    # Batch 2: 2 samples.
    mask2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 0, 0, 1]
    ])

    # Define a fixed distance matrix for 4 traps.
    # For example, we use:
    # D = [[0, 1, 2, 3],
    #      [1, 0, 1, 2],
    #      [2, 1, 0, 1],
    #      [3, 2, 1, 0]]
    distances = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0]
    ])

    # Compute the Chamfer kernel matrix.
    chamfer_mat = chamfer_kernel(mask1, mask2, distances)
    # Expected shape: (N, M) = (3, 2)
    assert chamfer_mat.shape == (3, 2), f"Expected shape (3, 2), got {chamfer_mat.shape}"

    # Check that all values are between 0 and 1.
    assert torch.all(chamfer_mat >= 0) and torch.all(chamfer_mat <= 1), "Kernel values not in [0,1]"

    # Manually compute expected Chamfer distance for the pair:
    # For mask1[0] = [1, 0, 1, 1] and mask2[0] = [1, 1, 1, 1]:
    # Active indices for mask1[0]: {0, 2, 3}.
    # Active indices for mask2[0]: {0, 1, 2, 3}.
    #
    # For each active index k in mask1[0]:
    #   k=0: distances from 0 to active indices in mask2[0]: [0, 1, 2, 3] → min = 0.
    #   k=2: distances from 2: [2, 1, 0, 1] → min = 0.
    #   k=3: distances from 3: [3, 2, 1, 0] → min = 0.
    # term1 = (0+0+0) / (2 * 3) = 0.
    #
    # For each active index in mask2[0]:
    #   Active indices for mask2[0] are all: {0,1,2,3}.
    #   For each index, compute the minimum distance to an active index in mask1[0] = {0,2,3}.
    #   For index 0: distances from 0: [0,2,3] → min = 0.
    #   For index 1: distances from 1: [1, 1, 2] → min = 1.
    #   For index 2: distances from 2: [2, 0, 1] → min = 0.
    #   For index 3: distances from 3: [3, 1, 0] → min = 0.
    # term2 = (0+1+0+0) / (2 * 4) = 1/8 = 0.125.
    # Total Chamfer distance = 0 + 0.125 = 0.125.
    # Kernel similarity = exp(-0.125) ≈ 0.8825.
    expected_similarity = math.exp(-0.125)
    computed_val = chamfer_mat[0, 0].item()
    assert abs(
        computed_val - expected_similarity) < 1e-3, f"Expected {expected_similarity} at (0,0), got {computed_val}"


def test_precompute_string_kernel():
    # For 2 qubits, K = 4. Define binary_rep as:
    binary_rep = torch.tensor([[0, 0],
                               [1, 0],
                               [0, 1],
                               [1, 1]], dtype=torch.double)
    # Expected: manually computed Hamming distances:
    # 0 vs 0:0, 0 vs 1:1, 0 vs 2:1, 0 vs 3:2,
    # 1 vs 1:0, 1 vs 2:2, 1 vs 3:1,
    # 2 vs 2:0, 2 vs 3:1,
    # 3 vs 3:0.
    expected = torch.tensor([[math.exp(0), math.exp(-1), math.exp(-1), math.exp(-2)],
                             [math.exp(-1), math.exp(0), math.exp(-2), math.exp(-1)],
                             [math.exp(-1), math.exp(-2), math.exp(0), math.exp(-1)],
                             [math.exp(-2), math.exp(-1), math.exp(-1), math.exp(0)]], )
    kernel_b = precompute_string_kernel(binary_rep)
    assert kernel_b.shape == (4, 4), f"Expected shape (4,4), got {kernel_b.shape}"
    assert torch.allclose(kernel_b, expected, atol=1e-6), f"Kernel matrix incorrect: {kernel_b} vs {expected}"


def test_compute_intra_mmd_expectation():
    # For N=2 samples and K=4 outcomes.
    n, d = 2, 4
    # Let each sample be uniform, so dist = 1/4 for all outcomes.
    dist = torch.full((n, d), 0.25, requires_grad=True)
    # Use the same binary_rep for 2 qubits.
    binary_rep = torch.tensor([[0, 0],
                               [1, 0],
                               [0, 1],
                               [1, 1]], dtype=torch.double)
    kernel_b = precompute_string_kernel(binary_rep)
    # For uniform distribution, expectation = (1/16)*sum(kernel_b).
    expected_val = kernel_b.sum() / 16.0
    expected = torch.full((n,), float(expected_val))
    intra_expect = compute_intra_mmd_expectation(dist, kernel_b)
    assert intra_expect.shape == (n,), f"Expected shape {(n,)}, got {intra_expect.shape}"
    assert torch.allclose(intra_expect, expected, atol=1e-6), f"Expected {expected}, got {intra_expect}"

    # Test gradients.
    loss = intra_expect.sum()
    loss.backward()
    assert dist.grad is not None, "No gradient computed for dist in intra expectation."


def test_compute_cross_mmd_expectation():
    # Let N=2 and M=3 samples, K=4 outcomes.
    b_n, b_m, d = 2, 3, 4
    # Use uniform distributions for both batches.
    dist1 = torch.full((b_n, d), 0.25, requires_grad=True)
    dist2 = torch.full((b_m, d), 0.25, requires_grad=True)
    binary_rep = torch.tensor([[0, 0],
                               [1, 0],
                               [0, 1],
                               [1, 1]], dtype=torch.double)
    kernel_b = precompute_string_kernel(binary_rep)
    # Expected cross expectation is (1/16)*sum(kernel_b), same for every pair.
    expected_val = kernel_b.sum() / 16.0
    expected = torch.full((b_n, b_m), float(expected_val))
    cross_expect = compute_cross_mmd_expectation(dist1, dist2, kernel_b)
    assert cross_expect.shape == (b_n, b_m), f"Expected shape {(b_n, b_m)}, got {cross_expect.shape}"
    assert torch.allclose(cross_expect, expected, atol=1e-6), f"Expected {expected}, got {cross_expect}"

    # Test gradients: backpropagate through dist1.
    loss = cross_expect.sum()
    loss.backward()
    assert not torch.isnan(dist1.grad).any(), "No gradient computed for dist1."
    assert not torch.all(0 == dist1.grad), f"Zero gradient for dist1: {dist1.grad}"


def test_mmd_kernel():
    # Define small arbitrary inputs.
    # For batch 1: N = 2, for batch 2: M = 3.
    expect_intra1 = torch.tensor([1.0, 2.0], requires_grad=True)
    expect_intra2 = torch.tensor([1.5, 2.5, 3.0], requires_grad=True)
    # Define a cross expectation and chamfer matrix.
    cross_exp = torch.tensor([[0.5, 0.6, 0.7],
                              [0.8, 0.9, 1.0]], requires_grad=True)
    chamfer_mat = torch.ones((2, 3), requires_grad=True)

    # Manually compute d2:
    # d2[i,j] = expect_intra1[i] + expect_intra2[j] - 2*(1 * cross_exp[i,j])
    # For i=0: [1+1.5-1.0, 1+2.5-1.2, 1+3.0-1.4] = [1.5, 2.3, 2.6]
    # For i=1: [2+1.5-1.6, 2+2.5-1.8, 2+3.0-2.0] = [1.9, 2.7, 3.0]
    d2_manual = torch.tensor([[1.5, 2.3, 2.6],
                              [1.9, 2.7, 3.0]])
    expected_kernel = torch.exp(-torch.sqrt(d2_manual))

    kernel_mat = mmd_kernel(expect_intra1, expect_intra2, cross_exp, chamfer_mat)
    assert kernel_mat.shape == (2, 3), f"Expected shape (2,3), got {kernel_mat.shape}"
    assert torch.allclose(kernel_mat, expected_kernel, atol=1e-6), f"Expected {expected_kernel}, got {kernel_mat}"

    # Test gradients: backpropagate through expect_intra1.
    loss = kernel_mat.sum()
    loss.backward()
    assert not torch.isnan(expect_intra1.grad).any(), "No gradient computed for expect_intra1."
    assert not torch.isnan(expect_intra2.grad).any(), "No gradient computed for expect_intra2."
    assert not torch.isnan(cross_exp.grad).any(), "No gradient computed for cross_exp."
    assert not torch.isnan(chamfer_mat.grad).any(), "No gradient computed for chamfer_mat."


if __name__ == '__main__':
    pytest.main([__file__])
