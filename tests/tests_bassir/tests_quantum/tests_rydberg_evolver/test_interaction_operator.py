import pytest
import torch
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.utils.qutils import get_default_register_topology


def test_interaction_operator_nqubits_1():
    """
    For a single-qubit system, the interaction operator tensor should have shape
    (1, 1, 2, 2) and equal the single-qubit number operator.
    """
    traps = get_default_register_topology(topology="all_to_all", n_qubits=1)
    evolver = RydbergEvolver(traps=traps, dim=10)
    computed_tensor = evolver._compute_interaction_operator_vectorized()
    expected_shape = (1, 1, 2, 2)

    assert computed_tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {computed_tensor.shape}"
    )
    expected_tensor = evolver.n_operator.unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
    assert torch.allclose(computed_tensor, expected_tensor, atol=1e-6), (
        f"Expected {expected_tensor}, got {computed_tensor}"
    )


def test_interaction_operator_nqubits_2_values():
    """
    For a two-qubit system, check that the computed interaction operator tensor
    has the expected numerical values:
      - At (0,0): expected = n_operator ⊗ I_2,
      - At (1,1): expected = I_2 ⊗ n_operator,
      - At (0,1) and (1,0): expected = n_operator ⊗ n_operator.
    """
    traps = get_default_register_topology(topology="all_to_all", n_qubits=2)
    evolver = RydbergEvolver(traps=traps, dim=10)
    computed_tensor = evolver._compute_interaction_operator_vectorized()
    expected_shape = (2, 2, 4, 4)
    assert computed_tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {computed_tensor.shape}"
    )
    n_op = evolver.n_operator  # (2,2)
    identity_2 = torch.eye(2, dtype=torch.cfloat, device=n_op.device)
    expected_00 = torch.kron(n_op, identity_2)  # shape (4,4)
    expected_11 = torch.kron(identity_2, n_op)
    expected_01 = torch.kron(n_op, n_op)
    # For indices (0,0)
    assert torch.allclose(computed_tensor[0, 0], expected_00, atol=1e-6), (
        f"At index (0,0): Expected {expected_00}, got {computed_tensor[0, 0]}"
    )
    # For indices (1,1)
    assert torch.allclose(computed_tensor[1, 1], expected_11, atol=1e-6), (
        f"At index (1,1): Expected {expected_11}, got {computed_tensor[1, 1]}"
    )
    # For indices (0,1) and (1,0)
    assert torch.allclose(computed_tensor[0, 1], expected_01, atol=1e-6), (
        f"At index (0,1): Expected {expected_01}, got {computed_tensor[0, 1]}"
    )
    assert torch.allclose(computed_tensor[1, 0], expected_01, atol=1e-6), (
        f"At index (1,0): Expected {expected_01}, got {computed_tensor[1, 0]}"
    )


def test_interaction_operator_nqubits_3_values():
    """
    For a three-qubit system, check that the computed interaction operator tensor
    has the expected numerical values:
      - At (0,0): expected = n_operator ⊗ I_2 ⊗ I_2,
      - At (1,1): expected = I_2 ⊗ n_operator ⊗ I_2,
      - At (2,2): expected = I_2 ⊗ I_2 ⊗ n_operator,
      - At (0,1) and (1,0): expected = n_operator ⊗ n_operator ⊗ I_2,
      - At (0,2) and (2,0): expected = n_operator ⊗ I_2 ⊗ n_operator,
      - At (1,2) and (2,1): expected = I_2 ⊗ n_operator ⊗ n_operator.
    """
    traps = get_default_register_topology(topology="all_to_all", n_qubits=3)
    evolver = RydbergEvolver(traps=traps, dim=10)
    computed_tensor = evolver._compute_interaction_operator_vectorized()
    expected_shape = (3, 3, 8, 8)
    assert computed_tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {computed_tensor.shape}"
    )

    # Test symmetry
    for i in range(3):
        for j in range(3):
            assert torch.allclose(
                computed_tensor[i, j], computed_tensor[j, i], atol=1e-6
            ), f"Tensor not symmetric for indices {i} and {j}"

    # Test content
    n_op = evolver.n_operator  # (2,2)
    identity_2 = torch.eye(2, dtype=torch.cfloat, device=n_op.device)

    # Diagonal elements:
    expected_00 = torch.kron(torch.kron(n_op, identity_2), identity_2)
    expected_11 = torch.kron(torch.kron(identity_2, n_op), identity_2)
    expected_22 = torch.kron(torch.kron(identity_2, identity_2), n_op)

    # Off-diagonal elements:
    expected_01 = torch.kron(torch.kron(n_op, n_op), identity_2)
    expected_02 = torch.kron(torch.kron(n_op, identity_2), n_op)
    expected_12 = torch.kron(torch.kron(identity_2, n_op), n_op)

    # Check diagonal indices:
    assert torch.allclose(computed_tensor[0, 0], expected_00, atol=1e-6), (
        f"At index (0,0): Expected {expected_00}, got {computed_tensor[0, 0]}"
    )
    assert torch.allclose(computed_tensor[1, 1], expected_11, atol=1e-6), (
        f"At index (1,1): Expected {expected_11}, got {computed_tensor[1, 1]}"
    )
    assert torch.allclose(computed_tensor[2, 2], expected_22, atol=1e-6), (
        f"At index (2,2): Expected {expected_22}, got {computed_tensor[2, 2]}"
    )

    # Check off-diagonal indices:
    assert torch.allclose(computed_tensor[0, 1], expected_01, atol=1e-6), (
        f"At index (0,1): Expected {expected_01}, got {computed_tensor[0, 1]}"
    )
    assert torch.allclose(computed_tensor[1, 0], expected_01, atol=1e-6), (
        f"At index (1,0): Expected {expected_01}, got {computed_tensor[1, 0]}"
    )
    assert torch.allclose(computed_tensor[0, 2], expected_02, atol=1e-6), (
        f"At index (0,2): Expected {expected_02}, got {computed_tensor[0, 2]}"
    )
    assert torch.allclose(computed_tensor[2, 0], expected_02, atol=1e-6), (
        f"At index (2,0): Expected {expected_02}, got {computed_tensor[2, 0]}"
    )
    assert torch.allclose(computed_tensor[1, 2], expected_12, atol=1e-6), (
        f"At index (1,2): Expected {expected_12}, got {computed_tensor[1, 2]}"
    )
    assert torch.allclose(computed_tensor[2, 1], expected_12, atol=1e-6), (
        f"At index (2,1): Expected {expected_12}, got {computed_tensor[2, 1]}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
