import pytest
import torch
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology


def test_single_operator_vectorized_2_qubits():
    n_qubits = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)

    # Expected operators:
    # For i=0: expected = h_single ⊗ I_2.
    # For i=1: expected = I_2 ⊗ h_single.
    identity_2 = torch.eye(n_qubits, dtype=torch.cfloat)
    expected_0 = torch.kron(evolver._h_single_qubit(), identity_2)
    expected_1 = torch.kron(identity_2, evolver._h_single_qubit())

    embedding_tensor = evolver._compute_single_operator_vectorized()

    assert embedding_tensor.shape == (n_qubits, 2 ** n_qubits, 2 ** n_qubits), (
        f"Expected shape {(n_qubits, 2 ** n_qubits, 2 ** n_qubits)}, got {embedding_tensor.shape}"
    )
    assert torch.allclose(embedding_tensor[0], expected_0, atol=1e-6), (
        f"For qubit index 0, expected:\n{expected_0}\ngot:\n{embedding_tensor[0]}"
    )
    assert torch.allclose(embedding_tensor[1], expected_1, atol=1e-6), (
        f"For qubit index 1, expected:\n{expected_1}\ngot:\n{embedding_tensor[1]}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
