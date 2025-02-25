from math import pi
import pytest
import torch

from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology


def test_h_single_qubit_sigma_x():
    """
    Test case: when phi = 0, delta = 0, and omega = 2,
    the single-qubit Hamiltonian should equal sigma_x.
    """
    n_qubits = 1
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)
    with torch.no_grad():
        evolver.omega = torch.tensor([2.0], dtype=torch.float32)
        evolver.phi = torch.tensor([0.0], dtype=torch.float32)
        evolver.delta = torch.tensor([0.0], dtype=torch.float32)
    h_single = evolver._h_single_qubit()  # Shape: (2, 2)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=h_single.dtype, device=h_single.device)
    assert torch.allclose(h_single, sigma_x, atol=1e-6), f"Expected sigma_x, got {h_single}"


def test_h_single_qubit_negative_sigma_y():
    """
    Test case: when phi = pi/2, delta = 0, and omega = 2,
    the single-qubit Hamiltonian should equal -sigma_y.
    """
    n_qubits = 1
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)
    with torch.no_grad():
        evolver.omega = torch.tensor([2.0], dtype=torch.float32)
        evolver.phi = torch.tensor([pi / 2], dtype=torch.float32)
        evolver.delta = torch.tensor([0.0], dtype=torch.float32)
    h_single = evolver._h_single_qubit()
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=h_single.dtype, device=h_single.device)
    expected_h = -sigma_y
    assert torch.allclose(h_single, expected_h, atol=1e-6), f"Expected -sigma_y, got {h_single}"


def test_h_single_qubit_detuning_only():
    """
    Test case: when omega = 0, phi = 0, and delta = 1,
    the single-qubit Hamiltonian should equal -n, where
    n = (I + sigma_z) / 2.
    """
    n_qubits = 1
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)
    with torch.no_grad():
        evolver.omega = torch.tensor([0.0], dtype=torch.float32)
        evolver.phi = torch.tensor([0.0], dtype=torch.float32)
        evolver.delta = torch.tensor([1.0], dtype=torch.float32)
    h_single = evolver._h_single_qubit()
    identity_2 = torch.eye(2, dtype=torch.cfloat, device=h_single.device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=h_single.dtype, device=h_single.device)

    number_operator = (identity_2 + sigma_z) / 2
    expected_h = -number_operator
    assert torch.allclose(h_single, expected_h, atol=1e-6), f"Expected -n_operator, got {h_single}"


if __name__ == "__main__":
    pytest.main([__file__])
