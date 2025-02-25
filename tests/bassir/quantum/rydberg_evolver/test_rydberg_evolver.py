import pytest
import torch
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology
from math import pi


# Test 1: Check that the forward output has the expected shape and is (approximately) normalized.
def test_forward_output_shape_and_normalization():
    n_qubits = 2
    batch_size = 3
    # Use a default topology for n_qubits
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)

    # Create a mask_batch; here we use all ones (i.e. all qubits are active)
    mask_batch = torch.ones((batch_size, n_qubits), dtype=torch.float32, device=evolver.n_operator.device)

    psi_out = evolver(mask_batch)  # Expect shape (batch_size, 2**n_qubits)
    expected_shape = (batch_size, 2 ** n_qubits)
    assert psi_out.shape == expected_shape, f"Expected shape {expected_shape}, got {psi_out.shape}"

    # Optionally, check approximate normalization (||psi|| ~ 1)
    norms = torch.linalg.norm(psi_out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"State norms not close to 1: {norms}"


# Test 2: Check that the quantum parameters are trainable (i.e. gradients are nonzero).
def test_parameters_trainable():
    n_qubits = 2
    batch_size = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)

    # Use a mask_batch of ones.
    mask_batch = torch.ones((batch_size, n_qubits), dtype=torch.float32, device=evolver.n_operator.device)

    # Zero out previous gradients.
    for param in evolver.raw_params.values():
        if param.grad is not None:
            param.grad.zero_()

    psi_out = evolver(mask_batch)
    # Define a simple loss function: for instance, the sum of squared absolute values of the output.
    loss = (psi_out.abs() ** 2).sum()
    loss.backward()

    # Check that gradients exist and are nonzero.
    for name, param in evolver.raw_params.items():
        assert param.grad is not None, f"Gradient for parameter {name} is None."
        assert torch.any(param.grad != 0), f"Gradient for parameter {name} is all zeros."


# Test 3: Check that the forward function is differentiable with respect to the input mask.
def test_forward_gradients_flow_from_mask():
    n_qubits = 2
    batch_size = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)

    # Create a mask_batch as a differentiable tensor (float) with requires_grad=True.
    mask_batch = torch.ones((batch_size, n_qubits), dtype=torch.float32, device=evolver.n_operator.device,
                            requires_grad=True)

    psi_out = evolver(mask_batch)
    loss = psi_out.real.sum()  # Use real part if needed
    loss.backward()

    # Check that mask_batch.grad is not None and nonzero.
    assert mask_batch.grad is not None, "No gradient computed for mask_batch."
    assert torch.any(0 != mask_batch.grad), "Gradient for mask_batch is all zeros."


def test_forward_single_qubit():
    """
    For a single-qubit system, test that when effective parameters are:
      omega = 2 rad/μs, φ = 0, δ = 0, and time = π/2 μs,
    the forward function returns the evolved state:
      exp(-i * (π/2) * σ_x)|0> = [0, -i]^T.
    """
    n_qubits = 1
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps)

    with torch.no_grad():
        # Set effective parameters via the setters.
        evolver.omega = torch.tensor([2.0], dtype=torch.float32)
        evolver.phi = torch.tensor([0.0], dtype=torch.float32)
        evolver.delta = torch.tensor([0.0], dtype=torch.float32)
        evolver.time = torch.tensor([pi / 2], dtype=torch.float32)

    # For a single qubit, the mask has shape (batch_size, n_qubits). Use batch size 1 with mask=1.
    mask_batch = torch.ones(1, n_qubits, dtype=torch.float32)
    psi = evolver(mask_batch)  # Expected shape: (1, 2)

    # Expected evolved state: exp(-i*(π/2)*σ_x)|0> = -i * σ_x|0> = [0, -i]^T.
    expected = torch.tensor([[0.0, -1j]], dtype=torch.cfloat, device=psi.device)

    assert torch.allclose(psi, expected, atol=1e-6), f"Expected {expected}, got {psi}"


def test_forward_cuda_nontrivial_mask():
    """
    Test the RydbergEvolver forward function on a CUDA device using a two-qubit system
    with a non-trivial mask. For mask [1, 0], only the first qubit is active, so the effective
    Hamiltonian reduces to H_eff = E_0 = H(pi) ⊗ I_2. With effective parameters set such that
    H(pi) = sigma_x, and with time = π/2, the evolution operator becomes
    exp(-i*(π/2)* (sigma_x ⊗ I_2)) = exp(-i*(π/2)*sigma_x) ⊗ I_2.
    Acting on the initial state |00> (i.e. [1,0,0,0]^T), this should yield
    [0, 0, -i, 0]^T.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping CUDA test.")

    device = torch.device("cuda")
    n_qubits = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps).to(device)

    with torch.no_grad():
        # Set effective parameters via the setters.
        # We want effective omega = 2, effective phi = 0, delta = 0, and effective time = π/2.
        # Recall: raw_omega = ln(exp(omega) - 1) and raw_time = ln(exp(time) - 1).
        # For omega=2: raw_omega ~ ln(e^2 - 1) ≈ ln(7.389 - 1)= ln(6.389) ≈ 1.857.
        # For time=π/2: raw_time ~ ln(e^(π/2) - 1) ≈ ln(4.810 - 1)= ln(3.810) ≈ 1.337.
        evolver.omega = torch.tensor([2.0], dtype=torch.float32, device=device)
        evolver.phi = torch.tensor([0.0], dtype=torch.float32, device=device)
        evolver.delta = torch.tensor([0.0], dtype=torch.float32, device=device)
        evolver.time = torch.tensor([pi / 2], dtype=torch.float32, device=device)

    # For a two-qubit system, choose a non-trivial mask: [1, 0] means only the first qubit is active.
    mask_batch = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)  # shape: (1, 2)

    psi = evolver(mask_batch)  # Expected shape: (1, 2**2) = (1, 4)

    # Expected evolution:
    # With effective omega=2 and phi=0, h_single = sigma_x.
    # For the first qubit, E_0 = sigma_x ⊗ I_2.
    # Then, exp(-i*(π/2)*(sigma_x ⊗ I_2)) = exp(-i*(π/2)*sigma_x) ⊗ I_2.
    # And exp(-i*(π/2)*sigma_x)|0> = [0, -i]^T.
    # So the expected state is [0, -i]^T ⊗ |0> = [0, 0, -i, 0]^T.
    expected = torch.tensor([[0.0, 0.0, -1j, 0.0]], dtype=torch.cfloat, device=device)

    # Check that output is on CUDA.
    assert psi.device.type == "cuda", "Output state is not on a CUDA device."
    # Check that the evolved state matches the expected state.
    assert torch.allclose(psi, expected, atol=1e-6), f"Expected {expected}, got {psi}"


if __name__ == '__main__':
    pytest.main([__file__])
