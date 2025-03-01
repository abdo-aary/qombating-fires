import pytest
import torch
from math import pi
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.utils.qutils import get_default_register_topology


# =============================================================================
# FixedVarier: A helper module that returns fixed variational parameters.
# =============================================================================
class FixedVarier(torch.nn.Module):
    """
    A fixed varyer that outputs constant variational parameters for testing.

    The parameters are a tensor of shape (4,) corresponding to
    [raw_omega, delta, raw_phi, raw_time].
    """

    def __init__(self, fixed_params: torch.Tensor):
        super().__init__()
        self.fixed_params = torch.nn.Parameter(fixed_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the fixed parameters to match the input batch size.
        b = x.shape[0]
        return self.fixed_params.unsqueeze(0).expand(b, 4)


# =============================================================================
# Test Case 1: Output shape and state normalization.
# =============================================================================
def test_forward_output_shape_and_normalization():
    """
    Verify that for a given random batch and an all-ones mask, the evolved state:
      (i) has the expected shape (batch_size, 2^(n_qubits)), and
      (ii) is approximately normalized.
    """
    n_qubits = 2
    batch_size = 3
    dim = 10
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    evolver = RydbergEvolver(traps=traps, dim=dim)
    x = torch.randn(batch_size, dim)
    mask_batch = torch.ones(batch_size, n_qubits, requires_grad=True)
    psi_out = evolver(x, mask_batch)
    expected_shape = (batch_size, 2 ** n_qubits)
    assert psi_out.shape == expected_shape, f"Expected shape {expected_shape}, got {psi_out.shape}"
    norms = torch.linalg.norm(psi_out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"State norms not close to 1: {norms}"


# =============================================================================
# Test Case 2: Gradient propagation to variational parameters and mask.
# =============================================================================
def test_parameters_gradients():
    """
    Verify that gradients propagate back to the evolver's parameters and the input mask.

    The test computes a simple loss (sum of absolute output values) and checks that
    nonzero gradients are computed for all learnable parameters as well as the mask.
    """
    n_qubits = 2
    batch_size = 2
    dim = 10
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    # Fixed parameters: effective ω=2, φ=0, δ=0, raw_time=1.0.
    raw_omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    fixed_params = torch.tensor([raw_omega, 0.0, 0.0, 1.0])
    varyer = FixedVarier(fixed_params)
    evolver = RydbergEvolver(traps=traps, dim=dim, varyer=varyer)
    x = torch.randn(batch_size, dim)
    mask_batch = torch.ones(batch_size, n_qubits, requires_grad=True)
    psi_out = evolver(x, mask_batch)
    loss = psi_out.abs().sum()
    loss.backward()
    n_zeros = 0
    for param in evolver.parameters():
        assert param.grad is not None, f"Some of the evolver's gradients are None."
        if torch.any(0 != param.grad):
            n_zeros += 1
    assert n_zeros != 0, f"Evolver's gradients are always zero."

    assert mask_batch.grad is not None, "No gradient computed for mask_batch."
    assert torch.any(0 != mask_batch.grad), "Gradient for mask_batch is all zeros."


# =============================================================================
# Test Case 3: Single-qubit evolution.
# =============================================================================
def test_forward_single_qubit():
    """
    For a single-qubit system, with fixed parameters:
      - ω = 2, φ = 0, δ = 0, and effective time = π/2,
    the evolution should yield:
      exp(-i*(π/2)*σₓ)|0> = [0, -i]ᵀ.
    """
    n_qubits = 1
    dim = 10
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    raw_omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    # Set raw_time so that softplus(raw_time) ≈ π/2.
    raw_time = torch.log(torch.exp(torch.tensor(pi / 2)) - 1)
    fixed_params = torch.tensor([raw_omega, 0.0, 0.0, raw_time], dtype=torch.float32)
    varyer = FixedVarier(fixed_params)
    evolver = RydbergEvolver(traps=traps, dim=dim, varyer=varyer)
    x = torch.randn(1, dim)
    mask_batch = torch.ones(1, n_qubits)
    psi = evolver(x, mask_batch)
    expected = torch.tensor([[0.0, -1j]], dtype=torch.cdouble, device=psi.device)
    assert torch.allclose(psi, expected, atol=1e-6), f"Expected {expected}, got {psi}"


if __name__ == '__main__':
    pytest.main([__file__])
