import pytest
import torch
from math import pi
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology


# A helper module that always returns the same fixed variational parameters (but learnable).
class FixedVaryer(torch.nn.Module):
    def __init__(self, params: torch.Tensor):
        """
        params: Tensor of shape (4,) corresponding to [raw_omega, delta, raw_phi, raw_time].
                      This parameter is learnable.
        """
        super().__init__()
        self.params = torch.nn.Parameter(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.params.unsqueeze(0).expand(batch_size, 4)


# Test case 1: When phi = 0, delta = 0, and omega = 2, we expect H = σₓ.
def test_h_single_qubit_sigma_x_grad_params():
    # Compute raw_omega such that softplus(raw_omega)=2.
    raw_omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    params = torch.tensor([raw_omega, 0.0, 0.0, 1.0], requires_grad=True)
    varyer = FixedVaryer(params)

    # Create dummy input x.
    x = torch.randn(5, 10)
    traps = get_default_register_topology(topology="all_to_all", n_qubits=1)
    evolver = RydbergEvolver(traps=traps, dim=10, varyer=varyer)

    h_batch = evolver._h_single_qubit(x)  # shape (5, 2, 2)
    sigma_x = torch.tensor([[0, 1], [1, 0]], device=h_batch.device, dtype=torch.cdouble)

    for i in range(h_batch.shape[0]):
        assert torch.allclose(h_batch[i], sigma_x, atol=1e-6), f"Sample {i}: Expected σₓ, got {h_batch[i]}"

    loss = h_batch.abs().sum()
    loss.backward()
    # Check that gradients flow back to the FixedVaryer parameters.
    assert varyer.params.grad is not None, "No gradient computed for varyer.params."
    assert torch.any(
        0 != varyer.params.grad), f"Zero gradient for varyer.params: {varyer.params.grad}"


# Test case 2: When phi = π/2, delta = 0, and omega = 2, we expect H = -σ_y.
def test_h_single_qubit_negative_sigma_y_grad_params():
    omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    params = torch.tensor([omega, 0.0, pi / 2, 1.0], requires_grad=True)
    varyer = FixedVaryer(params)

    x = torch.randn(4, 10)
    traps = get_default_register_topology(topology="all_to_all", n_qubits=1)
    evolver = RydbergEvolver(traps=traps, dim=10, varyer=varyer)

    h_batch = evolver._h_single_qubit(x)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble, device=h_batch.device)
    expected = -sigma_y
    for i in range(h_batch.shape[0]):
        assert torch.allclose(h_batch[i], expected, atol=1e-6), f"Sample {i}: Expected -σ_y, got {h_batch[i]}"


# Test case 3: When raw_omega is very negative (effectively 0), phi = 0, and delta = 1, we expect H = -n_operator.
def test_h_single_qubit_detuning_only_grad_params():
    # Set raw_omega to -100 so that softplus(-100) is ~0.
    params = torch.tensor([-100.0, 1.0, 0.0, 1.0], requires_grad=True)
    varyer = FixedVaryer(params)

    x = torch.randn(5, 10, requires_grad=True)
    traps = get_default_register_topology(topology="all_to_all", n_qubits=1)
    evolver = RydbergEvolver(traps=traps, dim=10, varyer=varyer)

    h_batch = evolver._h_single_qubit(x)
    identity2 = torch.eye(2, dtype=torch.cdouble, device=h_batch.device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble, device=h_batch.device)
    number_op = (identity2 + sigma_z) / 2  # n_operator = [[1,0],[0,0]]
    expected = -number_op  # Expected: [[-1,0],[0,0]]
    for i in range(h_batch.shape[0]):
        assert torch.allclose(h_batch[i], expected, atol=1e-6), f"Sample {i}: Expected -n_operator, got {h_batch[i]}"


if __name__ == "__main__":
    pytest.main([__file__])
