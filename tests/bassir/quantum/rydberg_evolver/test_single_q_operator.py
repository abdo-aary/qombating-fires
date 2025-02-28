import pytest
import torch
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology


# Helper module: a fixed varyer that outputs learnable fixed parameters.
class FixedVarier(torch.nn.Module):
    def __init__(self, params: torch.Tensor):
        """
        params: Tensor of shape (4,) corresponding to [raw_omega, delta, raw_phi, raw_time].
                      This parameter is learnable.
        """
        super().__init__()
        self.params = torch.nn.Parameter(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.params.unsqueeze(0).expand(b, 4)


def test_compute_single_operator_vectorized_2_qubits_correctness():
    """
    For a 2-qubit system, with fixed variational parameters that yield
    H = sigma_x (i.e. ω=2, δ=0, φ=0), the embedding tensors for each sample should be:
      E_0 = H ⊗ I  and  E_1 = I ⊗ H.
    """
    n_qubits = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    # Compute raw_omega such that softplus(raw_omega) = 2.
    raw_omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    params = torch.tensor([raw_omega, 0.0, 0.0, 1.0])
    varyer = FixedVarier(params)

    # Create a dummy input x of shape (B, dim) with B=3.
    x = torch.randn(3, 10)
    evolver = RydbergEvolver(traps=traps, dim=10, varyer=varyer)

    # Compute the embedding tensor: shape (B, n_qubits, 2^(n_qubits), 2^(n_qubits)).
    embedding_tensor = evolver._compute_single_operator_vectorized(x)
    # Expected shape is (3, 2, 4, 4)
    assert embedding_tensor.shape == (3, 2, 4, 4), f"Expected shape (3,2,4,4), got {embedding_tensor.shape}"

    # Get the single-qubit gate for each sample.
    h_batch = evolver._h_single_qubit(x)  # shape (3, 2, 2)
    identity2 = torch.eye(2, dtype=torch.cfloat, device=embedding_tensor.device)

    for i in range(3):
        expected_0 = torch.kron(h_batch[i], identity2)  # for qubit index 0
        expected_1 = torch.kron(identity2, h_batch[i])  # for qubit index 1
        assert torch.allclose(embedding_tensor[i, 0], expected_0, atol=1e-6), \
            f"Sample {i}, qubit 0: expected {expected_0}, got {embedding_tensor[i, 0]}"
        assert torch.allclose(embedding_tensor[i, 1], expected_1, atol=1e-6), \
            f"Sample {i}, qubit 1: expected {expected_1}, got {embedding_tensor[i, 1]}"

    loss = embedding_tensor.abs().sum()
    loss.backward()

    # Check that gradients flow to the fixed parameters of the varyer.
    assert varyer.params.grad is not None, "No gradient computed for varyer.params."
    assert torch.any(0 != varyer.params.grad), \
        f"varyer.params.grad is zero: {varyer.params.grad}"


if __name__ == "__main__":
    pytest.main([__file__])
