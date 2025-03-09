import pytest
import torch
from math import pi

from bassir.models.quantum.embed_bassir_kernel import EmbedBassirKernel, SimpleEmbedder
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.utils.qutils import get_default_register_topology
from torch import Tensor


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


# A dummy positioner that always outputs a mask of ones.
class DummyPositioner(torch.nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        # Assume number of qubits is 1 (for the simple test) or can be adjusted.
        n_qubits = 1
        return torch.ones(batch_size, n_qubits, device=x.device)


def test_bassir_kernel_simple_case():
    """
    Test that the end kernel function computes the expected value.
    For a single-qubit system with fixed variational parameters that yield:
      ω = 2, φ = 0, δ = 0, and t = π/2,
    the evolved state is [0, -i]^T so that the measurement distribution is [0, 1].
    Hence, the intra- and cross-expectations are 1 and, with an all-ones mask,
    the spatial (Chamfer) kernel is 1. This leads to a zero MMD distance and an
    end kernel value of exp(0)=1.
    """
    n_qubits = 1
    batch_size = 2
    dim = 10
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

    # Fixed varyer: compute raw_omega and raw_time so that softplus(raw_omega)=2 and softplus(raw_time)=π/2.
    raw_omega = torch.log(torch.exp(torch.tensor(2.0)) - 1)
    raw_time = torch.log(torch.exp(torch.tensor(pi / 2)) - 1)
    fixed_params = torch.tensor([raw_omega, 0.0, 0.0, raw_time])
    varyer = FixedVaryer(fixed_params)

    # Instantiate evolver with the fixed varyer.
    evolver = RydbergEvolver(traps=traps, dim=dim, varyer=varyer)
    # Use the dummy positioner so that masks are all ones.
    positioner = DummyPositioner()
    identity_layer = torch.nn.Identity()

    embedder = SimpleEmbedder(dim_in=dim, dim_out=dim, embedding_layer=identity_layer)

    kernel = EmbedBassirKernel(traps=traps, embedder=embedder, positioner=positioner, evolver=evolver)

    # Create dummy inputs (the varyer output is fixed, so x-values are irrelevant).
    x1 = torch.randn(batch_size, dim)
    x2 = torch.randn(batch_size, dim)

    kernel_mat = kernel(x1, x2).to_dense()  # Expected shape: (batch_size, batch_size)
    expected = torch.ones(batch_size, batch_size, dtype=kernel_mat.dtype, device=kernel_mat.device)
    assert torch.allclose(kernel_mat, expected, atol=1e-6), f"Expected kernel {expected}, got {kernel_mat}"


def test_kernel_symmetry_psdness():
    with torch.no_grad():
        n_qubits = 2
        batch_size = 4
        dim = 10
        traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedder = SimpleEmbedder(dim_in=dim, dim_out=dim*2)
        kernel = EmbedBassirKernel(traps=traps, embedder=embedder).to(device)

        # Check PSDness for multiple samples
        for i in range(10):
            x = torch.randn((batch_size, dim)).to(device)
            kernel_mat = kernel(x).to_dense()

            # Check symmetry.
            assert torch.allclose(kernel_mat, kernel_mat.t(), atol=1e-6), "Kernel matrix is not symmetric."

            # Check that diagonal entries are 1.
            for j in range(batch_size):
                diag_val = kernel_mat[j, j]
                expected_diag = torch.tensor(1.0, dtype=kernel_mat.dtype, device=kernel_mat.device)
                assert torch.allclose(diag_val, expected_diag, atol=1e-6), \
                    f"Diagonal element {j} is not 1: {diag_val}"

            # Check positive semidefiniteness.
            # Since the kernel matrix is symmetric and real, use torch.linalg.eigvalsh.
            eigenvals = torch.linalg.eigvalsh(kernel_mat)
            tol = 1e-6
            assert torch.all(eigenvals >= -tol), f"Kernel matrix is not PSD. Eigenvalues: {eigenvals}"


def test_bassir_kernel_gradients():
    """
    Test that gradients propagate from the kernel function back to the inputs and the varyer parameters.
    We compute the kernel between two batches, define a scalar loss (the sum of the kernel matrix),
    and then check that the gradient of x1 and the fixed varyer's parameters are nonzero.
    """
    n_qubits = 2
    dim = 10
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SimpleEmbedder(dim_in=dim, dim_out=dim * 2)
    kernel = EmbedBassirKernel(traps=traps, embedder=embedder).to(device)

    # Forward pass: compute mask and then psi.
    batch_size = 16
    x = torch.randn((batch_size, dim)).to(device)

    # kernel_mat = kernel(x1, x2).to_dense()
    kernel_mat = kernel(x).to_dense()
    loss = kernel_mat.sum()
    loss.backward()

    # Check that all kernel parameters (positioner's and evolver's) gradients are computed.
    n_params, n_zeros = 0, 0
    for param in kernel.parameters():
        n_params += 1
        assert not torch.isnan(param.grad).any(), "Some of the kernel's gradients are None."
        if torch.all(0 == param.grad):
            n_zeros += 1
    assert n_zeros != n_params, "Kernel's gradients are always zero."


if __name__ == '__main__':
    pytest.main([__file__])
