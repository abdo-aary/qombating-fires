import torch
from torch import randn
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.positioner import Positioner
from bassir.models.quantum.bassir_kernel import BassirKernel
from bassir.models.quantum.qutils import get_default_register_topology


def test_bassir_kernel_gradients():
    """
    Test that the combined BassirKernel (using the Positioner and RydbergEvolver)
    is differentiable with respect to its inputs x1.

    We check that when we compute the loss from the kernel matrix, gradients propagate back to x1.
    Since x1 may become non-leaf during forward computations, we call x1.retain_grad() explicitly.
    """
    batch_size_1, batch_size_2, dim = 3, 4, 12
    n_qubits = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random inputs with requires_grad=True
    x1 = randn((batch_size_1, dim), dtype=torch.float32, requires_grad=True).to(device)
    x1.retain_grad()  # Ensure gradients are retained on x1.
    x2 = randn((batch_size_2, dim), dtype=torch.float32).to(device)

    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

    # Instantiate Positioner, RydbergEvolver, and the BassirKernel.
    positioner = Positioner(dim, traps).to(torch.float32).to(device)
    evolver = RydbergEvolver(traps=traps).to(device)
    qkernel = BassirKernel(traps, positioner, evolver).to(device)

    # Compute the kernel matrix.
    chamfer_k_mat = qkernel(x1, x2).to_dense()

    # For our simple test, define a scalar loss.
    loss = chamfer_k_mat.sum()
    loss.backward()

    # Check that gradients are propagated to x1.
    assert x1.grad is not None, "No gradient computed for x1."
    assert torch.any(0 != x1.grad), f"Gradients are zero: {x1.grad}"

    # Also check that the kernel values are in a plausible range, e.g. between 0 and 1.
    assert torch.all((chamfer_k_mat >= 0) & (chamfer_k_mat <= 1)), f"Kernel values out of range: {chamfer_k_mat}"

    print("BassirKernel gradient test passed.")


if __name__ == '__main__':
    test_bassir_kernel_gradients()
