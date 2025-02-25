import torch
import pytest
from torch import randn, tensor, kron
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.qutils import get_default_register_topology
from bassir.models.quantum.positioner import Positioner  # assuming your Positioner module is defined here


def test_gradients_through_positioner():
    """
    Test that gradients computed from the output of RydbergEvolver propagate
    back to the Positioner parameters. We define a simple loss on the output state
    and check that the gradients for the Positioner's parameters are non-zero.
    """
    batch_size, dim = 4, 12
    n_qubits = 2
    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

    # Instantiate Positioner and RydbergEvolver.
    positioner = Positioner(dim, traps).to(torch.float32)
    evolver = RydbergEvolver(traps=traps)

    # Ensure both modules are on the same device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positioner = positioner.to(device)
    evolver = evolver.to(device)

    # Forward pass: compute mask and then psi.
    x = randn((batch_size, dim), dtype=torch.float32).to(device)
    mask_batch = positioner(x)  # shape: (batch_size, n_qubits)
    mask_batch = mask_batch.float()
    psi_out = evolver(mask_batch)

    # Define a simple loss function: sum of the L2 norms of psi_out.
    loss = psi_out.norm()
    loss.backward()

    # Check that gradients are computed for all Positioner parameters.
    for name, param in positioner.named_parameters():
        assert param.grad is not None, f"No gradient computed for parameter '{name}' in Positioner"
        assert torch.any(param.grad != 0), f"Zero gradient for parameter '{name}' in Positioner"


if __name__ == '__main__':
    pytest.main([__file__])