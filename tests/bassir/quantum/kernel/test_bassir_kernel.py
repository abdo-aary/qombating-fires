#%%
import torch
from torch import randn
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.models.quantum.positioner import Positioner
from bassir.models.quantum.bassir_kernel import BassirKernel
from bassir.models.quantum.qutils import get_default_register_topology


dim = 12
n_qubits = 2
traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)

# Instantiate Positioner and RydbergEvolver.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

positioner = Positioner(dim, traps).to(torch.float32).to(device)
evolver = RydbergEvolver(traps=traps).to(device)
qkernel = BassirKernel(traps, positioner, evolver).to(device)

# Forward pass: compute mask and then psi.
batch_size_1, batch_size_2 = 3, 4
x1 = randn((batch_size_1, dim), dtype=torch.float32, requires_grad=True).to(device)
x2 = randn((batch_size_2, dim), dtype=torch.float32).to(device)

chamfer_gram_mat = qkernel(x1, x2).to_dense()
print(f"chamfer_gram_mat = {chamfer_gram_mat}")
loss = chamfer_gram_mat.sum()
loss.backward()

assert x1.grad is not None, "No gradients computed for x1"
assert torch.any(0 != x1.grad), f"Gradients are zero: {x1.grad}"
