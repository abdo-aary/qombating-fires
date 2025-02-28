# %%
from torch import nn, Tensor
from bassir.utils.metrics import binary_activation
from networkx import Graph


class Positioner(nn.Module):
    def __init__(self, dim: int, traps: Graph, projector: nn.Module = None, tau: float = 1.0):
        """

        :param dim: Input feature dimension.
        :param traps: Graph representing the available trap locations.
        :param tau: Temperature for the binary activation.
        """
        super().__init__()
        n_qubits = len(traps)
        self.dim = dim
        self.tau = tau

        # Prepare the projector submodule
        self.projector = nn.Sequential(nn.Linear(dim, (n_qubits + dim) // 2),
                                       nn.ReLU(),
                                       nn.Linear((n_qubits + dim) // 2, n_qubits)) if not projector else projector

    def forward(self, x: Tensor) -> Tensor:
        logits = self.projector(x)
        out = binary_activation(logits, tau=self.tau)
        return out
