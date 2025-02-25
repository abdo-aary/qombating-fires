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
        self.traps = traps
        self.dim = dim
        self.tau = tau
        self.projector = nn.Linear(dim, len(traps)) if not projector else projector

    def forward(self, x: Tensor) -> Tensor:
        logits = self.projector(x)
        out = binary_activation(logits, tau=self.tau)
        return out
