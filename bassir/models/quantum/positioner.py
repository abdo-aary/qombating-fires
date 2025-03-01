# %%
from torch import nn, Tensor
from bassir.utils.metrics import binary_activation_with_min_activation  #binary_activation
from networkx import Graph


class Positioner(nn.Module):
    def __init__(self, dim: int, traps: Graph, projector: nn.Module = None):
        """
        Implements the spatial arrangements generating function x \mapsto R_{\theta_1}(x).

        :param dim: Input feature dimension.
        :param traps: Graph representing the available trap locations.
        :param tau: Temperature for the binary activation.
        """
        super().__init__()
        n_qubits = len(traps)
        self.dim = dim

        # Prepare the projector submodule
        self.projector = nn.Sequential(nn.Linear(dim, (n_qubits + dim) // 2),
                                       nn.ReLU(),
                                       nn.Linear((n_qubits + dim) // 2, n_qubits)) if not projector else projector

    def forward(self, x: Tensor) -> Tensor:
        logits = self.projector(x)
        out = binary_activation_with_min_activation(logits)
        return out
