from torch import nn, Tensor
from bassir.utils.qutils import binary_activation_with_min_activation
from networkx import Graph


class Positioner(nn.Module):
    def __init__(self, traps: Graph, projector: nn.Module = None, dim: int = None):
        """
        Implements the spatial arrangements generating function x mapsto R_{theta_1}(x).

        :param dim: Input feature dimension.
        :param traps: Graph representing the available trap locations.
        :param dim: the input dimension
        """
        super().__init__()
        n_qubits = len(traps)
        self.dim = dim

        if projector:
            # Prepare the projector submodule
            self.projector = projector
        else:
            assert dim is not None, "Argument dim should not be None if no projector is given"
            self.projector = nn.Sequential(nn.Linear(dim, (n_qubits + dim) // 2),
                                           nn.ReLU(),
                                           nn.Linear((n_qubits + dim) // 2, n_qubits))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.projector(x)
        out = binary_activation_with_min_activation(logits)
        return out
