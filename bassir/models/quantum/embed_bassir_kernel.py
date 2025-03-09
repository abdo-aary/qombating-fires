from abc import ABC, abstractmethod
from typing import Union
from linear_operator import LinearOperator
from gpytorch.kernels import Kernel
from torch import Tensor, nn
import torch

from bassir.models.quantum.positioner import Positioner
from bassir.utils.qutils import (get_distances_traps, chamfer_kernel, compute_intra_mmd_expectation,
                                 precompute_string_kernel, compute_cross_mmd_expectation, mmd_kernel)
from bassir.models.quantum.rydberg import RydbergEvolver
from networkx import Graph


class Embedder(ABC, nn.Module):
    @property
    @abstractmethod
    def dim_in(self) -> int:
        """Input dimension property that must be implemented."""
        pass

    @property
    @abstractmethod
    def dim_out(self) -> int:
        """Output (or embedding) dimension property that must be implemented."""
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward method must be implemented in subclasses."""
        pass


class SimpleEmbedder(Embedder):
    def __init__(self, dim_in: int, dim_out: int, embedding_layer: nn.Module = None):
        super().__init__()
        self._dim_in = dim_in
        self._dim_out = dim_out
        if embedding_layer:
            self.embedding_layer = embedding_layer
        else:
            self.embedding_layer = nn.Sequential(
                nn.Linear(dim_in, (dim_in + dim_out) // 2),
                nn.ReLU(),
                nn.Linear((dim_in + dim_out) // 2, dim_out)
            )

    @property
    def dim_in(self) -> int:
        return self._dim_in

    @property
    def dim_out(self) -> int:
        return self._dim_out

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding_layer(x)


class EmbedBassirKernel(Kernel):
    # the kernel is not necessarily stationary
    is_stationary = False

    def __init__(self, traps: Graph, embedder: Embedder, positioner: nn.Module = None,
                 evolver: nn.Module = None, **kwargs):
        """
        Initializes the BassirKernel class.

        :param traps: the used topology
        :param embedder: submodule that embeds temporal features into a latent space
        :param kwargs:
        """
        super().__init__(**kwargs)
        n_qubits = len(traps)
        self.n_qubits = n_qubits
        dim_embed_out = embedder.dim_out

        self.traps = traps
        self.embedder = embedder

        if not positioner:
            # The projector produces spatial locations out of the embedder's outputs. A simple neural net.
            projector = nn.Sequential(nn.Linear(dim_embed_out, (n_qubits + dim_embed_out) // 2),
                                      nn.ReLU(),
                                      nn.Linear((n_qubits + dim_embed_out) // 2, n_qubits))

            self.positioner = Positioner(traps=traps, projector=projector)
        else:
            self.positioner = positioner

        if not evolver:
            # The varyer is a linear layer that gets the embedder's outputs and produce the pulses parameters.
            # A simple neural net.
            n_pulse_params = 4
            varyer = nn.Sequential(nn.Linear(dim_embed_out, (n_pulse_params + dim_embed_out) // 2),
                                   nn.ReLU(),
                                   nn.Linear((n_pulse_params + dim_embed_out) // 2, n_pulse_params))

            self.evolver = RydbergEvolver(traps=traps, varyer=varyer)
        else:
            self.evolver = evolver

        self.distances = get_distances_traps(traps, self.evolver.reference_distance_d_0)

        # Retrieve the precomputed binary representation tensor from the evolver.
        binary_representation = self.evolver.binary_representation  # shape (2^(n_qubits), n_qubits)
        self.string_kernel_mat = precompute_string_kernel(binary_representation)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Union[Tensor, LinearOperator]:
        """
        Computes the end kernel function between two batches of inputs x1 and x2.

        For each input, the Positioner produces a mask of active qubits and the RydbergEvolver
        produces the corresponding global state from which we obtain a probability distribution.
        We marginalize these distributions (via compute_marginal_probs) to obtain local distributions.
        Then, using a precomputed string kernel Gram matrix (self.string_kernel_mat) computed over the global
        outcome space, we calculate intra-sample and cross-sample expectations. Combined with the spatial
        similarity given by the Chamfer kernel, we form the squared MMD distance:

          d_M^2 = E_intra(x1) + E_intra(x2) - 2 * (chamfer_mat * E_cross(x1,x2),

        and return the final kernel:

          k(x1,x2) = exp( - sqrt( d_M^2 ) ).

        Args:
            x1: Tensor of shape (..., N, dim)
            x2: Tensor of shape (..., M, dim)
            diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)

        Returns:
            Kernel matrix of shape (..., N, M) or self-evaluations diagonal of shape (..., N)
        """
        string_kernel_mat = self.string_kernel_mat.to(x1.device)
        if x1.size() == x2.size() and torch.equal(x1, x2):
            if diag:
                out = torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                # Embed the data input
                e = self.embedder(x1)

                # Compute the mask out of the embedding vector
                mask = self.positioner(e)  # shape (..., N, n_qubits)

                # Compute spatial similarity (Chamfer kernel) between trap positions.
                chamfer_gram_mat = chamfer_kernel(mask, mask, self.distances.to(mask.device))  # shape (..., N, M)

                # Evolve the quantum state to obtain global probability distributions.
                psi_out = self.evolver(e, mask)  # shape (..., N, 2^(n_qubits))
                global_probs1 = psi_out.abs() ** 2  # shape (..., N, 2^(n_qubits))

                # Compute the intra-sample string kernel expectation.
                expect_intra1 = compute_intra_mmd_expectation(global_probs1, string_kernel_mat)  # shape (N,)

                # Compute cross expectation between the two batches.
                cross_exp = compute_cross_mmd_expectation(global_probs1, global_probs1,
                                                          string_kernel_mat)  # shape (N, M)

                # Compute the squared MMD distance:
                out = mmd_kernel(expect_intra1, expect_intra1, cross_exp, chamfer_gram_mat)
        else:
            # Embed the data inputs
            e1, e2 = self.embedder(x1), self.embedder(x2)

            # Compute the masks out of the embedding vectors
            mask1 = self.positioner(e1)  # shape (..., N, n_qubits)
            mask2 = self.positioner(e2)  # shape (..., M, n_qubits)

            # Compute spatial similarity (Chamfer kernel) between trap positions.
            chamfer_gram_mat = chamfer_kernel(mask1, mask2, self.distances.to(mask1.device))  # shape (..., N, M)

            # Evolve the quantum state to obtain global probability distributions.
            psi_out1 = self.evolver(e1, mask1)  # shape (..., N, 2^(n_qubits))
            psi_out2 = self.evolver(e2, mask2)  # shape (..., M, 2^(n_qubits))

            global_probs1 = psi_out1.abs() ** 2  # shape (..., N, 2^(n_qubits))
            global_probs2 = psi_out2.abs() ** 2  # shape (..., M, 2^(n_qubits))

            # For each sample, compute the intra-sample string kernel expectation.
            # Here, self.string_kernel_mat is the precomputed Gram matrix K_B of shape (2^(n_qubits), 2^(n_qubits)).
            # We assume that the batch dimensions are the same (or can be merged) for simplicity.
            # For example, if x1 and x2 are 2D (i.e. shape (N, dim) and (M, dim)), then:
            expect_intra1 = compute_intra_mmd_expectation(global_probs1, string_kernel_mat)  # shape (N,)
            expect_intra2 = compute_intra_mmd_expectation(global_probs2, string_kernel_mat)  # shape (M,)

            # Compute cross expectation between the two batches.
            cross_exp = compute_cross_mmd_expectation(global_probs1, global_probs2, string_kernel_mat)  # shape (N, M)

            # Compute the squared MMD distance:
            out = mmd_kernel(expect_intra1, expect_intra2, cross_exp, chamfer_gram_mat)

        return out
