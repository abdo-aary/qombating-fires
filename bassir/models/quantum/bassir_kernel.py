from typing import Union
from linear_operator import LinearOperator
from gpytorch.kernels import Kernel
from torch import Tensor, nn

from bassir.utils.qutils import (get_distances_traps, chamfer_kernel, compute_intra_mmd_expectation,
                                 precompute_string_kernel, compute_cross_mmd_expectation, mmd_kernel)
from bassir.models.quantum.rydberg import RydbergEvolver
from networkx import Graph


class BassirKernel(Kernel):
    # the kernel is not necessarily stationary
    is_stationary = False

    def __init__(self, traps: Graph, positioner: nn.Module, evolver: RydbergEvolver, **kwargs):
        """
        Initializes the BassirKernel class.

        :param traps: the used topology
        :param positioner: the used positioner
        :param evolver: the used evolver
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.traps = traps
        self.positioner = positioner
        self.evolver = evolver
        self.distances = get_distances_traps(traps, evolver.reference_distance_d_0)

        # Retrieve the precomputed binary representation tensor from the evolver.
        binary_representation = self.evolver.binary_representation  # shape (2^(n_qubits), n_qubits)
        self.string_kernel_mat = precompute_string_kernel(binary_representation)

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Union[Tensor, LinearOperator]:
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
            **params: additional parameters (if any)

        Returns:
            Kernel matrix of shape (..., N, M)
        """
        mask1 = self.positioner(x1)  # shape (..., N, n_qubits)
        mask2 = self.positioner(x2)  # shape (..., M, n_qubits)

        # Compute spatial similarity (Chamfer kernel) between trap positions.
        chamfer_gram_mat = chamfer_kernel(mask1, mask2, self.distances.to(mask1.device))  # shape (..., N, M)

        # Evolve the quantum state to obtain global probability distributions.
        psi_out1 = self.evolver(x1, mask1)  # shape (..., N, 2^(n_qubits))
        psi_out2 = self.evolver(x2, mask2)  # shape (..., M, 2^(n_qubits))

        global_probs1 = psi_out1.abs() ** 2  # shape (..., N, 2^(n_qubits))
        global_probs2 = psi_out2.abs() ** 2  # shape (..., M, 2^(n_qubits))

        string_kernel_mat = self.string_kernel_mat.to(mask1.device)
        # For each sample, compute the intra-sample string kernel expectation.
        # Here, self.string_kernel_mat is the precomputed Gram matrix K_B of shape (2^(n_qubits), 2^(n_qubits)).
        # We assume that the batch dimensions are the same (or can be merged) for simplicity.
        # For example, if x1 and x2 are 2D (i.e. shape (N, dim) and (M, dim)), then:

        expect_intra1 = compute_intra_mmd_expectation(global_probs1, string_kernel_mat)  # shape (N,)
        expect_intra2 = compute_intra_mmd_expectation(global_probs2, string_kernel_mat)  # shape (M,)

        # Compute cross expectation between the two batches.
        cross_exp = compute_cross_mmd_expectation(global_probs1, global_probs2, string_kernel_mat)  # shape (N, M)

        # Compute the squared MMD distance:
        kernel_mat = mmd_kernel(expect_intra1, expect_intra2, cross_exp, chamfer_gram_mat)

        return kernel_mat
