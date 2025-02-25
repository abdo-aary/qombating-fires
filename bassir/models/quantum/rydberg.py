import torch
import torch.nn as nn
from torch import Tensor, vmap
from networkx import Graph, adjacency_matrix
from math import pi

# Precompute einsum strings for n in {1,...,10}
EINSUM_STRINGS = {
    n: ','.join([f"{chr(97 + k)}{chr(65 + k)}" for k in range(n)]) + '->' +
       ''.join([chr(97 + k) for k in range(n)] + [chr(65 + k) for k in range(n)])
    for n in range(1, 11)
}


def get_binary_tensor(num_qubits: int) -> Tensor:
    """
    Returns a tensor of shape (2**num_qubits, num_qubits) where each row is the binary representation
    (with 0s and 1s) of the integers 0,...,2**num_qubits - 1.
    """
    num_values = 2 ** num_qubits
    indices = torch.arange(num_values, dtype=torch.long)
    binary_tensor = ((indices.unsqueeze(1) & (1 << torch.arange(num_qubits))) > 0).long()
    return binary_tensor  # Shape: (2**num_qubits, num_qubits)


def n_operator() -> Tensor:
    """
    Returns the single-qubit number operator n = (I+σ_z)/2.
    """
    identity_2 = torch.eye(2, dtype=torch.cfloat)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    return (identity_2 + sigma_z) / 2


def kron_product(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute the Kronecker product of a sequence of 2x2 matrices without explicit loops
    during the heavy contraction. The input 'matrices' is a tensor of shape (n, 2, 2),
    where n is the number of matrices, and the output is a tensor of shape (2**n, 2**n).
    """
    n = matrices.shape[0]
    einsum_str = EINSUM_STRINGS[n]  # Retrieve precomputed einsum string.
    matrices_tuple = torch.unbind(matrices, dim=0)  # Returns a tuple of n tensors of shape (2,2)
    result = torch.einsum(einsum_str, *matrices_tuple)
    return result.reshape(2 ** n, 2 ** n)


class RydbergEvolver(nn.Module):
    def __init__(self, traps: Graph,
                 c_6_ceoff: float = 866,
                 reference_distance_d_0: float = 3,
                 initial_state: Tensor = None):
        """
        Implements the Rydberg-based time-independent evolution.

        :param traps: The topology of total system.
        :param c_6_ceoff: the used C_6 coefficient for a Rydberg level n \approx 60.
                          Default = 866 [rad/μs^6/ ns].
        :param reference_distance_d_0: the reference distance to use. Default = 3 [μm] that adheres to
                                       neutral-atoms hardware with 87RB at a principal quantum number n \approx 60.
        :param initial_state: initial state, default = the zero state.
        """
        super().__init__()
        self.n_qubits = len(traps)

        self.interaction_weight = c_6_ceoff / reference_distance_d_0 ** 6

        self.raw_params = nn.ParameterDict({
            "raw_omega": nn.Parameter(torch.tensor([15.0], dtype=torch.float32)),  # unconstrained
            "delta": nn.Parameter(torch.tensor([0.0], dtype=torch.float32)),  # unconstrained
            "raw_phi": nn.Parameter(torch.tensor([0.0], dtype=torch.float32)),  # unconstrained
            "raw_time": nn.Parameter(torch.tensor([1.0], dtype=torch.float32))  # unconstrained
        })

        # Precompute the binary index tensor (depends only on n_qubits).
        binary_tensor = get_binary_tensor(self.n_qubits)  # Shape: (2**n_qubits, n_qubits)
        self.register_buffer("binary_tensor", binary_tensor)

        # Compute the interaction adjacency matrix.
        adj_matrix = Tensor(adjacency_matrix(traps).toarray())
        self.register_buffer("adj_matrix", adj_matrix)
        self.register_buffer("n_operator", n_operator())

        # Compute and register the interaction operator tensor (vectorized version).
        # Shape: (n_qubits, n_qubits, 2**n_qubits, 2**n_qubits)
        self.register_buffer("interaction_operator_tensor", self._compute_interaction_operator_vectorized())

        # Prepare the initial state |0>^{⊗ n_qubits} for each batch element.
        self.initial_state = initial_state if initial_state else \
            torch.zeros(2 ** self.n_qubits, dtype=torch.cfloat, device=self.n_operator.device)
        self.initial_state[0] = 1.0  # |0>^{⊗ n_qubits}

    @property
    def omega(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_params["raw_omega"])

    @omega.setter
    def omega(self, value: torch.Tensor):
        # Ensure value is positive. Inverse softplus: raw = log(exp(value) - 1)
        self.raw_params["raw_omega"].data.copy_(torch.log(torch.exp(value) - 1))

    @property
    def time(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_params["raw_time"])

    @time.setter
    def time(self, value: torch.Tensor):
        self.raw_params["raw_time"].data.copy_(torch.log(torch.exp(value) - 1))

    @property
    def phi(self) -> torch.Tensor:
        # Wrap the phase to [0, 2*pi)
        return self.raw_params["raw_phi"] % (2 * pi)

    @phi.setter
    def phi(self, value: torch.Tensor):
        self.raw_params["raw_phi"].data.copy_(value)

    @property
    def delta(self) -> torch.Tensor:
        return self.raw_params["delta"]

    @delta.setter
    def delta(self, value: torch.Tensor):
        self.raw_params["delta"].data.copy_(value)

    def _h_single_qubit(self) -> Tensor:
        """
        Computes the single-qubit Hamiltonian H(pi) of shape (2,2) given the current parameters.
        """
        omega = self.omega
        phi = self.phi
        delta = self.delta
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=omega.device)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=omega.device)
        identity_2 = torch.eye(2, dtype=torch.cfloat, device=omega.device)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=omega.device)
        number_operator = (identity_2 + sigma_z) / 2  # (I+σ_z)/2
        h_single = (omega / 2) * (torch.cos(phi) * sigma_x - torch.sin(phi) * sigma_y) - delta * number_operator
        return h_single  # Shape: (2,2)

    def _compute_single_operator_vectorized(self) -> torch.Tensor:
        """
        Computes the single-qubit embedding tensor of shape
        (n_qubits, 2**n_qubits, 2**n_qubits) such that for each qubit index i,
        the operator is E_i = I_2^{⊗ i} ⊗ h_single ⊗ I_2^{⊗ (n_qubits-i-1)}.


        Returns:
          embedding_tensor: a tensor of shape (n_qubits, 2**n_qubits, 2**n_qubits).
        """
        h_single = self._h_single_qubit()
        n_qubits = self.n_qubits

        identity_2 = torch.eye(2, dtype=h_single.dtype, device=h_single.device)
        # Create a tensor of shape (n_qubits, n_qubits, 2,2) where for each i (row) and factor index k:
        # use h_single if k == i, else identity_2.
        n = n_qubits
        idx_i = torch.arange(n, device=h_single.device).view(n, 1).expand(n, n)
        idx_k = torch.arange(n, device=h_single.device).view(1, n).expand(n, n)
        # Broadcasting: condition has shape (n, n, 1, 1).
        condition = (idx_i == idx_k).unsqueeze(-1).unsqueeze(-1)
        # matrix_tensor has shape (n, n, 2, 2).
        matrix_tensor = torch.where(condition, h_single, identity_2)
        # For each qubit index i, matrix_tensor[i] is a stack of n matrices.
        # Use vmap to compute the Kronecker product for each i.
        embedding_tensor = vmap(kron_product)(matrix_tensor)  # Shape: (n, 2**n, 2**n)
        return embedding_tensor

    def _compute_interaction_operator_vectorized(self) -> Tensor:
        """
        Computes the two-qubit interaction tensor of shape
        (n_qubits, n_qubits, 2**n_qubits, 2**n_qubits) without explicit Python loops.
        For each pair (i, j), the interaction operator is defined as
          interaction_operator_tensor[i, j] = ⨂_{k=0}^{n_qubits-1} m[i,j,k],
        where for each qubit position k:
          m[i, j, k] = n_operator    if (k == i) or (k == j and i != j),
                       identity_2   otherwise.
        """
        num_qubits = self.n_qubits
        # n_operator and identity_2 are 2x2 matrices.
        n_op = self.n_operator  # shape (2, 2)
        identity_2 = torch.eye(2, dtype=torch.cfloat, device=n_op.device)

        # Create index tensors for i, j, k (all 0-indexed).
        idx_i = torch.arange(num_qubits, device=n_op.device).view(num_qubits, 1, 1).expand(num_qubits, num_qubits,
                                                                                           num_qubits)
        idx_j = torch.arange(num_qubits, device=n_op.device).view(1, num_qubits, 1).expand(num_qubits, num_qubits,
                                                                                           num_qubits)
        idx_k = torch.arange(num_qubits, device=n_op.device).view(1, 1, num_qubits).expand(num_qubits, num_qubits,
                                                                                           num_qubits)
        # Condition: for each (i, j, k):
        # if i == j, choose n_op only when k == i;
        # if i != j, choose n_op if k equals i or k equals j; otherwise identity.
        condition = (idx_k == idx_i) | ((idx_i != idx_j) & (idx_k == idx_j))
        # Build m_matrix of shape (num_qubits, num_qubits, num_qubits, 2, 2)
        m_matrix = torch.where(condition.unsqueeze(-1).unsqueeze(-1), n_op, identity_2)

        # Now define a helper that computes the Kronecker product of a sequence of 2x2 matrices.

        # Use functorch.vmap to apply kron_product over the first two dimensions.
        kron_vmap = vmap(vmap(kron_product, in_dims=0), in_dims=0)
        interaction_operator_tensor = kron_vmap(m_matrix)
        # The output has shape (num_qubits, num_qubits, 2**num_qubits, 2**num_qubits)
        return interaction_operator_tensor

    def forward(self, mask_batch: Tensor) -> Tensor:
        """
        :param mask_batch: Tensor of shape (batch_size, n_qubits) representing the mask for each input.
        :return: psi_out_batch: Tensor of shape (batch_size, 2**n_qubits) representing the evolved state.
        """
        batch_size, num_qubits = mask_batch.shape
        # mask_batch = mask_batch.float()  # Ensure differentiability.

        # Compute the single-qubit operator (embedding tensor) vectorized.
        embedding_tensor = self._compute_single_operator_vectorized()  # (n_qubits, 2**n_qubits, 2**n_qubits)
        embedding_tensor_expanded = embedding_tensor.unsqueeze(0)  # (1, n_qubits, 2**n_qubits, 2**n_qubits)
        mask_reshaped = mask_batch.view(batch_size, num_qubits, 1, 1)  # (batch_size, n_qubits, 1, 1)
        h_single_batch = (mask_reshaped * embedding_tensor_expanded).sum(
            dim=1)  # (batch_size, 2**n_qubits, 2**n_qubits)

        # Compute the interaction part.
        mask_outer = mask_batch.unsqueeze(2) * mask_batch.unsqueeze(1)  # (batch_size, n_qubits, n_qubits)
        adj_matrix_expanded = self.adj_matrix.unsqueeze(0)  # (1, n_qubits, n_qubits)
        weight_matrix = self.interaction_weight * adj_matrix_expanded * mask_outer  # (b_s, n_qubits, n_qubits)
        weight_matrix = weight_matrix.to(dtype=torch.cfloat)
        interaction_operator_tensor_expanded = self.interaction_operator_tensor.unsqueeze(
            0)  # (1, n_qubits, n_qubits, 2**n_qubits, 2**n_qubits)
        h_interaction_batch = torch.einsum('bij,bijlk->blk', weight_matrix,
                                           interaction_operator_tensor_expanded)  # (b_s, 2**n_qubits, 2**n_qubits)

        # Assemble the overall effective Hamiltonian.
        h_effective_batch = h_single_batch + h_interaction_batch  # (batch_size, 2**n_qubits, 2**n_qubits)

        # initial_state_batch.shape = (batch_size, 2**n_qubits, 1)
        initial_state_batch = self.initial_state.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1).to(mask_batch.device)

        # Compute the evolved state using a batched matrix exponential.
        exp_h_batch = torch.linalg.matrix_exp(
            -1j * self.time * h_effective_batch)  # (batch_size, 2**n_qubits, 2**n_qubits)
        psi_output_batch = (exp_h_batch @ initial_state_batch).squeeze(-1)  # (batch_size, 2**n_qubits)

        return psi_output_batch
