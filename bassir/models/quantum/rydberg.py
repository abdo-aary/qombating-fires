import torch
import torch.nn as nn
from torch import Tensor, vmap
from networkx import Graph, adjacency_matrix
from math import pi
from bassir.utils.qutils import get_binary_representation
import string

# Create a set of allowed characters with letters and digits.
allowed_chars = string.ascii_letters + "0123456789"  # 26+26+10 = 62 characters


def get_einsum_string(n: int) -> str:
    # Ensure we have enough unique labels: we need 2*n distinct characters.
    if 2 * n > len(allowed_chars):
        raise ValueError(f"n={n} is too high; need at most {len(allowed_chars) // 2} matrices.")
    # For each matrix, assign a pair of indices: one from the first half and one from the second half.
    # This gives a string like: "aB,cD,..." (if n=2, for example)
    input_subscripts = [f"{allowed_chars[k]}{allowed_chars[n + k]}" for k in range(n)]
    # The output will be the concatenation of the first n characters and then the next n characters.
    output_subscript = "".join(allowed_chars[:n] + allowed_chars[n:2 * n])
    return ",".join(input_subscripts) + "->" + output_subscript


# Precompute einsum strings for n in {1,...,30}
EINSUM_STRINGS = {n: get_einsum_string(n) for n in range(1, 31)}


def n_operator() -> Tensor:
    """Returns the single-qubit number operator n = (I+σ_z)/2."""
    identity2 = torch.eye(2, dtype=torch.cfloat)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    return (identity2 + sigma_z) / 2


def kron_product(matrices: Tensor) -> Tensor:
    """
    Compute the Kronecker product of a sequence of 2×2 matrices.
    Input:
        matrices: Tensor of shape (n, 2, 2)
    Output:
        Tensor of shape (2^n, 2^n)
    """
    n = matrices.shape[0]
    einsum_str = EINSUM_STRINGS[n]
    matrices_tuple = torch.unbind(matrices, dim=0)
    result = torch.einsum(einsum_str, *matrices_tuple)
    return result.reshape(2 ** n, 2 ** n)


class RydbergEvolver(nn.Module):
    def __init__(self, traps: Graph,
                 dim: int = None,
                 c_6_ceoff: float = 866,
                 reference_distance_d_0: float = 3.0,
                 initial_state: Tensor = None,
                 varyer: nn.Module = None):
        """
        Implements Rydberg-based time-independent evolution with input-dependent variational parameters.

        :param traps: Topology of the total system.
        :param dim: Input feature dimension. Needs to be set if no varyer object is provided
        :param c_6_ceoff: C₆ coefficient (e.g. 866).
        :param reference_distance_d_0: Reference distance (e.g., 3 μm).
        :param initial_state: Initial state; default is the zero state.
        :param varyer: A neural network mapping input features (shape (b_size, dim))
                        to 4 outputs corresponding to [raw_omega, delta, raw_phi, raw_time].
                        If None, a default MLP is used.
        """
        super().__init__()
        self.n_qubits = len(traps)
        self.reference_distance_d_0 = reference_distance_d_0
        self.interaction_weight = c_6_ceoff / reference_distance_d_0 ** 6

        # Precompute binary representation.
        binary_representation = get_binary_representation(self.n_qubits)  # shape (2**n_qubits, n_qubits)
        self.register_buffer("binary_representation", binary_representation)

        # Interaction adjacency matrix.
        adj_matrix = Tensor(adjacency_matrix(traps).toarray())
        self.register_buffer("adj_matrix", adj_matrix)
        self.register_buffer("n_operator", n_operator())

        # Compute and register the interaction operator tensor.
        self.register_buffer("interaction_operator_tensor", self._compute_interaction_operator_vectorized())

        # Prepare the initial state |0>^(⊗ n_qubits).
        self.initial_state = (initial_state if initial_state is not None else
                              torch.zeros(2 ** self.n_qubits, dtype=torch.cdouble, device=self.n_operator.device))
        self.initial_state[0] = 1.0

        # Define the varyer submodule: x \mapsto V_{\theta_2}(x).
        self.varyer = (nn.Sequential(
            nn.Linear(dim, (4 + dim) // 2),
            nn.ReLU(),
            nn.Linear((4 + dim) // 2, 4)
        ) if varyer is None else varyer)

    def _h_single_qubit(self, x: Tensor) -> Tensor:
        """
        Computes a batch of single-qubit Hamiltonians H(π) from input x via self.varyer.
        For each sample i:
          H[i] = (ω_i/2) (cos(φ_i) σₓ - sin(φ_i) σ_y) - δ_i * ((I+σ_z)/2),
        where ω_i = softplus(raw_omega_i), φ_i = raw_phi_i mod 2π,
        and δ_i is taken as-is.
        Input:
          x: Tensor of shape (b_size, dim)
        Returns:
          H: Tensor of shape (b_size, 2, 2)
        """
        b_size = x.shape[0]
        var_params = self.varyer(x)  # shape (b_size, 4)
        raw_omega = var_params[:, 0]
        delta = var_params[:, 1]
        raw_phi = var_params[:, 2]
        # Compute effective parameters.
        omega = torch.nn.functional.softplus(raw_omega)  # ensure ω > 0
        phi = raw_phi % (2 * pi)

        # Predefine constant matrices.
        device = x.device
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)
        identity2 = torch.eye(2, dtype=torch.cfloat, device=device)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
        number_op = (identity2 + sigma_z) / 2

        # Reshape scalars for broadcasting.
        omega = omega.view(b_size, 1, 1)
        phi = phi.view(b_size, 1, 1)
        delta = delta.view(b_size, 1, 1)

        # Compute the Hamiltonians.
        h = (omega / 2) * (torch.cos(phi) * sigma_x - torch.sin(phi) * sigma_y) - delta * number_op
        return h  # shape (b_size, 2, 2)

    def _compute_single_operator_vectorized(self, x: Tensor) -> Tensor:
        """
        Computes the single-qubit embedding tensors for each sample.
        For each sample i with gate H[i] = _h_single_qubit(x)[i] and for each qubit index j,
        we form
          E^{(i)}_j = I_2^{⊗ j} ⊗ H[i] ⊗ I_2^{⊗ (n_qubits - j - 1)}.
        Returns:
          Tensor of shape (b_size, n_qubits, 2^(n_qubits), 2^(n_qubits)).
        """
        b_size = x.shape[0]
        n = self.n_qubits
        # h_batch: shape (b_size, 2,2)
        h_batch = self._h_single_qubit(x)
        device = x.device
        identity2 = torch.eye(2, dtype=h_batch.dtype, device=device)
        # Create an index mask of shape (n, n) that is True on the diagonal.
        idx_mask = (torch.arange(n, device=device).unsqueeze(1) == torch.arange(n, device=device).unsqueeze(0))
        # Expand to shape (b_size, n, n, 1, 1) for broadcasting.
        idx_mask_exp = idx_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, n, n, 1, 1)
        # Expand h_batch and identity2 appropriately.
        h_exp = h_batch.unsqueeze(1).unsqueeze(1).expand(b_size, n, n, 2, 2)
        identity2_exp = identity2.expand(b_size, n, n, 2, 2)
        # For each sample, for each qubit index j, select h_batch for the j-th position and identity otherwise.
        m_mat = torch.where(idx_mask_exp, h_exp, identity2_exp)  # shape (b_size, n, n, 2, 2)
        # Now, for each sample i and qubit index j, compute the Kronecker product along axis 1 (over n matrices).
        # We use vmap twice to vectorize over the batch and the qubit index.

        embedding_tensor = vmap(vmap(kron_product, in_dims=0), in_dims=0)(m_mat)
        # embedding_tensor: shape (b_size, n, 2^n, 2^n)
        return embedding_tensor

    def _compute_interaction_operator_vectorized(self) -> Tensor:
        """
        Computes the two-qubit interaction tensor (unchanged).
        Returns a tensor of shape (n_qubits, n_qubits, 2^(n_qubits), 2^(n_qubits)).
        """
        n = self.n_qubits
        n_op = self.n_operator  # shape (2,2)
        identity2 = torch.eye(2, dtype=torch.cfloat, device=n_op.device)
        idx_i = torch.arange(n, device=n_op.device).unsqueeze(1).unsqueeze(2).expand(n, n, n)
        idx_j = torch.arange(n, device=n_op.device).unsqueeze(0).unsqueeze(2).expand(n, n, n)
        idx_k = torch.arange(n, device=n_op.device).unsqueeze(0).unsqueeze(0).expand(n, n, n)
        condition = (idx_k == idx_i) | ((idx_i != idx_j) & (idx_k == idx_j))
        m_matrix = torch.where(condition.unsqueeze(-1).unsqueeze(-1), n_op, identity2)
        kron_vmap = vmap(vmap(kron_product, in_dims=0), in_dims=0)
        interaction_operator_tensor = kron_vmap(m_matrix)
        return interaction_operator_tensor

    def forward(self, x: Tensor, mask_batch: Tensor) -> Tensor:
        """
        Computes the evolved state for a batch of inputs using input-dependent variational parameters.

        Args:
            x: Tensor of shape (batch_size, dim) used by the varyer to produce variational parameters.
            mask_batch: Tensor of shape (batch_size, n_qubits) produced by the Positioner.

        Returns:
            psi_out_batch: Tensor of shape (batch_size, 2^(n_qubits)) representing the evolved state.
        """
        b_size, n = mask_batch.shape

        # Compute sample-dependent single-qubit embedding tensors.
        embedding_tensor = self._compute_single_operator_vectorized(x)  # shape (b_size, n, 2^n, 2^n)
        # Multiply by the mask (reshaped for broadcasting) and sum over qubits.
        mask_reshaped = mask_batch.view(b_size, n, 1, 1)
        h_single_batch = (mask_reshaped * embedding_tensor).sum(dim=1)  # shape (b_size, 2^n, 2^n)

        # Interaction term (same as before).
        mask_outer = mask_batch.unsqueeze(2) * mask_batch.unsqueeze(1)  # (b_size, n, n)
        adj_exp = self.adj_matrix.unsqueeze(0)  # (1, n, n)
        weight_matrix = self.interaction_weight * adj_exp * mask_outer
        weight_matrix = weight_matrix.to(dtype=torch.cfloat)
        inter_op_exp = self.interaction_operator_tensor.unsqueeze(0)
        h_interaction_batch = torch.einsum('bij,bijlk->blk', weight_matrix, inter_op_exp)  # (b_size, 2^n, 2^n)

        # Effective Hamiltonian.
        h_eff = h_single_batch + h_interaction_batch  # (b_size, 2^n, 2^n)

        init_state = self.initial_state.unsqueeze(0).expand(b_size, -1).unsqueeze(-1).to(mask_batch.device)

        # Get time from varyer.
        var_params = self.varyer(x)  # shape (b_size, 4)
        raw_time = var_params[:, 3]
        time = torch.nn.functional.softplus(raw_time)

        exp_h = torch.linalg.matrix_exp(-1j * time.view(b_size, 1, 1) * h_eff)
        psi_out = (exp_h @ init_state).squeeze(-1)  # shape (b_size, 2^n)
        return psi_out
