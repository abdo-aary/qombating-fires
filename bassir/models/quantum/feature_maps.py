import math
import torch
import gpytorch
import qadence as qd
from qadence import RydbergDevice, IdealDevice, Register, RydbergEvolution
from qadence.parameters import VariationalParameter, FeatureParameter
import networkx as nx
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
import numpy as np
import unittest
import pickle
from typing import Any, Union, List, Tuple

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- Neural Network Mapping (Simplified for Testing) ---
def map_input_to_positions(x: Tensor, traps: nx.Graph, max_nodes: int = 5) -> List[int]:
    """Simplified mapping of input vector to trap positions (placeholder for NN with Gumbel-Softmax)."""
    # For simplicity in testing, map x to a subset of nodes based on its value
    num_nodes = min(max_nodes, len(traps.nodes))
    idx = int(torch.sigmoid(x).item() * num_nodes) + 1  # Scale to 1 to max_nodes
    return list(traps.nodes)[:idx]  # Return subset of nodes



# --- Unit Test ---
class TestQuantumKernelGP(unittest.TestCase):
    def setUp(self):
        """Set up training data and model before each test."""
        # Generate synthetic data
        self.train_x = torch.linspace(0, 1, 10).to(DEVICE)  # Reduced size for speed
        self.train_y = torch.sin(self.train_x * (2 * math.pi)) + torch.randn(self.train_x.size(),
                                                                             device=DEVICE) * math.sqrt(0.04)

        # Initialize traps (global topology)
        self.traps = get_default_register_topology('triangular_lattice', n_rows=2, n_cols=3, spacing=1.0)

        # Initialize model
        self.likelihood = GaussianLikelihood().to(DEVICE)
        self.kernel = QuantumKernel(traps=self.traps, max_nodes=3).to(DEVICE)
        self.model = GPModel(self.train_x, self.train_y, self.likelihood, self.kernel).to(DEVICE)

        # Train the model
        train(self.model, self.likelihood, training_iter=10, train_x=self.train_x, train_y=self.train_y)

    def test_model_training(self):
        """Test if the model parameters are optimized."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.likelihood)
        self.assertGreater(self.model.likelihood.noise.item(), 0)  # Noise should be positive

    def test_model_evaluation(self):
        """Test if the model makes predictions."""
        test_x = torch.linspace(0, 1, 5).to(DEVICE)
        observed_pred = predict(self.model, self.likelihood, test_x)
        self.assertIsNotNone(observed_pred)
        self.assertEqual(observed_pred.mean.shape[0], 5)  # 5 test points

        # Save and load predictions
        save_predictions(observed_pred, "test_pred.pkl")
        loaded_pred = load_predictions("test_pred.pkl")
        self.assertTrue(torch.allclose(observed_pred.mean, loaded_pred.mean))


if __name__ == '__main__':
    unittest.main()