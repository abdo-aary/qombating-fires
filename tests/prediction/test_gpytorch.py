import math
import torch
import gpytorch
import unittest
from src.models.gp_models.gp_model import GPModel


def generate_data():
    """Generates training data"""
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    return train_x.to('cuda'), train_y.to('cuda')


def train_model(train_x, train_y, num_iters=50):
    """Trains a GP model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.RBFKernel()
    model = GPModel(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel).to('cuda')

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood


def evaluate_model(model, likelihood):
    """Evaluates the trained model"""
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51).to('cuda')
        observed_pred = likelihood(model(test_x))

    return observed_pred


class TestGPModel(unittest.TestCase):
    """Unit tests for the GP Model"""

    def setUp(self):
        """Set up training data and model before each test"""
        self.train_x, self.train_y = generate_data()
        self.model, self.likelihood = train_model(self.train_x, self.train_y, num_iters=10)

    def test_model_training(self):
        """Test if the model parameters are optimized"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.likelihood)

        # Ensure lengthscale and noise are positive
        self.assertGreater(self.model.covar_module.lengthscale.item(), 0)
        self.assertGreater(self.model.likelihood.noise.item(), 0)

    def test_model_evaluation(self):
        """Test if the model makes predictions"""
        observed_pred = evaluate_model(self.model, self.likelihood)
        self.assertIsNotNone(observed_pred)
        self.assertEqual(observed_pred.mean.shape[0], 51)  # 51 test points


if __name__ == '__main__':
    unittest.main()
