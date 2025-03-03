import math
import torch
import gpytorch
import pytest
from bassir.models.factory.gp_model import GPModel
from bassir.models.quantum.bassir_kernel import BassirKernel
from bassir.models.quantum.positioner import Positioner
from bassir.models.quantum.rydberg import RydbergEvolver
from bassir.utils.qutils import get_default_register_topology

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_data():
    """Generates training data"""
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    train_x = train_x.unsqueeze(-1)
    return train_x.to(DEVICE), train_y.to(DEVICE)


def train_model(train_x, train_y, num_iters=50):
    """Trains a GP model"""
    n_qubits = 4
    dim = train_x.shape[-1]

    traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
    positioner = Positioner(dim, traps)
    evolver = RydbergEvolver(traps=traps, dim=dim)
    kernel = BassirKernel(traps=traps, positioner=positioner, evolver=evolver)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel).to(DEVICE)

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
        test_x = torch.linspace(0, 1, 51).to(DEVICE)
        observed_pred = likelihood(model(test_x))

    return observed_pred


@pytest.fixture
def trained_model():
    """Fixture that sets up the model before each test"""
    train_x, train_y = generate_data()
    model, likelihood = train_model(train_x, train_y, num_iters=10)
    return model, likelihood


def test_model_training(trained_model):
    """Test if the model parameters are optimized"""
    model, likelihood = trained_model

    assert model is not None
    assert likelihood is not None

    # Ensure noise is positive
    assert model.likelihood.noise.item() >= 0


def test_model_evaluation(trained_model):
    """Test if the model makes predictions"""
    model, likelihood = trained_model
    observed_pred = evaluate_model(model, likelihood)

    assert observed_pred is not None
    assert observed_pred.mean.shape[0] == 51  # 51 test points
