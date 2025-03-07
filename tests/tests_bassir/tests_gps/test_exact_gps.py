import math
import torch
import gpytorch
import pytest
from bassir.models.factory.gp_models import ExactGP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(params=["rbf", "bassir"])
def kernel_setup(request):
    """
    Parameterized fixture that returns functions for generating data,
    training the model, and evaluating predictions, based on the kernel type.
    """
    kernel_type = request.param

    def generate_data():
        """Generates training data. For the Bassir kernel, add an extra dimension."""
        train_x = torch.linspace(0, 1, 100)
        train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
        train_x = train_x.unsqueeze(-1)
        return train_x.to(torch.float64).to(DEVICE), train_y.to(torch.float64).to(DEVICE)

    def train_model(train_x, train_y, num_iters=5):
        """Trains a GP model with the chosen kernel."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if kernel_type == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == "bassir":
            # Import the Bassir kernel and related quantum modules.
            from bassir.models.quantum.bassir_kernel import BassirKernel
            from bassir.models.quantum.positioner import Positioner
            from bassir.models.quantum.rydberg import RydbergEvolver
            from bassir.utils.qutils import get_default_register_topology

            n_qubits = 4
            dim = train_x.shape[-1]
            traps = get_default_register_topology(topology="all_to_all", n_qubits=n_qubits)
            positioner = Positioner(traps, dim=dim)
            evolver = RydbergEvolver(traps=traps, dim=dim)
            kernel = BassirKernel(traps=traps, positioner=positioner, evolver=evolver)
        else:
            raise ValueError("Unknown kernel type")

        model = ExactGP(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel).to(DEVICE)
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
        """Evaluates the trained model on a fixed set of test points."""
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 51).to(DEVICE)
            observed_pred = likelihood(model(test_x))
        return observed_pred

    return kernel_type, generate_data, train_model, evaluate_model


def test_model_training(kernel_setup):
    """
    Tests whether the model is trained properly. For the RBF kernel, we check the lengthscale.
    For both kernels, we check that the likelihood noise is positive.
    """
    kernel_type, generate_data, train_model, _ = kernel_setup
    train_x, train_y = generate_data()
    print(f"train_x.dtype = {train_x.dtype}")
    model, likelihood = train_model(train_x, train_y, num_iters=10)

    assert model is not None
    assert likelihood is not None

    if kernel_type == "rbf":
        # RBF kernel should have a positive lengthscale.
        assert model.covar_module.lengthscale.item() > 0

    # For both kernels, the noise (variance) should be positive.
    assert model.likelihood.noise.item() > 0


def test_model_evaluation(kernel_setup):
    """
    Tests whether the model produces predictions of the expected shape.
    """
    _, generate_data, train_model, evaluate_model = kernel_setup
    train_x, train_y = generate_data()
    model, likelihood = train_model(train_x, train_y, num_iters=10)
    observed_pred = evaluate_model(model, likelihood)

    assert observed_pred is not None
    # We expect predictions on 51 test points.
    assert observed_pred.mean.shape[0] == 51


if __name__ == '__main__':
    pytest.main([__file__])
