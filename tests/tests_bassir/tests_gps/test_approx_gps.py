import math
import torch
import gpytorch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from bassir.models.factory.gp_models import ApproximateGP, PGLikelihood

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


@pytest.fixture(params=["rbf", "bassir"])
def kernel_setup(request):
    """
    Parameterized fixture for approximate GP testing.
    Provides helper functions for generating binary classification data,
    creating DataLoaders (with train, validation, and test splits),
    training the model, and evaluating its performance, based on the kernel type.
    """
    kernel_type = request.param

    def generate_data():
        """Generate synthetic binary classification data with n_data = 200."""
        n_data = 200
        x_data = torch.linspace(-1., 1., n_data)
        # Compute probabilities from a sinusoidal function:
        probs = (torch.sin(x_data * math.pi).add(1.).div(2.))
        y = torch.distributions.Bernoulli(probs=probs).sample()
        x_data = x_data.unsqueeze(-1)  # shape: (N, 1)
        # Move data to DEVICE:
        x_data = x_data.to(DEVICE)
        y = y.to(DEVICE)
        # Split: 70% train, 10% validation, 20% test.
        train_n = int(0.7 * n_data)  # 140
        val_n = int(0.1 * n_data)    # 20
        indices = torch.randperm(n_data)
        train_x = x_data[indices[:train_n]].contiguous()
        train_y = y[indices[:train_n]].contiguous()
        val_x = x_data[indices[train_n:train_n + val_n]].contiguous()
        val_y = y[indices[train_n:train_n + val_n]].contiguous()
        test_x = x_data[indices[train_n + val_n:]].contiguous()
        test_y = y[indices[train_n + val_n:]].contiguous()
        return train_x, train_y, val_x, val_y, test_x, test_y

    def create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y):
        """Create DataLoaders for train, validation, and test splits."""
        batch_size_train = 16
        batch_size_val = 8
        batch_size_test = 8
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        test_dataset = TensorDataset(test_x, test_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
        return train_loader, val_loader, test_loader

    def train_model(train_loader, num_epochs=3):
        """
        Trains an approximate GP classifier.
        Sets up inducing points, builds the specified kernel,
        and optimizes variational parameters using NGD and hyperparameters using Adam.
        """
        # Use one batch to get shape info:
        train_x, train_y = next(iter(train_loader))
        n_inducing_pts = 10  # number of inducing points
        inducing_points = torch.linspace(-2., 2., n_inducing_pts, dtype=train_x.dtype,
                                         device=train_x.device).unsqueeze(-1)
        if kernel_type == "rbf":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == "bassir":
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

        model = ApproximateGP(inducing_points=inducing_points, kernel=kernel)
        if kernel_type == "rbf":
            model.covar_module.base_kernel.initialize(lengthscale=0.2)
        likelihood = PGLikelihood()

        model = model.to(DEVICE)
        likelihood = likelihood.to(DEVICE)

        variational_optimizer = gpytorch.optim.NGD(
            model.variational_parameters(), num_data=train_y.size(0), lr=0.1
        )
        hyper_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()}
        ], lr=0.01)

        model.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                variational_optimizer.zero_grad()
                hyper_optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                variational_optimizer.step()
                hyper_optimizer.step()
        return model, likelihood

    def evaluate_model(model, likelihood, loader):
        """
        Evaluates the model on a given DataLoader.
        Returns the mean negative log marginal likelihood and accuracy.
        """
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            x, y = loader.dataset.tensors
            outputs = model(x)
            nll = -likelihood.log_marginal(y, outputs)
            preds = likelihood(outputs).probs
            acc = (preds.gt(0.5) == y.bool()).float().mean()
        return nll.mean().item(), acc.item()

    return kernel_type, generate_data, create_dataloaders, train_model, evaluate_model


def test_approximate_gp_training(kernel_setup):
    """
    Test that the approximate GP model trains without error and produces valid model objects.
    """
    kernel_type, generate_data, create_dataloaders, train_model, _ = kernel_setup
    train_x, train_y, val_x, val_y, test_x, test_y = generate_data()
    train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y)
    model, likelihood = train_model(train_loader, num_epochs=3)
    assert model is not None
    assert likelihood is not None


def test_approximate_gp_evaluation(kernel_setup):
    """
    Test that the approximate GP model produces valid predictions on both validation and test sets.
    Checks that the accuracy is between 0 and 1 and the NLL is finite.
    """
    kernel_type, generate_data, create_dataloaders, train_model, evaluate_model = kernel_setup
    train_x, train_y, val_x, val_y, test_x, test_y = generate_data()
    train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y)
    model, likelihood = train_model(train_loader, num_epochs=3)
    test_nll, test_acc = evaluate_model(model, likelihood, test_loader)
    val_nll, val_acc = evaluate_model(model, likelihood, val_loader)
    assert 0.0 <= test_acc <= 1.0
    assert test_nll > -float('inf')
    assert 0.0 <= val_acc <= 1.0
    assert val_nll > -float('inf')


if __name__ == '__main__':
    pytest.main([__file__])
