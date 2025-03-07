import gpytorch
from gpytorch.kernels import Kernel
import torch
from torch import Tensor
from gpytorch.likelihoods import GaussianLikelihood


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood, kernel: Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# define the actual GP model (kernels, inducing points, etc.)
class ApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel: Kernel):
        """

        :param inducing_points:
        :param kernel:
        """
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(ApproximateGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PGLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    This classe effectively computes the expected log likelihood contribution to Eqn (10) in Reference: "Florian Wenzel,
    Theo Galy-Fajou, Christan Donner, Marius Kloft, Manfred Opper. Efficient Gaussian process classification using
    Pòlya-Gamma data augmentation. AAAI. 2019."

    """

    def expected_log_prob(self, target, input, *args, **kwargs):
        mean, variance = input.mean, input.variance
        # Compute the expectation E[f_i^2]
        raw_second_moment = variance + mean.pow(2)

        # Translate targets to be -1, 1
        target = target.to(mean.dtype).mul(2.).sub(1.)

        # We detach the following variable since we do not want
        # to differentiate through the closed-form PG update.
        c = raw_second_moment.detach().sqrt()
        # Compute mean of PG auxiliary variable omega: 0.5 * Expectation[omega]
        # See Eqn (11) and Appendix A2 and A3 in Reference [1] for details.
        half_omega = 0.25 * torch.tanh(0.5 * c) / c

        # Expected log likelihood
        res = 0.5 * target * mean - half_omega * raw_second_moment
        # Sum over data points in mini-batch
        res = res.sum(dim=-1)

        return res

    # define the likelihood
    def forward(self, function_samples, **kwargs):
        return torch.distributions.Bernoulli(logits=function_samples)

    # define the marginal likelihood using Gauss Hermite quadrature
    def marginal(self, function_dist, **kwargs):
        def prob_fn(function_samples):
            return self.forward(function_samples).probs
        probs = self.quadrature(prob_fn, function_dist)
        return torch.distributions.Bernoulli(probs=probs)
