import gpytorch
import pytorch_lightning as pl
from gpytorch.likelihoods import Likelihood
from pytorch_lightning.utilities.types import STEP_OUTPUT
from bassir.models.factory.gp_models import ExactGP
from bassir.utils.metrics import MEAN_METRICS, PROBABILISTIC_METRICS, get_metric_fn
from torch import optim, Tensor

log_likelihood = get_metric_fn("LogLikelihood")


class ExactLightningGP(pl.LightningModule):
    def __init__(self, gp_model: ExactGP, likelihood: Likelihood, lr: float = 1e-3):
        super().__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.lr = lr

    @property
    def mll(self):
        return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        If ExactGP is used, batch = the whole train dataset

        :param batch: batch input
        :param batch_idx: it index
        :return: minus marginal log likelihood: - log p(y_train | X_train)
        """
        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        out = self(x)  # This computes the marginal distribution
        loss = - self.mll(out, y)
        self.log("train_mll", -loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes tests on the posterior p(y_val | y_pred)

        :param batch: (X_val, y_val)
        :param batch_idx: index
        :return:
        """
        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        post_dist = self.likelihood(self(x))  # This computes the multivariate posterior distribution
        ll = log_likelihood(post_dist, y)
        self.log("val_ll", ll)

    def test_step(self, batch, batch_idx):
        # this computes the marginal log likelihood on the test dataset
        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        post_dist = self.likelihood(self(x))  # This computes the multivariate posterior distribution

        for metric_name in MEAN_METRICS:
            metric_fn = get_metric_fn(metric_name)
            metric_value = metric_fn(prediction=post_dist.mean, target=y)
            self.log(metric_name, metric_value)

        # Compute the probabilistic metrics
        for metric_name in PROBABILISTIC_METRICS:
            metric_fn = get_metric_fn(metric_name)
            metric_value = metric_fn(prediction=post_dist, target=y)
            self.log(metric_name, metric_value)

    def configure_optimizers(self, ):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x: Tensor):
        return self.gp_model(x)
