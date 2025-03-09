import gpytorch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from bassir.models.factory.gp_models import ApproximateGP, PGLikelihood
from torch import optim, Tensor


class ApproxLightningGP(pl.LightningModule):
    def __init__(self, gp_model: ApproximateGP,
                 likelihood: PGLikelihood,
                 num_data: int,
                 lr_var: float = 1e-3,
                 lr_hyper: float = 1e-4):
        """
        Implements a Lightning rapper over the GP model

        :param gp_model:
        :param likelihood:
        :param num_data: the number of datapoints in the train dataset
        :param lr_var: the learning rate for variational parameters
        :param lr_hyper: the learning rate for the GP's (hyper) parameters
        """
        super().__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.num_data = num_data
        self.lr_var = lr_var
        self.lr_hyper = lr_hyper
        self.mll_produced = False
        self.automatic_optimization = False

    @property
    def mll(self):
        return gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, self.num_data)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Defines the training step by maximizing the marginal log likelihood

        :param batch: batch input
        :param batch_idx: it index
        :return: minus marginal log likelihood: - log p(y_train | X_train)
        """
        var_optim, hyper_optim = self.optimizers()
        var_optim.zero_grad()
        hyper_optim.zero_grad()

        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        out = self(x)  # This computes the marginal distribution
        loss = - self.mll(out, y)
        self.manual_backward(loss)  # Backpropagate gradients
        self.log("train_mll", -loss)  # Log the batch marginal log likelihood

        var_optim.step()  # Optimize variational parameters
        hyper_optim.step()  # Optimize hyperparameters

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes validation results on the posterior p(y_val | y_pred)

        :param batch: (X_val, y_val)
        :param batch_idx: index
        """
        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        post_dist = self.likelihood(self(x))  # This computes the multivariate posterior distribution
        ll = self.likelihood.log_marginal(y, post_dist).mean()
        acc = (post_dist.probs.gt(0.5) == y.bool()).float().mean()
        self.log("val_acc", acc, sync_dist=True)  # Log the validation accuracy
        self.log("val_ll", ll, sync_dist=True)  # Log the validation log likelihood

    def test_step(self, batch, batch_idx):
        """
        Computes test results on the posterior p(y_test | y_pred)

        :param batch: (X_val, y_val)
        :param batch_idx: index
        """
        # this computes the marginal log likelihood on the test dataset
        x, y = batch  # x.shape (batch_size, dim), y.shape = (batch_size)
        post_dist = self.likelihood(self(x))  # This computes the multivariate posterior distribution
        ll = self.likelihood.log_marginal(y, post_dist).mean()  # Log the validation log likelihood
        acc = (post_dist.probs.gt(0.5) == y.bool()).float().mean()  # Log the validation accuracy
        self.log("test_acc", acc, sync_dist=True)
        self.log("test_ll", ll, sync_dist=True)

    def configure_optimizers(self, ):
        variational_ngd_optimizer = gpytorch.optim.NGD(self.gp_model.variational_parameters(), num_data=self.num_data,
                                                       lr=self.lr_var)
        hyperparameter_optimizer = optim.Adam([
            {'params': self.gp_model.hyperparameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr_hyper)

        return variational_ngd_optimizer, hyperparameter_optimizer

    def forward(self, x: Tensor):
        return self.gp_model(x)
