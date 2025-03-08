import logging

import gpytorch
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
import torch

from bassir.models.factory.gp_models import ApproximateGP, PGLikelihood
from bassir.models.factory.lightning_approx import ApproxLightningGP
from bassir.models.quantum.bassir_kernel import BassirKernel
from bassir.models.quantum.embed_bassir_kernel import SimpleEmbedder, EmbedBassirKernel
from bassir.models.quantum.positioner import Positioner
from bassir.models.quantum.rydberg import RydbergEvolver
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from bassir.utils.qutils import get_topology

logger = logging.getLogger(__name__)


def get_lightning_model(cfg: DictConfig, train_loader: DataLoader) -> pl.LightningModule:
    # Get train data
    x_train, _ = train_loader.dataset.tensors
    n_train_samples, dim = x_train.size(0), x_train.size(-1)

    if cfg.kernel.name == "rbf":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif cfg.kernel.name == "bassir":
        traps = get_topology(cfg.kernel.topology)
        positioner = Positioner(traps=traps, projector=cfg.kernel.projector, dim=dim)
        evolver = RydbergEvolver(traps=traps, varyer=cfg.kernel.varyer, dim=dim)
        kernel = BassirKernel(traps=traps, positioner=positioner, evolver=evolver)
    elif cfg.kernel.name == "embed_bassir":
        traps = get_topology(cfg.kernel.topology)
        if cfg.kernel.embedder.type == "simple_embedder":
            # Set the dim_out to half the input dimension if "auto" is chosen.
            dim_out = dim // 2 if cfg.kernel.embedder.dim_out == "auto" else cfg.kernel.embedder.dim_out
            embedder = SimpleEmbedder(dim_in=dim, dim_out=dim_out)
            kernel = EmbedBassirKernel(traps=traps, embedder=embedder)
        else:
            raise ValueError("Unknown embedder type. Got {cfg.kernel.embedder.type}")
    else:
        raise ValueError("Unknown kernel type. Got {cfg.kernel.name}")

    # Initialize the inducing points
    # Set the number of inducing points to sqrt(n_train_samples) if no number is given
    n_ind_pts = int(math.sqrt(n_train_samples)) if not cfg.inducing_pts.num else cfg.inducing_pts.num
    kmeans_batch_size = get_auto_batch_size(n_train_samples, 128) if not cfg.inducing_pts.kmeans_batch_size\
        else cfg.inducing_pts.kmeans_batch_size

    kmeans = MiniBatchKMeans(n_clusters=n_ind_pts,
                             init='k-means++',
                             batch_size=kmeans_batch_size,
                             random_state=cfg.inducing_pts.random_state)
    kmeans.fit(x_train)
    inducing_points = torch.tensor(kmeans.cluster_centers_)
    logger.info(f"inducing_points.shape = {inducing_points.shape}")

    # Build the Lightning module
    model_gp = ApproximateGP(inducing_points=inducing_points, kernel=kernel)
    likelihood = PGLikelihood()
    lightning_model = ApproxLightningGP(gp_model=model_gp, likelihood=likelihood, num_data=n_train_samples)
    return lightning_model


def get_auto_batch_size(n_train_samples, max_batch_size):
    # Candidate is the lesser of the total samples and the maximum batch size.
    candidate = min(n_train_samples, max_batch_size)
    # Compute the power-of-2 that is less than or equal to candidate.
    # Ensure candidate is at least 1 to avoid log issues.
    candidate = max(1, candidate)
    batch_size = 2 ** int(math.floor(math.log(candidate, 2)))
    return batch_size
