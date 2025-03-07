import hydra
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
from bassir.utils.build import get_lightning_model
import os
from bassir.utils.settings import EXPERIMENTS_PATH, TRAIN_CONFIGS_PATH
from bassir.utils.loading.data_loaders import get_data_loaders
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

# Set the PROJECT_ROOT environment variable before initializing Hydra.
os.environ['EXPERIMENTS_PATH'] = EXPERIMENTS_PATH

torch.set_float32_matmul_precision('high')

# # Make PyTorch Lightning logs propagate to root logger
logger = logging.getLogger("pytorch_lightning")
logger.propagate = True
logger.setLevel(logging.INFO)


@hydra.main(config_path=TRAIN_CONFIGS_PATH, config_name="single_run")
def main(cfg: DictConfig):
    # Set the seed for reproducibility.
    if cfg.run_specs.seed:
        pl.seed_everything(cfg.run_specs.seed, workers=True)
        random.seed(cfg.run_specs.seed)
        np.random.seed(cfg.run_specs.seed)

    rank_zero_info("\n" + "Experiment specs:" + "\n" + OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()  # Get the Hydra config.

    # Let hydra manage direcotry outputs
    tensorboard_logger = pl.loggers.TensorBoardLogger(".", "", "", default_hp_metric=False)

    # Use Hydra’s run directory as the base save_dir:
    run_dir = hydra_cfg.run.dir

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_acc",  # or any validation metric you choose
        mode="max",
        save_top_k=1,
    )
    callbacks = [checkpoint_callback]

    train_loader, val_loader, test_loader = get_data_loaders(cfg.data)

    lightning_model = get_lightning_model(cfg=cfg.model, train_loader=train_loader)

    tensorboard_logger.experiment.add_text("model_repr", str(lightning_model), global_step=0)

    # Train the model
    # Log 10 times per epoch as default
    log_every_n_steps = max(1, len(train_loader) // 10) if not cfg.experiment.train_specs.log_every_n_steps else (
        cfg.experiment.train_specs.log_every_n_steps)
    trainer = pl.Trainer(max_epochs=cfg.experiment.train_specs.max_epochs,
                         accelerator=cfg.experiment.train_specs.accelerator,
                         devices=cfg.experiment.train_specs.devices,
                         log_every_n_steps=log_every_n_steps,
                         precision=cfg.experiment.train_specs.precision,
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         enable_progress_bar=False,
                         )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test the model using a separate trainer for testing if needed:
    test_trainer = pl.Trainer(accelerator=cfg.experiment.test_specs.accelerator,
                              devices=cfg.experiment.test_specs.devices,
                              enable_progress_bar=False,
                              logger=tensorboard_logger)

    rank_zero_info(test_trainer.test(lightning_model, dataloaders=test_loader))


if __name__ == "__main__":
    main()
