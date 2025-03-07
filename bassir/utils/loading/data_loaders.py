from typing import Tuple, Union
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

from bassir.utils.build import get_auto_batch_size
from bassir.utils.loading.utils import generate_complex_synthetic_data


def get_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, val, and test dataloaders depending on the data

    :param cfg: the input config
    :return: train_loader, val_loader, test_loader
    """
    if cfg.name == "complex_synthetic":
        return get_complex_synth_data_loaders(n_samples=cfg.data_specs.n_samples,
                                              dim=cfg.data_specs.dim,
                                              noise_std=cfg.data_specs.noise_std,
                                              num_workers=cfg.loader.specs.num_workers,
                                              train_ratio=cfg.loader.specs.train_ratio,
                                              val_ratio=cfg.loader.specs.val_ratio,
                                              train_b_size=cfg.loader.specs.train_b_size,
                                              val_b_size=cfg.loader.specs.val_b_size,
                                              test_b_size=cfg.loader.specs.test_b_size,
                                              max_b_size=cfg.loader.specs.max_b_size)

    elif cfg.name == "wildfires":
        raise NotImplementedError(f"Handling the {cfg.name} data is not yet provided.")
    else:
        raise ValueError("Unknown data {cfg.name}!")


def get_complex_synth_data_loaders(n_samples: int, dim: int, noise_std: float = 0.1,
                                   num_workers: Union[int, str] = "auto",
                                   train_ratio: float = 0.7, val_ratio: float = 0.1,
                                   train_b_size: Union[int, str] = "auto",
                                   val_b_size: Union[int, str] = "auto",
                                   test_b_size: Union[int, str] = "auto",
                                   max_b_size: int = 128):
    if isinstance(num_workers, int):
        assert num_workers <= os.cpu_count(), (f"Got num_workers = {num_workers}, "
                                               f"which is bigger than the maximum number {os.cpu_count()}")
    elif isinstance(num_workers, str):
        assert num_workers == "auto", f"Unknown num_workers ``{num_workers}``. Needs to be set to``auto`` or an int."

    # 1. Generate the dataset
    x, y = generate_complex_synthetic_data(n_samples=n_samples, dim=dim, noise_std=noise_std)

    # 2. Split the dataset: 70% train, 10% validation, 20% test
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)

    # Randomly permute indices for splitting
    indices = torch.randperm(n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Create splits
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # 3. Create datasets and DataLoaders with one batch per split
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    # DataLoaders with batch_size equal to dataset length
    if num_workers == "auto":
        num_workers = max(1, os.cpu_count() - 1)
    if train_b_size == "auto":
        train_b_size = get_auto_batch_size(y_train.size(0), max_batch_size=max_b_size)
    if val_b_size == "auto":
        val_b_size = get_auto_batch_size(y_val.size(0), max_batch_size=max_b_size)
    if test_b_size == "auto":
        test_b_size = get_auto_batch_size(y_test.size(0), max_batch_size=max_b_size)

    train_loader = DataLoader(train_dataset, batch_size=train_b_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_b_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_b_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader
