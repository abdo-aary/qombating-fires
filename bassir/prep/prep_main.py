import os

import hydra
from omegaconf import DictConfig

from bassir.prep.temporal_prep import WildfireWindowDataset, IGNORE_COLS
from bassir.utils.settings import STORAGE_PATH, DATA_CONFIGS_PATH


@hydra.main(config_path=DATA_CONFIGS_PATH, config_name="wildfires")
def main(cfg: DictConfig):
    wildfire_path = os.path.join(STORAGE_PATH, "dataset", cfg.name, cfg.name + "_data.csv")
    dataset = WildfireWindowDataset(
        data_csv_path=wildfire_path,
        date_col="DATE",
        cell_lat_col="CELL_LAT",
        cell_lon_col="CELL_LON",
        label_col="IS_FIRE_NEXT_DAY",
        ignore_cols=IGNORE_COLS,
        window_size=cfg.specs.window_size,
        train_ratio=cfg.loader.specs.train_ratio,
        val_ratio=cfg.loader.specs.val_ratio,
        bal_factor=cfg.specs.bal_factor
    )
    dataset_object_path = os.path.join(STORAGE_PATH, "dataset", cfg.name, "preprocessed_dataset.pkl")

    dataset.save(dataset_object_path)


if __name__ == "__main__":
    main()
