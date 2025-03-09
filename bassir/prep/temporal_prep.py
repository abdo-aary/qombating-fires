from typing import Union

import pandas as pd
import numpy as np
import os
import logging
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader

from bassir.prep.utils import preprocess_windows, get_augmented_agg_data
from bassir.utils.build import get_auto_batch_size

logger = logging.getLogger(__name__)

IGNORE_COLS = ["CELL_LAT", "CELL_LON", "COORDINATES_LAT", "COORDINATES_LON", "AREA_HA", "SIZE_HA", "BURNED_DENSITY",
               "IS_FIRE"]
COLS = ['DATE', 'CELL_LAT', 'CELL_LON', 'D2M', 'T2M', 'DEPTHBELOWLANDLAYER', 'STL1', 'U10', 'V10', 'SP', 'LAI_HV',
        'LAI_LV', 'CVH', 'COORDINATES_LAT', 'COORDINATES_LON', 'AREA_HA', 'RH', 'VPD', 'SIZE_HA', 'BURNED_DENSITY',
        'IS_FIRE', 'TP', 'IS_FIRE_NEXT_DAY']

torch.set_float32_matmul_precision('high')


class WildfireWindowDataset:
    """
    Reads a CSV of wildfire data with columns (for example):
      - DATE (daily frequency)
      - CELL_LAT, CELL_LON (identifies the cell)
      - COORDINATES_LAT, COORDINATES_LON (actual geographic coordinates)
      - multiple feature columns (e.g., weather, vegetation)
      - IS_FIRE_NEXT_DAY (0 or 1, label for whether next day has a fire)

    Creates a window-based dataset for time-series classification:
      - X of shape (num_samples, window_size, n_features)
      - y of shape (num_samples,)
    Also stores the date, CELL_LAT, CELL_LON, COORDINATES_LAT, COORDINATES_LON for each sample
    so that test predictions can later be merged with these identifiers.
    """

    def __init__(
            self,
            data_csv_path: str = None,
            date_col: str = "DATE",
            cell_lat_col: str = "CELL_LAT",
            cell_lon_col: str = "CELL_LON",
            label_col: str = "IS_FIRE_NEXT_DAY",
            ignore_cols: list = None,
            window_size: int = 7,
            train_ratio: float = 0.8,
            val_ratio: float = 0.05,
            bal_factor: int = 10,  # The majority class will have bal_factor more than the minority class
    ):
        """
        :param data_csv_path: Path to the CSV file.
        :param date_col: Name of the date column.
        :param cell_lat_col: Name of the cell latitude column.
        :param cell_lon_col: Name of the cell longitude column.
        :param label_col: Name of the label column.
        :param ignore_cols: List of columns to exclude from the feature set.
        :param window_size: Number of days in each input window.
        :param train_ratio: Fraction of samples for training.
        :param val_ratio: Fraction of samples for validation.
        :param bal_factor: Desired ratio (majority:minority) for label 0 to label 1 samples.
        """
        self.data_csv_path = data_csv_path
        self.date_col = date_col
        self.cell_lat_col = cell_lat_col
        self.cell_lon_col = cell_lon_col
        self.label_col = label_col
        self.ignore_cols = ignore_cols if ignore_cols else []
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        # Arrays for data and metadata
        self.X = None  # shape: (num_samples, window_size, n_features)
        self.y = None  # shape: (num_samples,)
        self.sample_dates = []  # date for each sample (label date)
        self.sample_cell_lats = []  # CELL_LAT for each sample
        self.sample_cell_lons = []  # CELL_LON for each sample
        self.sample_coord_lats = []  # COORDINATES_LAT for each sample
        self.sample_coord_lons = []  # COORDINATES_LON for each sample

        # Full (unbalanced) dataset caches (for test sample selection)
        self.X_full = None
        self.y_full = None
        self.sample_dates_full = None
        self.sample_cell_lats_full = None
        self.sample_cell_lons_full = None
        self.sample_coord_lats_full = None
        self.sample_coord_lons_full = None

        # Indices for splits (balanced data)
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # Aggregated test sample attributes
        self.X_agg_test_samples = None  # shape: (n_cells, d_agg)
        self.y_test_samples = None  # shape: (n_cells,)
        self.test_sample_dates = None  # the selected test sample date

        # Save initial column lists (optional)
        self.cols = None
        self.feature_cols = None

        # 1. Prepare full unbalanced data (and metadata)
        self._prepare_data()

        # 2. Balance the data (for training/validation) and split without future leakage
        self._balance_data(bal_factor)
        self._split_data_no_future_leakage()

        # 3. Prepare aggregated test sample from the full unbalanced test set
        self._prepare_test_samples()

    def _prepare_data(self):
        """Reads the CSV, sorts by cell/date, extracts windows, and builds X, y plus metadata arrays."""
        df = pd.read_csv(self.data_csv_path)

        # Convert date column to datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Sort by (CELL_LAT, CELL_LON, DATE)
        df.sort_values(by=[self.cell_lat_col, self.cell_lon_col, self.date_col], inplace=True)

        # Identify feature columns: remove date, cell lat/lon, label_col, plus ignore_cols.
        # We'll leave COORDINATES_LAT and COORDINATES_LON in the DataFrame.
        all_cols = list(df.columns)
        self.cols = all_cols

        remove_cols = {self.date_col, self.cell_lat_col, self.cell_lon_col, self.label_col} | set(self.ignore_cols)
        feature_cols = [c for c in all_cols if c not in remove_cols]
        self.feature_cols = feature_cols

        # Group by each cell so we can create time-series windows.
        grouped = df.groupby([self.cell_lat_col, self.cell_lon_col])

        X_list = []
        y_list = []
        date_list = []
        cell_lat_list = []
        cell_lon_list = []
        coord_lat_list = []
        coord_lon_list = []

        for (cell_lat, cell_lon), group_df in grouped:
            group_df = group_df.reset_index(drop=True)
            group_features = group_df[feature_cols].values  # shape: (num_timesteps, n_features)
            group_labels = group_df[self.label_col].values
            group_dates = group_df[self.date_col].values

            # For each cell, extract all valid windows
            for start_idx in range(0, len(group_df) - self.window_size + 1):
                end_idx = start_idx + self.window_size
                window_features = group_features[start_idx:end_idx, :]
                label = group_labels[end_idx - 1]

                X_list.append(window_features)
                y_list.append(label)
                # Use the date of the last day in the window
                date_list.append(group_dates[end_idx - 1])
                cell_lat_list.append(cell_lat)
                cell_lon_list.append(cell_lon)
                # Use COORDINATES if available; otherwise fallback to cell coordinates
                if "COORDINATES_LAT" in df.columns:
                    coord_lat = group_df["COORDINATES_LAT"].iloc[end_idx - 1]
                else:
                    coord_lat = cell_lat
                if "COORDINATES_LON" in df.columns:
                    coord_lon = group_df["COORDINATES_LON"].iloc[end_idx - 1]
                else:
                    coord_lon = cell_lon
                coord_lat_list.append(coord_lat)
                coord_lon_list.append(coord_lon)

        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.sample_dates = np.array(date_list)
        self.sample_cell_lats = np.array(cell_lat_list)
        self.sample_cell_lons = np.array(cell_lon_list)
        self.sample_coord_lats = np.array(coord_lat_list)
        self.sample_coord_lons = np.array(coord_lon_list)

        # Cache the full unbalanced arrays (to be used for test sample selection)
        self.X_full = self.X.copy()
        self.y_full = self.y.copy()
        self.sample_dates_full = self.sample_dates.copy()
        self.sample_cell_lats_full = self.sample_cell_lats.copy()
        self.sample_cell_lons_full = self.sample_cell_lons.copy()
        self.sample_coord_lats_full = self.sample_coord_lats.copy()
        self.sample_coord_lons_full = self.sample_coord_lons.copy()

    def _balance_data(self, bal_factor: int):
        """
        Balances the dataset so that the number of majority class (label 0) samples
        is equal to bal_factor times the number of minority class (label 1) samples.
        Updates self.X, self.y, and all metadata arrays accordingly. Here we balance the data by random
        undersampling from the majority class.
        """
        # Find indices for minority and majority classes
        minority_indices = np.where(self.y == 1)[0]
        majority_indices = np.where(self.y == 0)[0]
        n_min = len(minority_indices)
        desired_majority = bal_factor * n_min

        if len(majority_indices) > desired_majority:
            selected_majority = np.random.choice(majority_indices, size=desired_majority, replace=False)
        else:
            selected_majority = majority_indices

        combined_indices = np.concatenate([minority_indices, selected_majority])
        np.random.shuffle(combined_indices)

        self.X = self.X[combined_indices]
        self.y = self.y[combined_indices]
        self.sample_dates = self.sample_dates[combined_indices]
        self.sample_cell_lats = self.sample_cell_lats[combined_indices]
        self.sample_cell_lons = self.sample_cell_lons[combined_indices]
        self.sample_coord_lats = self.sample_coord_lats[combined_indices]
        self.sample_coord_lons = self.sample_coord_lons[combined_indices]

    def _split_data_no_future_leakage(self):
        """
        Naively split data into train/val/test sets.
        For a strict chronological split, you would sort by self.sample_dates and then
        choose splits based on date cutoffs.
        """
        num_samples = len(self.X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        train_end = int(self.train_ratio * num_samples)
        val_end = int((self.train_ratio + self.val_ratio) * num_samples)

        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

    def _prepare_test_samples(self, scaler=None):
        """
        Prepares aggregated test samples from the full (unbalanced) dataset for a contiguous week.

        Steps:
          1. Identify the set of unique test dates from the full data.
          2. Slide a window of length equal to self.window_size (e.g. 7 days) over these dates,
             and for each contiguous block (i.e. where the difference between the last and first is exactly window_size - 1 days),
             compute the total number of fire flags.
          3. Select the contiguous week (block) with the maximum total fire flags.
          4. For each day in that week, select one sample per unique cell.
          5. Aggregate these selected windows using preprocess_windows to obtain:
                - self.X_agg_test_samples of shape (window_size * n_cells, d_agg)
                - self.y_test_samples of shape (window_size * n_cells,)
             Also cache the per-sample dates and cell metadata for later CSV production.
          6. Clear the full-data caches.
        """
        # --- Step 1: Get unique test dates (sorted) from full unbalanced data ---
        test_dates = np.unique(self.sample_dates_full)
        test_dates = np.sort(test_dates)  # assuming these are numpy datetime64 or strings in YYYY-MM-DD format
        # Convert to pandas datetime for easy date arithmetic
        test_dates_pd = pd.to_datetime(test_dates)

        block_length = self.window_size  # number of days for the test week
        n_dates = len(test_dates_pd)
        best_block = None
        best_block_sum = -1

        # --- Step 2: Slide a window over the test dates ---
        for i in range(n_dates - block_length + 1):
            block = test_dates_pd[i:i + block_length]
            # Check if the block is contiguous: difference between last and first equals block_length - 1 days
            if (block[-1] - block[0]).days == block_length - 1:
                mask_block = np.isin(self.sample_dates_full, block)
                block_sum = np.sum(self.y_full[mask_block])
                if block_sum > best_block_sum:
                    best_block_sum = block_sum
                    best_block = block

        if best_block is None:
            # Fallback: choose the last block_length dates
            best_block = test_dates_pd[-block_length:]
        # Convert best_block to an array of strings in YYYY-MM-DD format
        best_block_str = best_block.strftime("%Y-%m-%d")

        # --- Step 3: For each day in the chosen week, select one window per unique cell ---
        X_week_list = []
        y_week_list = []
        test_sample_dates = []  # store date corresponding to each aggregated sample
        unique_cell_lat_week = []
        unique_cell_lon_week = []
        unique_coord_lat_week = []
        unique_coord_lon_week = []

        for d in best_block_str:
            # Convert d to numpy datetime64 if necessary
            d_val = np.datetime64(d)
            mask_day = self.sample_dates_full == d_val
            X_day = self.X_full[mask_day]
            y_day = self.y_full[mask_day]
            cell_lat_day = self.sample_cell_lats_full[mask_day]
            cell_lon_day = self.sample_cell_lons_full[mask_day]
            coord_lat_day = self.sample_coord_lats_full[mask_day]
            coord_lon_day = self.sample_coord_lons_full[mask_day]

            unique_cells = {}
            X_day_selected = []
            y_day_selected = []
            cell_lat_sel = []
            cell_lon_sel = []
            coord_lat_sel = []
            coord_lon_sel = []
            for i, (lat, lon) in enumerate(zip(cell_lat_day, cell_lon_day)):
                cell_id = (lat, lon)
                if cell_id not in unique_cells:
                    unique_cells[cell_id] = True
                    X_day_selected.append(X_day[i])
                    y_day_selected.append(y_day[i])
                    cell_lat_sel.append(lat)
                    cell_lon_sel.append(lon)
                    coord_lat_sel.append(coord_lat_day[i])
                    coord_lon_sel.append(coord_lon_day[i])
            X_week_list.extend(X_day_selected)
            y_week_list.extend(y_day_selected)
            # Extend dates and metadata arrays with one entry per unique cell of that day
            test_sample_dates.extend([d] * len(X_day_selected))
            unique_cell_lat_week.extend(cell_lat_sel)
            unique_cell_lon_week.extend(cell_lon_sel)
            unique_coord_lat_week.extend(coord_lat_sel)
            unique_coord_lon_week.extend(coord_lon_sel)

        X_week_selected = np.array(X_week_list)  # shape: (week*n_cells, window_size, n_features)
        y_week_selected = np.array(y_week_list)  # shape: (week*n_cells,)

        # --- Step 4: Aggregate the selected windows ---
        X_agg_week = preprocess_windows(X_week_selected, scaler=scaler)
        self.X_agg_test_samples = X_agg_week  # shape: (week*n_cells, d_agg)
        self.y_test_samples = y_week_selected

        # Save metadata for CSV production
        self.test_sample_dates = np.array(test_sample_dates)
        self._unique_test_cell_lats = unique_cell_lat_week
        self._unique_test_cell_lons = unique_cell_lon_week
        self._unique_test_coord_lats = unique_coord_lat_week
        self._unique_test_coord_lons = unique_coord_lon_week

        # --- Step 5: Clear full-data caches ---
        self.X_full = None
        self.y_full = None
        self.sample_dates_full = None
        self.sample_cell_lats_full = None
        self.sample_cell_lons_full = None
        self.sample_coord_lats_full = None
        self.sample_coord_lons_full = None

    def produce_test_csv(self, probabilities: np.ndarray, out_csv_path: str):
        """
        Given an array of predicted probabilities for the test set (shape matching self.y_test_samples),
        produce a CSV with columns:
           [DATE, CELL_LAT, CELL_LON, COORDINATES_LAT, COORDINATES_LON, IS_FIRE_NEXT_DAY, IGNITION_LIKELIHOOD]
        using the metadata from the selected week.

        :param probabilities: np.ndarray of shape (window_size * n_cells,)
        :param out_csv_path: Path to save the resulting CSV.
        """
        if len(probabilities) != len(self.y_test_samples):
            raise ValueError("Length of probabilities must match number of test samples for the selected window.")

        # Now, self.test_sample_dates contains the date for each aggregated test sample.
        meta_df = pd.DataFrame({
            "DATE": self.test_sample_dates,
            "CELL_LAT": self._unique_test_cell_lats,
            "CELL_LON": self._unique_test_cell_lons,
            "COORDINATES_LAT": self._unique_test_coord_lats,
            "COORDINATES_LON": self._unique_test_coord_lons,
            "IS_FIRE_NEXT_DAY": self.y_test_samples,
            "IGNITION_LIKELIHOOD": probabilities
        })

        output_dir = os.path.dirname(out_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        meta_df.to_csv(out_csv_path, index=False)
        logger.info(f"Predictions CSV saved to {out_csv_path}")
        return meta_df

    def get_data(self, phase: str):
        """
        Returns X and y for the specified phase.
        :param phase: "train", "val", or "test"
        """
        if phase == "train":
            indices = self.train_indices
        elif phase == "val":
            indices = self.val_indices
        elif phase == "test":
            indices = self.test_indices
        else:
            raise ValueError("phase must be one of 'train', 'val', or 'test'.")
        return self.X[indices], self.y[indices]

    def get_dates(self, phase: str):
        """
        Returns a dictionary with the metadata (DATE, CELL_LAT, CELL_LON, COORDINATES_LAT, COORDINATES_LON)
        for the specified phase.
        :param phase: "train", "val", or "test"
        :return: dict with keys: DATE, CELL_LAT, CELL_LON, COORDINATES_LAT, COORDINATES_LON
        """
        if phase == "train":
            indices = self.train_indices
        elif phase == "val":
            indices = self.val_indices
        elif phase == "test":
            indices = self.test_indices
        else:
            raise ValueError("phase must be one of 'train', 'val', or 'test'.")
        return {
            "DATE": self.sample_dates[indices],
            "CELL_LAT": self.sample_cell_lats[indices],
            "CELL_LON": self.sample_cell_lons[indices],
            "COORDINATES_LAT": self.sample_coord_lats[indices],
            "COORDINATES_LON": self.sample_coord_lons[indices],
        }

    def get_aggregated_data(self, phase: str, scaler=None):
        """
        Returns aggregated features and y for the specified phase.
        :param phase: "train", "val", or "test"
        :param scaler: Optional StandardScaler. If not provided, standardizes per window.
        :return: (X_agg, y) where X_agg has shape (num_samples, d)
        """
        X_raw, y_phase = self.get_data(phase)
        X_agg = preprocess_windows(X_raw, scaler=scaler)
        return X_agg, y_phase

    def get_augmented_agg_data(self, augmentation_factor: int = 3):
        """
        Augments the training data

        :param augmentation_factor:
        :return: X_tran
        """
        X_agg_train, y_train = self.get_aggregated_data("train")
        X_agg_val, y_val = self.get_aggregated_data("val")
        X_agg_test, y_test = self.get_aggregated_data("test")
        X_agg_train_augmented, y_train_augmented = get_augmented_agg_data(X_agg_train, y_train,
                                                                          augmentation_factor)
        X_agg_val_augmented, y_val_augmented = get_augmented_agg_data(X_agg_val, y_val,
                                                                      augmentation_factor)
        X_agg_test_augmented, y_test_augmented = get_augmented_agg_data(X_agg_test, y_test,
                                                                        augmentation_factor)
        return (X_agg_train_augmented, y_train_augmented, X_agg_val_augmented, y_val_augmented,
                X_agg_test_augmented, y_test_augmented)

    def get_dataloaders(self, augmentation_factor: int = 3, train_b_size: Union[int, str] = "auto",
                        val_b_size: Union[int, str] = "auto", test_b_size: Union[int, str] = "auto",
                        max_b_size: int = 128, num_workers: Union[int, str] = "auto"):
        """
        Generates torch dataloaders for train, val, and test sets.

        :param augmentation_factor: Number of synthetic samples to generate per minority sample in training and
                                    validation.
        :param train_b_size: Batch size for the training dataloader.
        :param val_b_size: Batch size for the validation dataloader.
        :param test_b_size: Batch size for the test dataloader.
        :param max_b_size: The maximum batch size allowed, used when batch_size is set to "auto"
        :param num_workers: Num workers to be used
        :return: A tuple (train_loader, val_loader, test_loader)
        """
        # Retrieve augmented aggregated training data
        X_aug_train, y_aug_train, X_aug_val, y_aug_val, X_aug_test, y_aug_test = self.get_augmented_agg_data(
            augmentation_factor)
        # Retrieve validation and test data (aggregated features)

        # Convert numpy arrays to torch tensors (float32 for features, float32 for labels)
        X_aug_train = torch.tensor(X_aug_train).to(dtype=torch.float64)
        y_aug_train = torch.tensor(y_aug_train).to(dtype=torch.float64)

        X_aug_val = torch.tensor(X_aug_val).to(dtype=torch.float64)
        y_aug_val = torch.tensor(y_aug_val).to(dtype=torch.float64)

        X_aug_test = torch.tensor(X_aug_test).to(dtype=torch.float64)
        y_aug_test = torch.tensor(y_aug_test).to(dtype=torch.float64)

        # Create TensorDatasets
        if num_workers == "auto":
            num_workers = max(1, os.cpu_count() - 1)
        if train_b_size == "auto":
            train_b_size = get_auto_batch_size(y_aug_train.size(0), max_batch_size=max_b_size)
        if val_b_size == "auto":
            val_b_size = get_auto_batch_size(X_aug_val.size(0), max_batch_size=max_b_size)
        if test_b_size == "auto":
            test_b_size = get_auto_batch_size(y_aug_test.size(0), max_batch_size=max_b_size)

        train_dataset = TensorDataset(X_aug_train, y_aug_train)
        val_dataset = TensorDataset(X_aug_val, y_aug_val)
        test_dataset = TensorDataset(X_aug_test, y_aug_test)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=train_b_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=val_b_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=test_b_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def save(self, file_path: str):
        """
        Serializes the dataset object to disk.

        :param file_path: Path to the file where the object will be saved.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Dataset object saved to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """
        Loads a serialized WildfireWindowDataset object from disk.

        :param file_path: Path to the file where the object is saved.
        :return: An instance of WildfireWindowDataset.
        """
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)
        logger.info(f"Dataset object loaded from {file_path}")
        return dataset
