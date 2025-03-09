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
        self.test_sample_date = None  # the selected test sample date

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
        Prepares aggregated test samples from the full (unbalanced) dataset.
        Steps:
          1. Split the full unbalanced data into a test set using a date-based split (to avoid future leakage).
          2. Among these test samples, identify the test date with the maximum total fire flags (np.sum(y==1)).
          3. For that date, select one window per unique cell (based on CELL_LAT and CELL_LON).
          4. Aggregate the selected windows via preprocess_windows() to obtain X_agg_test_samples.
          5. Save the corresponding labels into y_test_samples and the chosen date in test_sample_date.
          6. Cache the unique cell metadata for later use (e.g., in produce_test_csv).
          7. Finally, clear the full-data caches.
        """
        # --- Step 1: Identify test dates from the full unbalanced data ---
        # Use unique dates (assumed sorted) from the full dataset
        unique_dates = np.unique(self.sample_dates_full)
        num_dates = len(unique_dates)
        # Determine number of test dates (at least one)
        num_test_dates = int(self.test_ratio * num_dates)
        if num_test_dates == 0:
            num_test_dates = 1
        # Choose the last num_test_dates as the test period
        test_dates = unique_dates[-num_test_dates:]

        # Filter full data to only include test dates
        mask_test = np.isin(self.sample_dates_full, test_dates)
        X_test_full = self.X_full[mask_test]
        y_test_full = self.y_full[mask_test]
        dates_test_full = self.sample_dates_full[mask_test]
        cell_lat_test_full = self.sample_cell_lats_full[mask_test]
        cell_lon_test_full = self.sample_cell_lons_full[mask_test]
        coord_lat_test_full = self.sample_coord_lats_full[mask_test]
        coord_lon_test_full = self.sample_coord_lons_full[mask_test]

        # --- Step 2: Select the test date with the maximum number of fire flags ---
        unique_test_dates = np.unique(dates_test_full)
        max_fire = -1
        best_date = None
        for d in unique_test_dates:
            fire_count = np.sum(y_test_full[dates_test_full == d])
            if fire_count > max_fire:
                max_fire = fire_count
                best_date = d
        self.test_sample_date = best_date

        # --- Step 3: For the best date, select one sample per unique cell ---
        mask_best = dates_test_full == best_date
        X_best = X_test_full[mask_best]
        y_best = y_test_full[mask_best]
        cell_lat_best = cell_lat_test_full[mask_best]
        cell_lon_best = cell_lon_test_full[mask_best]
        coord_lat_best = coord_lat_test_full[mask_best]
        coord_lon_best = coord_lon_test_full[mask_best]

        # Build dictionary to keep one sample per unique cell (using (CELL_LAT, CELL_LON) as identifier)
        unique_cells = {}
        X_best_selected = []
        y_best_selected = []
        unique_cell_lat = []
        unique_cell_lon = []
        unique_coord_lat = []
        unique_coord_lon = []
        for i, (lat, lon) in enumerate(zip(cell_lat_best, cell_lon_best)):
            cell_id = (lat, lon)
            if cell_id not in unique_cells:
                unique_cells[cell_id] = True
                X_best_selected.append(X_best[i])
                y_best_selected.append(y_best[i])
                unique_cell_lat.append(lat)
                unique_cell_lon.append(lon)
                unique_coord_lat.append(coord_lat_best[i])
                unique_coord_lon.append(coord_lon_best[i])

        X_best_selected = np.array(X_best_selected)
        y_best_selected = np.array(y_best_selected)

        # --- Step 4: Aggregate the selected windows ---
        X_agg_test = preprocess_windows(X_best_selected, scaler=scaler)
        self.X_agg_test_samples = X_agg_test
        self.y_test_samples = y_best_selected

        # Cache unique cell metadata for CSV production
        self._unique_test_cell_lats = unique_cell_lat
        self._unique_test_cell_lons = unique_cell_lon
        self._unique_test_coord_lats = unique_coord_lat
        self._unique_test_coord_lons = unique_coord_lon

        # --- Step 5: Free up memory by clearing full-data caches ---
        self.X_full = None
        self.y_full = None
        self.sample_dates_full = None
        self.sample_cell_lats_full = None
        self.sample_cell_lons_full = None
        self.sample_coord_lats_full = None
        self.sample_coord_lons_full = None

    def produce_test_csv(self, probabilities: np.ndarray, out_csv_path: str):
        """
        Given an array of predicted probabilities for the test set (shape = y_test_samples.shape),
        produce a CSV with columns:
           [DATE, CELL_LAT, CELL_LON, COORDINATES_LAT, COORDINATES_LON, IGNITION_LIKELIHOOD]
        using the metadata from the selected test sample date.
        :param probabilities: np.ndarray of shape (n_cells,)
        :param out_csv_path: Path to save the resulting CSV.
        """
        if len(probabilities) != len(self.y_test_samples):
            raise ValueError("Length of probabilities must match number of test samples for the selected date.")

        # To reconstruct metadata for the selected test samples, we must reapply the mask used during
        # _prepare_test_samples. For simplicity, assume that self.test_sample_date, and the associated cell
        # coordinates, are stored. (In a more complete implementation you might cache these metadata arrays as
        # attributes.) Here, we assume that the ordering of self.X_agg_test_samples matches that of the unique cell
        # identifiers. You might want to store a separate metadata dictionary during _prepare_test_samples. For
        # demonstration, we rebuild a DataFrame from the test sample selection:
        meta_df = pd.DataFrame({
            "DATE": [self.test_sample_date] * len(self.y_test_samples),
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
