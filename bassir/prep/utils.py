import pandas as pd
import numpy as np
import torch
import torch.fft
import logging
import os

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_toy_wildfire_csv_unbalanced(output_path, unbalanced_factor=10,
                                         start_date="2020-01-01", end_date="2024-31-12"):
    """
    Generates a toy CSV file with the following columns:
      DATE, CELL_LAT, CELL_LON, D2M, T2M, DEPTHBELOWLANDLAYER, STL1, U10, V10, SP,
      LAI_HV, LAI_LV, CVH, COORDINATES_LAT, COORDINATES_LON, AREA_HA, RH, VPD,
      SIZE_HA, BURNED_DENSITY, IS_FIRE, TP, IS_FIRE_NEXT_DAY

    Data is generated for daily dates from 2025-01-01 to 2025-03-01.
    The imbalance is controlled via unbalanced_factor: if unbalanced_factor=10,
    approximately 10 times as many rows will have IS_FIRE_NEXT_DAY = 0 as those with 1.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Define sample cell coordinates (you can add more if needed)
    cell_coords = [
        (5.25, -75.25),
        (5.26, -75.24),
        (5.27, -75.27),
        (5.28, -75.30),
        (5.29, -75.40),
        (6.27, -71.27),
        (7.28, -72.30),
        (8.29, -70.40),
    ]

    # List of columns as specified
    cols = ['DATE', 'CELL_LAT', 'CELL_LON', 'D2M', 'T2M', 'DEPTHBELOWLANDLAYER', 'STL1',
            'U10', 'V10', 'SP', 'LAI_HV', 'LAI_LV', 'CVH', 'COORDINATES_LAT', 'COORDINATES_LON',
            'AREA_HA', 'RH', 'VPD', 'SIZE_HA', 'BURNED_DENSITY', 'IS_FIRE', 'TP', 'IS_FIRE_NEXT_DAY']

    rows = []
    # Determine probability for label 1:
    # unbalanced_factor = 10 means roughly 1 fire (1) for every 10 no fire (0)
    p_fire = 1 / (1 + unbalanced_factor)

    for date in date_range:
        for (lat, lon) in cell_coords:
            row = {}
            row["DATE"] = date.strftime("%Y-%m-%d")
            row["CELL_LAT"] = lat
            row["CELL_LON"] = lon
            # For COORDINATES_LAT and COORDINATES_LON, use the same as cell coordinates
            row["COORDINATES_LAT"] = lat
            row["COORDINATES_LON"] = lon
            # Generate random values for other features (adjust ranges as needed)
            row["D2M"] = np.random.uniform(0, 30)
            row["T2M"] = np.random.uniform(0, 40)
            row["DEPTHBELOWLANDLAYER"] = np.random.uniform(0, 3)
            row["STL1"] = np.random.uniform(0, 1)
            row["U10"] = np.random.uniform(-5, 5)
            row["V10"] = np.random.uniform(-5, 5)
            row["SP"] = np.random.uniform(900, 1100)
            row["LAI_HV"] = np.random.uniform(0, 5)
            row["LAI_LV"] = np.random.uniform(0, 5)
            row["CVH"] = np.random.uniform(0, 1)
            row["AREA_HA"] = np.random.randint(1, 100)
            row["RH"] = np.random.uniform(20, 100)
            row["VPD"] = np.random.uniform(0, 5)
            row["SIZE_HA"] = np.random.randint(0, 50)
            row["BURNED_DENSITY"] = np.random.uniform(0, 5)
            row["IS_FIRE"] = np.random.randint(0, 2)
            row["TP"] = np.random.uniform(0, 20)
            # Set IS_FIRE_NEXT_DAY based on the desired imbalance
            row["IS_FIRE_NEXT_DAY"] = 1 if np.random.rand() < p_fire else 0

            rows.append(row)

    # Build the DataFrame
    df = pd.DataFrame(rows, columns=cols)
    # Sort to mimic typical ordering: by cell then date
    df.sort_values(by=["CELL_LAT", "CELL_LON", "DATE"], inplace=True)

    # Ensure parent directories exist before saving
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Toy unbalanced CSV saved to: {output_path}")


# Preprocessor utility functions using torch for vectorization
def preprocess_windows(X: np.ndarray, scaler=None) -> np.ndarray:
    """
    Vectorized preprocessor using torch for a numpy array X of shape (num_samples, window_size, dim).
    Returns an array of shape (num_samples, d), where d is the aggregated feature dimension.
    """
    num_samples = X.shape[0]
    # Convert entire array to torch tensor
    X_tensor = torch.tensor(X).to(DEVICE)  # shape (num_samples, w, dim)

    # Standardize per window if no global scaler is provided
    if scaler is None:
        means = X_tensor.mean(dim=1, keepdim=True)
        stds = X_tensor.std(dim=1, keepdim=True)
        stds[stds == 0] = 1
        X_std = (X_tensor - means) / stds
    else:
        # Otherwise, process each window individually (or convert scaler to tensor if possible)
        X_np = X  # keep original numpy array for now
        X_std = torch.stack([torch.tensor(scaler.transform(win)) for win in X_np])

    # Now, for each window in X_std, compute aggregated features using vectorized operations
    # We will apply our feature extractors along dimension 1 (the window dimension)
    # Note: Some operations (like skew and kurtosis) are not natively available in torch,
    # so we stick to our vectorized implementations where possible.

    # Statistical summaries:
    stat_means = X_std.mean(dim=1)  # shape (num_samples, dim)
    stat_stds = X_std.std(dim=1)  # shape (num_samples, dim)
    stat_mins = X_std.min(dim=1).values  # shape (num_samples, dim)
    stat_maxs = X_std.max(dim=1).values  # shape (num_samples, dim)
    stat_feats = torch.cat([stat_means, stat_stds, stat_mins, stat_maxs], dim=1)  # shape (num_samples, 4*dim)

    # Frequency domain features:
    fft_vals = torch.abs(torch.fft.rfft(X_std, dim=1))  # shape (num_samples, w_fft, dim)
    fft_vals_nodc = fft_vals[:, 1:, :]  # shape (num_samples, w_fft-1, dim)

    # If the entire X_std is zero, then the window is constant.
    if torch.allclose(X_std, torch.zeros_like(X_std)):
        dom_freq_idx = torch.zeros(X_std.shape[0], X_std.shape[2], device=X_std.device)
    else:
        dom_freq_idx = fft_vals_nodc.argmax(dim=1).float() + 1.0
    energy = fft_vals_nodc.sum(dim=1)  # shape (num_samples, dim)
    freq_feats = torch.cat([dom_freq_idx, energy], dim=1)  # shape (num_samples, 2*dim)

    # Trend features (slope) using a vectorized approach:
    num_samples, w, dim_ = X_std.shape
    t = torch.arange(w, dtype=X_std.dtype, device=X_std.device).unsqueeze(0).expand(num_samples, w)
    t_mean = t.mean(dim=1, keepdim=True)
    x_mean = X_std.mean(dim=1, keepdim=True)
    numerator = ((t - t_mean).unsqueeze(2) * (X_std - x_mean)).sum(dim=1)  # shape (num_samples, dim)
    denominator = ((t - t_mean) ** 2).sum(dim=1).unsqueeze(1)  # shape (num_samples, 1)
    trend_feats = numerator / denominator  # shape (num_samples, dim)

    # Seasonal amplitude: using vectorized approach
    trend = trend_feats.unsqueeze(1) * t.unsqueeze(2) + x_mean  # shape (num_samples, w, dim)
    detrended = X_std - trend
    seasonal_min, _ = detrended.min(dim=1)  # shape (num_samples, dim)
    seasonal_max, _ = detrended.max(dim=1)  # shape (num_samples, dim)
    seasonal_feats = seasonal_max - seasonal_min  # shape (num_samples, dim)

    # Concatenate all features along dim=1
    aggregated_feats = torch.cat([stat_feats, freq_feats, trend_feats, seasonal_feats], dim=1)
    return aggregated_feats.cpu().numpy()


def get_augmented_agg_data(X_agg_train: np.array, y_train: np.array, augmentation_factor: int = 3):
    """
    Augments the training data using a SMOTE-style interpolation on the aggregated features.

    :param X_agg_train: input array of shape (n_train, d_agg)
    :param y_train: label array of shape (n_train,)
    :param augmentation_factor: The number of synthetic samples to generate per minority sample.
    :return: (X_augmented, y_augmented) where:
             - X_augmented is a numpy array of shape (n_train + n_min * augmentation_factor, d_agg)
             - y_augmented is a numpy array of shape (n_train + n_min * augmentation_factor,)
    """
    # Convert data to torch tensors (using float32 for consistency)
    X = torch.tensor(X_agg_train, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    # Identify minority samples (label == 1)
    minority_mask = (y == 1)
    X_min = X[minority_mask]  # shape: (n_min, d_agg)
    n_min = X_min.shape[0]

    # SMOTE requires at least 2 minority samples. Otherwise, just return the original data.
    if n_min < 2:
        return X_agg_train, y_train

    # Set number of nearest neighbors for SMOTE (cannot exceed n_min-1)
    k = min(5, n_min - 1)

    # Compute pairwise Euclidean distances between minority samples
    distances = torch.cdist(X_min, X_min, p=2)  # shape: (n_min, n_min)
    # Sort distances to get nearest neighbors; the first column is the self-distance (0)
    _, sorted_indices = torch.sort(distances, dim=1)
    # Exclude self by taking indices 1 to k+1 for each minority sample
    neighbors = sorted_indices[:, 1:k + 1]  # shape: (n_min, k)

    # For each minority sample, generate augmentation_factor synthetic samples.
    X_min_expanded = X_min.repeat_interleave(augmentation_factor, dim=0)  # shape: (n_min * augmentation_factor, d_agg)
    # For each synthetic sample, determine its originating minority sample index.
    original_indices = torch.arange(n_min, device=DEVICE).repeat_interleave(augmentation_factor)
    # Randomly select one neighbor from the k nearest neighbors for each sample.
    rand_idx = torch.randint(low=0, high=k, size=(n_min * augmentation_factor,), device=DEVICE)
    neighbor_indices = neighbors[original_indices, rand_idx]  # shape: (n_min * augmentation_factor,)
    X_neighbors = X_min[neighbor_indices]  # shape: (n_min * augmentation_factor, d_agg)

    # Generate random interpolation coefficients in [0, 1)
    lam = torch.rand(n_min * augmentation_factor, device=DEVICE, dtype=torch.float32).unsqueeze(
        1)  # shape: (n_min * augmentation_factor, 1)
    # Create synthetic samples via linear interpolation
    X_synth = X_min_expanded + lam * (X_neighbors - X_min_expanded)

    # Concatenate synthetic samples with original training data
    X_augmented = torch.cat([X, X_synth], dim=0)
    # For synthetic samples, assign label 1
    y_synth = torch.ones(n_min * augmentation_factor, device=DEVICE, dtype=torch.float32)
    y_augmented = torch.cat([y, y_synth], dim=0)

    # Convert back to numpy arrays
    X_aug_np = X_augmented.cpu().numpy()
    y_aug_np = y_augmented.cpu().numpy()

    return X_aug_np, y_aug_np

