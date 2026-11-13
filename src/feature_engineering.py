"""
GridPulse AI - Feature Engineering
Generates cyclic calendar embeddings, autoregressive lags, rolling statistics,
and formats sliding lookback sequences for hybrid ANN and LSTM models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclic trigonometric representations of time alongside calendar flags.
    """
    df_out = df.copy()
    idx = df_out.index

    # Raw temporal units
    hours = idx.hour
    dayofweek = idx.dayofweek
    month = idx.month

    # Cyclic Hour of Day (period = 24)
    df_out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Cyclic Day of Week (period = 7)
    df_out["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df_out["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    # Cyclic Month (period = 12)
    df_out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Calendar Indicators
    df_out["is_weekend"] = (dayofweek >= 5).astype(int)
    df_out["is_peak_window"] = ((hours >= 18) & (hours <= 21)).astype(int)

    return df_out


def build_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    lags: List[int] = [1, 2, 3, 24, 48, 168],
    windows: List[int] = [6, 24]
) -> pd.DataFrame:
    """
    Constructs autoregressive lag values and rolling statistical moments.
    """
    df_out = df.copy()

    # Autoregressive lags
    for lag in lags:
        df_out[f"{target_col}_lag_{lag}"] = df_out[target_col].shift(lag)

    # Rolling statistics (computed over shifted series to prevent target leakage)
    shifted = df_out[target_col].shift(1)
    for w in windows:
        df_out[f"{target_col}_roll_mean_{w}h"] = shifted.rolling(window=w).mean()
        df_out[f"{target_col}_roll_std_{w}h"] = shifted.rolling(window=w).std()
        df_out[f"{target_col}_roll_max_{w}h"] = shifted.rolling(window=w).max()
        df_out[f"{target_col}_roll_min_{w}h"] = shifted.rolling(window=w).min()

    # Target: Next-Hour Demand (y = Global_active_power at t + 1)
    df_out["target_next_hour"] = df_out[target_col].shift(-1)

    return df_out


def prepare_feature_dataset(
    df: pd.DataFrame,
    target_col: str = "Global_active_power"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combines all engineered features and drops NaN rows caused by shifting.
    Returns the processed DataFrame and the list of feature column names.
    """
    df_features = build_time_features(df)
    df_features = build_lag_and_rolling_features(df_features, target_col=target_col)

    # Drop unneeded or future-leaking raw columns
    exclude_cols = ["target_next_hour"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    # Clean rows with NaN from lags/shifts
    clean_df = df_features.dropna().copy()
    return clean_df, feature_cols


def create_hybrid_dataset_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_next_hour",
    lookback_window: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Formats dataset into dual inputs for the Hybrid ANN-RNN architecture:
    - X_ann: 2D array of static & lag features at timestep t (samples, n_features)
    - X_rnn: 3D sequence array over past [t - lookback_window + 1 : t] (samples, lookback_window, n_features)
    - y: 1D array of target next-hour electricity consumption (samples,)
    - timestamps: DatetimeIndex of the corresponding forecast points
    """
    feature_values = df[feature_cols].values
    target_values = df[target_col].values
    timestamps = df.index

    n_samples = len(df) - lookback_window + 1
    if n_samples <= 0:
        raise ValueError(f"Dataset has {len(df)} rows, which is smaller than lookback_window={lookback_window}")

    n_features = len(feature_cols)
    X_rnn = np.zeros((n_samples, lookback_window, n_features), dtype=np.float32)
    X_ann = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    valid_timestamps = []

    for i in range(n_samples):
        start_idx = i
        end_idx = i + lookback_window
        # 3D Sequence for RNN (LSTM)
        X_rnn[i] = feature_values[start_idx:end_idx]
        # 2D Vector for ANN (current latest timestep)
        X_ann[i] = feature_values[end_idx - 1]
        # Target next-hour consumption
        y[i] = target_values[end_idx - 1]
        valid_timestamps.append(timestamps[end_idx - 1])

    return X_ann, X_rnn, y, pd.DatetimeIndex(valid_timestamps)
