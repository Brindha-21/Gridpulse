"""
GridPulse AI - Data Preprocessor
Cleans raw electricity records, handles imputation, computes electrical balances,
and resamples time-series data to hourly frequency.
"""

import numpy as np
import pandas as pd


class ElectricityPreprocessor:
    """
    Standard preprocessing pipeline for Individual Household Electricity Consumption data.
    """

    def __init__(self, resample_freq="1h"):
        self.resample_freq = resample_freq.lower()

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans invalid readings, infills missing values using forward fill and interpolation.
        """
        df_clean = df.copy()

        # Ensure index is sorted datetime
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        df_clean = df_clean.sort_index()

        # Drop duplicate timestamps if any
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

        # Physical boundary clipping
        if "Global_active_power" in df_clean.columns:
            df_clean["Global_active_power"] = df_clean["Global_active_power"].clip(lower=0.0, upper=15.0)
        if "Global_reactive_power" in df_clean.columns:
            df_clean["Global_reactive_power"] = df_clean["Global_reactive_power"].clip(lower=0.0, upper=5.0)
        if "Voltage" in df_clean.columns:
            df_clean["Voltage"] = df_clean["Voltage"].clip(lower=180.0, upper=265.0)
        if "Global_intensity" in df_clean.columns:
            df_clean["Global_intensity"] = df_clean["Global_intensity"].clip(lower=0.0, upper=65.0)

        # Impute missing values: time-based interpolation bounded by forward/back fill
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = (
            df_clean[numeric_cols]
            .interpolate(method="time", limit=24)
            .bfill()
            .ffill()
        )

        return df_clean

    def resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples fine-grained readings (e.g., 1-minute) into hourly aggregated records.
        Averages power/voltage/current and aggregates energy metrics.
        """
        # Determine aggregation rule
        agg_rules = {
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum" if "Sub_metering_1" in df.columns else "mean",
            "Sub_metering_2": "sum" if "Sub_metering_2" in df.columns else "mean",
            "Sub_metering_3": "sum" if "Sub_metering_3" in df.columns else "mean",
        }
        
        # Only keep rules for existing columns
        active_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        
        # Resample to hourly frequency
        df_hourly = df.resample(self.resample_freq).agg(active_rules)
        df_hourly = df_hourly.interpolate(method="time").bfill().ffill()

        return df_hourly

    def compute_electrical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates electrical power engineering metrics:
        - Apparent Power S (kVA) = (Voltage * Current) / 1000
        - Power Factor = Active Power / Apparent Power
        - Unmetered Active Energy (Sub_metering_remainder)
        """
        df_feat = df.copy()

        # Apparent power & Power factor
        if "Voltage" in df_feat.columns and "Global_intensity" in df_feat.columns:
            df_feat["Apparent_power"] = (df_feat["Voltage"] * df_feat["Global_intensity"]) / 1000.0
            
            # Avoid division by zero
            denom = np.maximum(df_feat["Apparent_power"].values, 0.05)
            df_feat["Power_factor"] = np.clip(df_feat["Global_active_power"].values / denom, 0.50, 1.0)
        else:
            df_feat["Power_factor"] = 0.92

        # Sub-metering remainder (active energy consumed not tracked by kitchen/laundry/HVAC)
        # Note: If hourly, active energy is Global_active_power * 1000 Wh
        sub1 = df_feat.get("Sub_metering_1", 0.0)
        sub2 = df_feat.get("Sub_metering_2", 0.0)
        sub3 = df_feat.get("Sub_metering_3", 0.0)
        
        total_active_wh = df_feat["Global_active_power"] * 1000.0
        df_feat["Sub_metering_remainder"] = np.maximum(0.0, total_active_wh - (sub1 + sub2 + sub3))

        return df_feat

    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs complete cleaning, resampling, and electrical feature generation."""
        cleaned = self.clean_and_impute(df)
        
        # Check if already hourly
        median_delta = pd.Series(cleaned.index).diff().median()
        if median_delta is not None and median_delta < pd.Timedelta(minutes=50):
            resampled = self.resample_hourly(cleaned)
        else:
            resampled = cleaned

        processed = self.compute_electrical_features(resampled)
        return processed
