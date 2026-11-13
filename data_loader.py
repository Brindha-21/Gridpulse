"""
GridPulse AI - Data Ingestion & UCI Benchmark Generator
Handles loading, downloading from UCI ML Repository, and generating realistic benchmark electricity data.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_SAMPLE_PATH = os.path.join(DEFAULT_DATA_DIR, "sample_electricity_uci.csv")


def download_uci_dataset(dest_dir=None):
    """
    Downloads the official UCI Individual Household Electric Power Consumption dataset.
    Extracts the 'household_power_consumption.txt' archive.
    """
    if dest_dir is None:
        dest_dir = os.path.join(DEFAULT_DATA_DIR, "raw")
    os.makedirs(dest_dir, exist_ok=True)
    
    zip_path = os.path.join(dest_dir, "household_power_consumption.zip")
    txt_path = os.path.join(dest_dir, "household_power_consumption.txt")
    
    if os.path.exists(txt_path):
        return txt_path

    try:
        urllib.request.urlretrieve(UCI_DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        return txt_path
    except Exception as e:
        print(f"[Warning] Failed to download from UCI: {e}. Falling back to benchmark generation.")
        return None


def load_uci_data(filepath, nrows=None):
    """
    Parses the standard UCI text dataset formatted with ';' delimiters.
    Missing values '?' are replaced with NaN and timestamps are converted to DatetimeIndex.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    # Standard UCI columns
    col_names = [
        "Date", "Time", "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    
    # Check if delimiter is comma or semicolon
    with open(filepath, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    
    df = pd.read_csv(
        filepath,
        sep=sep,
        na_values=['?', 'NA', 'nan', ''],
        low_memory=False,
        nrows=nrows
    )

    # Standardize column naming
    df.columns = [c.strip() for c in df.columns]

    # Combine Date & Time if present
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )
        df.set_index("Datetime", inplace=True)
        df.drop(columns=["Date", "Time"], inplace=True)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    # Convert numeric fields
    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def generate_benchmark_uci_data(
    start_date="2023-01-01 00:00:00",
    end_date="2024-06-30 23:00:00",
    freq="1H",
    seed=42,
    save_path=None
):
    """
    Generates a physically authentic electricity consumption time series matching UCI specifications.
    
    Simulates:
    - Diurnal human activity profiles (morning rush 07:00-09:00, evening surge 18:00-22:00, night dip).
    - Day-of-week habits (weekdays vs weekends).
    - Weather/seasonal thermal heating/cooling dependencies.
    - Power systems physics:
        * Active power (kW), Reactive power (kW)
        * Grid Voltage (~230V +/- 3V fluctuation)
        * Current intensity I ~ (P * 1000) / (V * cos_phi)
        * Sub_metering_1: Kitchen appliances (dishwashers, microwaves, ovens)
        * Sub_metering_2: Laundry & refrigeration (washing machine, dryer, fridge)
        * Sub_metering_3: HVAC / electric water heater & AC
        * Sub_metering_remainder: Unmetered lighting, electronics, standby loads.
    - Sporadic extreme peak demand spikes (extreme heatwaves, concurrent peak loads).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)

    hours = timestamps.hour.values
    dayofweek = timestamps.dayofweek.values
    month = timestamps.month.values
    dayofyear = timestamps.dayofyear.values

    # 1. Base Diurnal Cycle (kW)
    # Double sinusoidal curve for morning and evening peaks
    morning_peak = np.exp(-((hours - 8) ** 2) / 6.0) * 1.6
    evening_peak = np.exp(-((hours - 20) ** 2) / 8.0) * 2.8
    overnight_baseline = 0.35 + 0.15 * np.cos(2 * np.pi * hours / 24.0)
    diurnal = morning_peak + evening_peak + overnight_baseline

    # 2. Weekend Shift
    # On weekends, peak demand starts later, spreads across the afternoon
    is_weekend = (dayofweek >= 5).astype(float)
    weekend_afternoon = is_weekend * (np.exp(-((hours - 14) ** 2) / 12.0) * 0.8 - morning_peak * 0.4)

    # 3. Seasonal Fluctuation (Summer AC peaks in June-August, Winter heating in Dec-Feb)
    seasonal_temp_effect = 0.4 * np.cos(2 * np.pi * (dayofyear - 20) / 365.25) + 0.3 * np.sin(2 * np.pi * (dayofyear - 180) / 365.25)
    
    # 4. Realistic Grid Voltage: Nominal 230V with slight inverse relationship to demand
    base_voltage = 238.0
    voltage_noise = rng.normal(0, 1.2, n)
    voltage = base_voltage - 1.5 * diurnal + voltage_noise

    # 5. Stochastic noise and random appliance activations
    appliance_bursts = rng.exponential(scale=0.18, size=n)
    
    # 6. Global Active Power (kW)
    global_active = diurnal + weekend_afternoon + seasonal_temp_effect + appliance_bursts
    global_active = np.clip(global_active, 0.12, 10.5)

    # Inject realistic severe peak events (e.g. heatwave, winter storm, major hosting party)
    peak_event_indices = rng.choice(n, size=int(n * 0.025), replace=False)
    global_active[peak_event_indices] += rng.uniform(2.5, 4.8, size=len(peak_event_indices))

    # 7. Global Reactive Power (kW) - typically 0.08 to 0.35 of active power with motor/inductive loads
    reactive_factor = 0.12 + 0.08 * rng.uniform(0.7, 1.3, size=n)
    global_reactive = np.clip(global_active * reactive_factor + rng.normal(0.04, 0.02, n), 0.02, 1.4)

    # 8. Global Intensity (Amperes): I = (P * 1000) / (V * power_factor)
    power_factor = 0.92 + rng.uniform(-0.04, 0.04, n)
    global_intensity = (global_active * 1000.0) / (voltage * power_factor)
    global_intensity = np.clip(global_intensity, 0.5, 45.0)

    # 9. Sub-meterings (in Watt-hours per hour or minute scaled)
    # Total active energy per hour in Wh:
    total_active_wh = global_active * 1000.0

    # Sub_metering_1 (Kitchen): active mainly around meal times (07:00-09:00, 12:00-13:30, 18:30-21:00)
    kitchen_prob = (
        0.3 * np.exp(-((hours - 8) ** 2) / 2.0) +
        0.2 * np.exp(-((hours - 13) ** 2) / 2.0) +
        0.5 * np.exp(-((hours - 20) ** 2) / 3.0)
    )
    sub1 = total_active_wh * (0.10 + 0.25 * kitchen_prob) * rng.uniform(0.8, 1.2, n)

    # Sub_metering_2 (Laundry Room): active predominantly on weekends and mornings
    laundry_prob = (0.4 * is_weekend + 0.1) * np.exp(-((hours - 11) ** 2) / 6.0)
    sub2 = total_active_wh * (0.05 + 0.30 * laundry_prob) * rng.uniform(0.7, 1.3, n)

    # Sub_metering_3 (HVAC, Water Heater): large share, active during temperature extremes & morning/evening
    hvac_load_ratio = 0.35 + 0.15 * np.abs(np.sin(2 * np.pi * dayofyear / 365.25))
    sub3 = total_active_wh * hvac_load_ratio * rng.uniform(0.85, 1.15, n)

    # Ensure submetering sum does not exceed total active energy
    sub_sum = sub1 + sub2 + sub3
    scale_factor = np.where(sub_sum > 0.88 * total_active_wh, (0.85 * total_active_wh) / np.maximum(sub_sum, 1e-5), 1.0)
    sub1 = np.round(np.maximum(sub1 * scale_factor, 0.0), 2)
    sub2 = np.round(np.maximum(sub2 * scale_factor, 0.0), 2)
    sub3 = np.round(np.maximum(sub3 * scale_factor, 0.0), 2)

    df = pd.DataFrame({
        "Datetime": timestamps,
        "Global_active_power": np.round(global_active, 4),
        "Global_reactive_power": np.round(global_reactive, 4),
        "Voltage": np.round(voltage, 2),
        "Global_intensity": np.round(global_intensity, 2),
        "Sub_metering_1": sub1,
        "Sub_metering_2": sub2,
        "Sub_metering_3": sub3,
    })
    df.set_index("Datetime", inplace=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Format matching UCI standard columns
        df_export = df.copy()
        df_export["Date"] = df_export.index.strftime("%d/%m/%Y")
        df_export["Time"] = df_export.index.strftime("%H:%M:%S")
        export_cols = ["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage", 
                       "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        df_export[export_cols].to_csv(save_path, sep=";", index=False)
        print(f"[Dataset] Benchmark electricity dataset saved successfully to {save_path} ({len(df)} records)")

    return df


def get_or_create_dataset(target_csv_path=None):
    """
    Returns a loaded DataFrame.
    If target_csv_path doesn't exist, generates and saves the default benchmark UCI dataset.
    """
    if target_csv_path is None:
        target_csv_path = DEFAULT_SAMPLE_PATH

    if os.path.exists(target_csv_path):
        return load_uci_data(target_csv_path)

    # If raw zip/txt exists, load it
    raw_txt = os.path.join(DEFAULT_DATA_DIR, "raw", "household_power_consumption.txt")
    if os.path.exists(raw_txt):
        return load_uci_data(raw_txt, nrows=50000)

    # Otherwise synthesize benchmark dataset
    print(f"[Data Loader] Initializing benchmark dataset at: {target_csv_path}")
    return generate_benchmark_uci_data(
        start_date="2023-01-01 00:00:00",
        end_date="2024-06-30 23:00:00",
        freq="1H",
        save_path=target_csv_path
    )


if __name__ == "__main__":
    df = get_or_create_dataset()
    print("Dataset loaded successfully. Shape:", df.shape)
    print("Head:\n", df.head())
