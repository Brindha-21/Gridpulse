"""
GridPulse AI - Training & Evaluation Pipeline Script
Executes complete data loading, preprocessing, feature engineering,
hybrid ANN-RNN training, peak detection, and artifact persistence.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_or_create_dataset
from src.preprocessor import ElectricityPreprocessor
from src.feature_engineering import prepare_feature_dataset, create_hybrid_dataset_arrays
from src.models_ann_rnn import GridPulseModelTrainer
from src.peak_detector import PeakDemandDetector
from src.visualizer import plot_actual_vs_predicted_static


def run_pipeline(
    data_path: str = None,
    epochs: int = 15,
    batch_size: int = 32,
    lookback: int = 24,
    save_plots: bool = True
):
    print("=" * 80)
    print("   ⚡ GRIDPULSE AI: HYBRID ANN-RNN TRAINING & EVALUATION PIPELINE ⚡   ")
    print("=" * 80)

    # 1. Ingestion
    print("\n[Step 1/6] Ingesting Electricity Consumption Dataset...")
    df_raw = get_or_create_dataset(data_path)
    print(f"  ✓ Loaded {len(df_raw):,} raw electricity records.")

    # 2. Preprocessing & Physical Balances
    print("\n[Step 2/6] Running Data Cleaning, Hourly Resampling & Power Calculations...")
    preprocessor = ElectricityPreprocessor(resample_freq="1h")
    df_processed = preprocessor.process_pipeline(df_raw)
    print(f"  ✓ Resampled to {len(df_processed):,} hourly intervals.")
    print(f"  ✓ Columns: {list(df_processed.columns)}")

    # 3. Feature Engineering
    print("\n[Step 3/6] Engineering Cyclic Embeddings, Autoregressive Lags & Rolling Moments...")
    df_features, feature_cols = prepare_feature_dataset(df_processed, target_col="Global_active_power")
    print(f"  ✓ Generated {len(feature_cols)} predictor features.")
    print(f"  ✓ Clean aligned dataset size: {len(df_features):,} rows.")

    # 4. Hybrid Formatting
    print("\n[Step 4/6] Structuring Dual ANN-RNN Input Tensors (Lookback Window = 24h)...")
    X_ann, X_rnn, y, timestamps = create_hybrid_dataset_arrays(
        df_features, feature_cols, target_col="target_next_hour", lookback_window=lookback
    )
    print(f"  ✓ ANN Input Tensor Shape:  {X_ann.shape} (Tabular/Lags)")
    print(f"  ✓ RNN Input Tensor Shape:  {X_rnn.shape} (Temporal Sequence)")
    print(f"  ✓ Target Output Shape:     {y.shape} (Next-Hour kW)")

    # 5. Deep Learning Model Training
    print("\n[Step 5/6] Training ANN, LSTM, and Hybrid ANN-RNN Models...")
    trainer = GridPulseModelTrainer()
    train_output = trainer.fit_and_evaluate(
        X_ann=X_ann,
        X_rnn=X_rnn,
        y=y,
        train_ratio=0.8,
        epochs=epochs,
        batch_size=batch_size
    )

    metrics_summary = train_output["metrics_summary"]
    print("\n" + "-" * 70)
    print(f"{'Model Architecture':<22} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'MAPE (%)':<10} | {'R²':<8}")
    print("-" * 70)
    for model_name, m in metrics_summary.items():
        print(f"{model_name:<22} | {m['MAE']:<10.4f} | {m['RMSE']:<10.4f} | {m['MAPE']:<10.2f} | {m['R2']:<8.4f}")
    print("-" * 70)

    # 6. Peak Demand Detection on Test Evaluation Set
    print("\n[Step 6/6] Running Peak-Demand Anomaly Detection on Test Period...")
    test_split = train_output["split_idx"]
    test_timestamps = timestamps[test_split:]
    y_test = train_output["y_test"]
    y_pred_hybrid = train_output["results"]["Hybrid_ANN_RNN"]["predictions"]

    detector = PeakDemandDetector(z_threshold=2.0, quantile_threshold=0.90, absolute_peak_kw=3.5)
    test_series = pd.Series(y_test, index=test_timestamps)
    peaks_df = detector.detect_peaks_statistical(test_series)

    num_peaks = int(peaks_df["is_peak"].sum())
    peak_pct = (num_peaks / len(peaks_df)) * 100.0
    print(f"  ✓ Evaluated {len(peaks_df)} test hours.")
    print(f"  ✓ Detected {num_peaks} peak-demand surges ({peak_pct:.1f}% of timeline).")

    # Save visualization
    if save_plots:
        plots_dir = os.path.join(trainer.models_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "actual_vs_predicted_hybrid.png")
        # Plot last 168 hours (1 full week) for crisp detail
        recent_n = min(168, len(test_timestamps))
        plot_actual_vs_predicted_static(
            test_timestamps[-recent_n:],
            y_test[-recent_n:],
            y_pred_hybrid[-recent_n:],
            save_path=plot_path
        )
        print(f"  ✓ High-resolution evaluation plot exported to: {plot_path}")

    print("\n================================================================================")
    print("  🎉 GRIDPULSE AI PIPELINE EXECUTION COMPLETE! READY FOR DASHBOARD LAUNCH 🎉   ")
    print("================================================================================\n")
    return train_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridPulse AI Hybrid ANN-RNN Training Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw or processed electricity CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch training size")
    parser.add_argument("--lookback", type=int, default=24, help="LSTM sequence lookback window (hours)")
    parser.add_argument("--no-plots", action="store_true", help="Disable static plot export")

    args = parser.parse_args()
    run_pipeline(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lookback=args.lookback,
        save_plots=not args.no_plots
    )
