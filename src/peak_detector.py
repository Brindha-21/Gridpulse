"""
GridPulse AI - Peak-Demand Detection & Anomaly System
Implements multi-tiered peak detection: dynamic rolling z-scores, quantile-based
capacity thresholds, time-of-use (TOU) tariff risk alerting, and demand-response advisory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class PeakDemandDetector:
    """
    Intelligent peak-demand analyzer for electricity consumption time series.
    """

    def __init__(
        self,
        z_threshold: float = 2.0,
        quantile_threshold: float = 0.90,
        absolute_peak_kw: float = 3.5
    ):
        self.z_threshold = z_threshold
        self.quantile_threshold = quantile_threshold
        self.absolute_peak_kw = absolute_peak_kw

    def detect_peaks_statistical(
        self,
        series: pd.Series,
        window: int = 24
    ) -> pd.DataFrame:
        """
        Detects demand spikes using rolling baseline moments:
        Z-Score = (Demand - Rolling_Mean) / (Rolling_Std + eps)
        """
        rolling_mean = series.rolling(window=window, min_periods=6).mean()
        rolling_std = series.rolling(window=window, min_periods=6).std().fillna(0.1)

        z_scores = (series - rolling_mean) / np.maximum(rolling_std, 0.05)
        is_z_peak = z_scores > self.z_threshold

        # Quantile threshold based on series history
        q_val = series.quantile(self.quantile_threshold)
        is_q_peak = series >= q_val

        # Absolute capacity breach
        is_abs_peak = series >= self.absolute_peak_kw

        # Composite peak flag
        is_peak = is_z_peak | is_q_peak | is_abs_peak

        results_df = pd.DataFrame({
            "consumption": series,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "z_score": z_scores,
            "is_z_peak": is_z_peak,
            "is_q_peak": is_q_peak,
            "is_abs_peak": is_abs_peak,
            "is_peak": is_peak,
            "threshold_quantile": q_val
        }, index=series.index)

        return results_df

    def classify_demand_level(self, value: float, q75: float, q90: float, q98: float) -> str:
        """
        Categorizes an instantaneous or predicted demand value into 4 operational tiers.
        """
        if value >= q98 or value >= self.absolute_peak_kw:
            return "CRITICAL_PEAK"
        elif value >= q90:
            return "HIGH_DEMAND"
        elif value >= q75:
            return "ELEVATED"
        else:
            return "NORMAL"

    def generate_peak_advisory(
        self,
        predicted_kw: float,
        current_hour: int,
        submeter_breakdown: Dict[str, float] = None,
        baseline_q90: float = 2.8
    ) -> Dict[str, Any]:
        """
        Generates actionable demand-response alerts and peak-shifting advice for the next hour.
        """
        is_tou_peak = 18 <= current_hour <= 21
        is_demand_peak = predicted_kw >= baseline_q90

        if is_demand_peak and is_tou_peak:
            status = "CRITICAL PEAK - HIGH TARIFF"
            severity = "danger"
            message = (
                f"Next hour forecast ({predicted_kw:.2f} kW) breaches peak threshold during high-tariff TOU window "
                f"({current_hour:02d}:00). Immediate load reduction advised."
            )
        elif is_demand_peak:
            status = "HIGH DEMAND PEAK"
            severity = "warning"
            message = (
                f"Next hour forecast ({predicted_kw:.2f} kW) exceeds 90th percentile threshold. "
                "Consider staggering heavy appliance usage."
            )
        elif is_tou_peak:
            status = "PEAK TARIFF WINDOW"
            severity = "info"
            message = f"Time-of-Use peak window active ({current_hour:02d}:00). Grid rates are elevated."
        else:
            status = "OPTIMAL / OFF-PEAK"
            severity = "success"
            message = f"Forecasted demand ({predicted_kw:.2f} kW) is within normal baseline limits."

        # Appliance attribution & shift suggestions
        actions = []
        if submeter_breakdown:
            top_submeter = max(submeter_breakdown.items(), key=lambda item: item[1])
            if top_submeter[0] == "Sub_metering_3" and is_demand_peak:
                actions.append("Adjust HVAC thermostat by +1°C to +2°C to shave up to 15-20% peak load.")
            elif top_submeter[0] == "Sub_metering_2" and is_demand_peak:
                actions.append("Defer laundry washer/dryer cycles to off-peak hours (after 22:00).")
            elif top_submeter[0] == "Sub_metering_1" and is_demand_peak:
                actions.append("Stagger electric oven and high-draw kitchen equipment.")

        if not actions and is_demand_peak:
            actions.append("Postpone EV charging and water heating until overnight baseline window.")

        return {
            "status": status,
            "severity": severity,
            "predicted_kw": round(predicted_kw, 3),
            "hour": current_hour,
            "is_tou_peak": is_tou_peak,
            "message": message,
            "recommended_actions": actions
        }
