"""
GridPulse AI - Data & Prediction Visualizer
Generates interactive Plotly visualizations for the Streamlit dashboard and
publication-grade Matplotlib figures for static reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional


# ==============================================================================
# Interactive Plotly Figures (For Streamlit Dashboard)
# ==============================================================================

def create_actual_vs_predicted_plot(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_mask: Optional[np.ndarray] = None,
    title: str = "Electricity Consumption: Actual vs. Hybrid ANN-RNN Forecast"
) -> go.Figure:
    """
    Renders an interactive time-series chart comparing actual vs predicted consumption
    with peak anomaly highlight markers and error ribbon.
    """
    df_plot = pd.DataFrame({
        "Timestamp": timestamps,
        "Actual": y_true,
        "Predicted": y_pred,
        "Error": y_true - y_pred
    })

    fig = go.Figure()

    # Actual Consumption Line
    fig.add_trace(go.Scatter(
        x=df_plot["Timestamp"],
        y=df_plot["Actual"],
        mode="lines",
        name="Actual Demand (kW)",
        line=dict(color="#00F2FE", width=2.2),
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Actual: %{y:.2f} kW<extra></extra>"
    ))

    # Hybrid Predicted Consumption Line
    fig.add_trace(go.Scatter(
        x=df_plot["Timestamp"],
        y=df_plot["Predicted"],
        mode="lines",
        name="Hybrid ANN-RNN Predicted",
        line=dict(color="#4FACFE", width=2.0, dash="dash"),
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Predicted: %{y:.2f} kW<extra></extra>"
    ))

    # Highlight Peak Demands
    if peak_mask is not None and np.any(peak_mask):
        peak_times = df_plot["Timestamp"][peak_mask]
        peak_vals = df_plot["Actual"][peak_mask]
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=peak_vals,
            mode="markers",
            name="Peak Demand Surge",
            marker=dict(color="#FF416C", size=9, symbol="triangle-up", line=dict(color="#FFF", width=1)),
            hovertemplate="<b>CRITICAL PEAK</b><br>%{x|%b %d, %H:%M}<br>Demand: %{y:.2f} kW<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#F3F4F6")),
        xaxis=dict(title="Time (Hourly)", showgrid=True, gridcolor="#1F2937", color="#9CA3AF"),
        yaxis=dict(title="Global Active Power (kW)", showgrid=True, gridcolor="#1F2937", color="#9CA3AF"),
        template="plotly_dark",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_daily_profile_plot(df: pd.DataFrame) -> go.Figure:
    """
    Visualizes the average diurnal (24-hour) demand curves for Weekdays vs Weekends.
    """
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy["day_type"] = np.where(df_copy.index.dayofweek >= 5, "Weekend", "Weekday")

    grouped = df_copy.groupby(["day_type", "hour"])["Global_active_power"].agg(
        mean="mean",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75)
    ).reset_index()

    fig = go.Figure()

    colors = {"Weekday": "#00F2FE", "Weekend": "#FFB86C"}

    for d_type in ["Weekday", "Weekend"]:
        sub = grouped[grouped["day_type"] == d_type]
        fig.add_trace(go.Scatter(
            x=sub["hour"],
            y=sub["mean"],
            mode="lines+markers",
            name=f"{d_type} (Mean)",
            line=dict(color=colors[d_type], width=2.5),
            marker=dict(size=6)
        ))
        # Upper and lower quartile band
        fig.add_trace(go.Scatter(
            x=list(sub["hour"]) + list(sub["hour"])[::-1],
            y=list(sub["p75"]) + list(sub["p25"])[::-1],
            fill="toself",
            fillcolor=colors[d_type].replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in colors[d_type] else "rgba(0, 242, 254, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=f"{d_type} IQR"
        ))

    # Add peak tariff background shading
    fig.add_vrect(
        x0=18, x1=21,
        annotation_text="Peak Tariff Window (18h-21h)",
        annotation_position="top left",
        fillcolor="rgba(255, 65, 108, 0.12)",
        line_width=1,
        line_color="rgba(255, 65, 108, 0.4)"
    )

    fig.update_layout(
        title=dict(text="Diurnal Electricity Demand Profile (Weekday vs. Weekend)", font=dict(size=16, color="#F3F4F6")),
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", tick0=0, dtick=2, gridcolor="#1F2937", color="#9CA3AF"),
        yaxis=dict(title="Average Active Power (kW)", gridcolor="#1F2937", color="#9CA3AF"),
        template="plotly_dark",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_weekly_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive heatmap showing demand intensity across 7 Days of the Week x 24 Hours.
    """
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy["day_name"] = df_copy.index.day_name()
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df_copy.pivot_table(
        index="day_name",
        columns="hour",
        values="Global_active_power",
        aggfunc="mean"
    ).reindex(day_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot.index,
        colorscale="Viridis",
        colorbar=dict(title="kW"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Mean Demand: %{z:.2f} kW<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Weekly Demand Pattern Heatmap (Day × Hour)", font=dict(size=16, color="#F3F4F6")),
        xaxis=dict(title="Hour of Day", color="#9CA3AF"),
        yaxis=dict(title="Day of Week", color="#9CA3AF"),
        template="plotly_dark",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19"
    )
    return fig


def create_submetering_breakdown_plot(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of sub-metering shares: Kitchen vs Laundry vs HVAC vs Remainder.
    """
    sub1_total = df.get("Sub_metering_1", pd.Series(0)).sum()
    sub2_total = df.get("Sub_metering_2", pd.Series(0)).sum()
    sub3_total = df.get("Sub_metering_3", pd.Series(0)).sum()
    rem_total = df.get("Sub_metering_remainder", pd.Series(0)).sum()

    labels = ["Kitchen (Sub 1)", "Laundry (Sub 2)", "HVAC & Heating (Sub 3)", "Unmetered Remainder"]
    values = [sub1_total, sub2_total, sub3_total, rem_total]
    colors = ["#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Consumption: %{value:,.0f} Wh (%{percent})<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="Cumulative Sub-Metering Energy Distribution", font=dict(size=16, color="#F3F4F6")),
        template="plotly_dark",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        showlegend=True
    )
    return fig


def create_model_comparison_bar(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Bar chart comparing models (ANN vs LSTM vs Hybrid) across MAE, RMSE, MAPE.
    """
    models = list(metrics_dict.keys())
    maes = [metrics_dict[m].get("MAE", 0) for m in models]
    rmses = [metrics_dict[m].get("RMSE", 0) for m in models]
    r2s = [metrics_dict[m].get("R2", 0) for m in models]

    fig = go.Figure(data=[
        go.Bar(name="MAE (kW)", x=models, y=maes, marker_color="#00F2FE"),
        go.Bar(name="RMSE (kW)", x=models, y=rmses, marker_color="#4FACFE"),
        go.Bar(name="R² Score", x=models, y=r2s, marker_color="#6BCB77")
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Performance Benchmarking (ANN vs. LSTM vs. Hybrid)", font=dict(size=16, color="#F3F4F6")),
        xaxis=dict(color="#9CA3AF"),
        yaxis=dict(title="Metric Value", gridcolor="#1F2937", color="#9CA3AF"),
        template="plotly_dark",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ==============================================================================
# Static Publication-Grade Matplotlib Visualizations
# ==============================================================================

def plot_actual_vs_predicted_static(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generates a publication-grade static figure using Matplotlib."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    ax.plot(timestamps, y_true, label="Actual Active Power (kW)", color="#00F2FE", linewidth=1.8, alpha=0.9)
    ax.plot(timestamps, y_pred, label="Hybrid Predicted (kW)", color="#FF8A00", linewidth=1.6, linestyle="--", alpha=0.9)

    ax.set_title("GridPulse AI: Actual vs Hybrid Forecasted Electricity Consumption", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Demand (kW)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=True, facecolor="#111827", edgecolor="#374151")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
