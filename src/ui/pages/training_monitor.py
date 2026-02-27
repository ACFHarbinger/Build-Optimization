"""
Training Monitor page — visualize RL training metrics.
"""

import os
from typing import List

import plotly.graph_objects as go
import streamlit as st

from ui.services.log_parser import discover_training_runs, parse_training_log
from ui.styles.colors import CHART_COLORS
from ui.styles.kpi import create_kpi_row


def render_training_monitor() -> None:
    """Render the Training Monitor page."""
    st.title("📈 Training Monitor")

    runs = discover_training_runs()

    if not runs:
        st.info(
            "No training runs found. Start RL training to generate log files "
            "(`metrics.csv` or `training_log.jsonl` in `outputs/`)."
        )
        _render_demo_training()
        return

    # Run selector
    selected_runs = st.multiselect(
        "Select Runs",
        options=runs,
        default=runs[: min(2, len(runs))],
        format_func=lambda p: os.path.basename(p),
    )

    if not selected_runs:
        st.warning("Select at least one run.")
        return

    # Metric selector
    first_df = parse_training_log(selected_runs[0])
    if first_df.empty:
        st.error("Could not parse training log.")
        return

    numeric_cols = [c for c in first_df.columns if first_df[c].dtype in ("float64", "int64", "float32")]
    if not numeric_cols:
        st.error("No numeric columns found in training log.")
        return

    primary_metric = st.selectbox("Primary Metric", options=numeric_cols, index=0)

    # Smoothing
    smoothing = st.slider("Smoothing", min_value=1, max_value=50, value=1)

    _render_training_charts(selected_runs, primary_metric, smoothing)


def _render_demo_training() -> None:
    """Render demo training curves."""
    import numpy as np

    st.markdown("### Demo Training Curves")

    epochs = list(range(1, 51))
    loss = [2.0 * np.exp(-0.05 * e) + np.random.normal(0, 0.05) for e in epochs]
    reward = [50 * (1 - np.exp(-0.08 * e)) + np.random.normal(0, 2) for e in epochs]

    kpi = {
        "Epochs": 50,
        "Latest Loss": round(loss[-1], 4),
        "Best Reward": round(max(reward), 2),
    }
    st.markdown(create_kpi_row(kpi), unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=epochs, y=loss, mode="lines", name="Loss", line=dict(color=CHART_COLORS[3], width=2))
        )
        fig.update_layout(
            title="Training Loss",
            yaxis_title="Loss",
            xaxis_title="Epoch",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=epochs, y=reward, mode="lines", name="Reward", line=dict(color=CHART_COLORS[2], width=2))
        )
        fig2.update_layout(
            title="Reward",
            yaxis_title="Reward",
            xaxis_title="Epoch",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_training_charts(runs: List[str], metric: str, smoothing: int) -> None:
    """Render training metric charts for selected runs."""
    fig = go.Figure()

    all_kpi = {}

    for i, run_dir in enumerate(runs):
        df = parse_training_log(run_dir)
        if df.empty or metric not in df.columns:
            continue

        name = os.path.basename(run_dir)
        y = df[metric]

        if smoothing > 1:
            y = y.rolling(window=smoothing, min_periods=1).mean()

        x = df["epoch"] if "epoch" in df.columns else list(range(len(y)))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            )
        )

        all_kpi[f"{name}"] = round(float(y.iloc[-1]), 4) if len(y) > 0 else 0

    # KPI row
    if all_kpi:
        latest_vals = list(all_kpi.values())
        summary_kpi = {
            "Runs": len(runs),
            "Best": min(latest_vals) if "loss" in metric.lower() else max(latest_vals),
        }
        st.markdown(create_kpi_row(summary_kpi), unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    fig.update_layout(
        title=f"{metric} over Training",
        yaxis_title=metric,
        xaxis_title="Epoch",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
