"""
Solver Comparison page — compare multiple solver results side by side.
"""

import os
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from ui.services.data_loader import discover_solver_results, load_solver_result
from ui.styles.colors import CHART_COLORS
from ui.styles.kpi import create_kpi_row


def render_solver_comparison() -> None:
    """Render the Solver Comparison page."""
    st.title("🏆 Solver Comparison")

    result_files = discover_solver_results()

    if not result_files:
        st.info("No solver results found in `outputs/`. Run solvers to generate results.")
        _render_demo_comparison()
        return

    # Multi-select
    selected = st.multiselect(
        "Select Results to Compare",
        options=result_files,
        default=result_files[: min(4, len(result_files))],
        format_func=lambda p: os.path.basename(p),
    )

    if not selected:
        st.warning("Select at least one result file to compare.")
        return

    results = []
    for path in selected:
        data = load_solver_result(path)
        data["_filename"] = os.path.basename(path)
        results.append(data)

    _render_comparison(results)


def _render_demo_comparison() -> None:
    """Render demo comparison data."""
    demo_results = [
        {"solver": "Simulated Annealing", "score": 342.5, "cost": 4200, "items_count": 5, "time_s": 1.2},
        {"solver": "Genetic Algorithm", "score": 338.1, "cost": 4350, "items_count": 6, "time_s": 3.8},
        {"solver": "ALNS", "score": 351.0, "cost": 4100, "items_count": 5, "time_s": 2.1},
        {"solver": "Random", "score": 210.7, "cost": 3900, "items_count": 4, "time_s": 0.01},
    ]
    st.markdown("### Demo Comparison")
    _render_comparison(demo_results)


def _render_comparison(results: List[Dict[str, Any]]) -> None:
    """Render comparison charts and table."""
    # KPI summary
    scores = [r.get("score", 0) for r in results]
    best_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    kpi = {
        "Solvers": len(results),
        "Best Score": best_score,
        "Avg Score": avg_score,
    }
    st.markdown(create_kpi_row(kpi), unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Bar chart: score comparison
    col1, col2 = st.columns(2)

    solver_names = [r.get("solver", r.get("_filename", f"Run {i}")) for i, r in enumerate(results)]

    with col1:
        st.subheader("📊 Score Comparison")
        fig = go.Figure(
            data=go.Bar(
                x=solver_names,
                y=scores,
                marker_color=CHART_COLORS[: len(results)],
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
            )
        )
        fig.update_layout(
            yaxis_title="Score",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Cost Comparison")
        costs = [r.get("cost", 0) for r in results]
        fig2 = go.Figure(
            data=go.Bar(
                x=solver_names,
                y=costs,
                marker_color=CHART_COLORS[: len(results)],
                text=[f"{c:,.0f}" for c in costs],
                textposition="outside",
            )
        )
        fig2.update_layout(
            yaxis_title="Cost (gold)",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Results table
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📋 Detailed Results")

    import pandas as pd

    rows = []
    for r in results:
        rows.append(
            {
                "Solver": r.get("solver", "?"),
                "Score": r.get("score", 0),
                "Cost": r.get("cost", 0),
                "Items": r.get("items_count", len(r.get("items", []))),
                "Time (s)": r.get("time_s", "—"),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
