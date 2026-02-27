"""
Build Explorer page — visualize a single build with stats, synergies, and score.
"""

import os
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st

from ui.services.data_loader import discover_solver_results, load_solver_result
from ui.styles.colors import RARITY_COLORS
from ui.styles.kpi import create_kpi_row


def render_build_explorer() -> None:
    """Render the Build Explorer page."""
    st.title("🛡️ Build Explorer")

    result_files = discover_solver_results()

    if not result_files:
        st.info("No build results found. Run a solver first to generate output files in `outputs/`.")
        _render_demo_build()
        return

    # File selector
    selected = st.selectbox(
        "Select Result File",
        options=result_files,
        format_func=lambda p: os.path.basename(p),
    )

    if selected:
        data = load_solver_result(selected)
        _render_build_detail(data)


def _render_demo_build() -> None:
    """Render a demo build for illustration when no results exist."""
    st.markdown("### Demo Build")

    demo = {
        "solver": "demo",
        "score": 342.5,
        "cost": 4200,
        "budget": 5000,
        "items": [
            {
                "name": "Flame Sword",
                "slot": "WEAPON",
                "rarity": "EPIC",
                "cost": 1200,
                "stats": {"attack": 85, "critical_rate": 12},
            },
            {
                "name": "Dragon Helm",
                "slot": "HELMET",
                "rarity": "RARE",
                "cost": 800,
                "stats": {"defense": 45, "health": 60},
            },
            {
                "name": "Shadow Vest",
                "slot": "CHEST",
                "rarity": "LEGENDARY",
                "cost": 1500,
                "stats": {"defense": 70, "speed": 25},
            },
            {
                "name": "Swift Boots",
                "slot": "BOOTS",
                "rarity": "UNCOMMON",
                "cost": 400,
                "stats": {"speed": 40, "defense": 15},
            },
            {
                "name": "Ruby Ring",
                "slot": "RING_1",
                "rarity": "RARE",
                "cost": 300,
                "stats": {"critical_damage": 30, "attack": 10},
            },
        ],
        "synergies": ["Fire Mastery (2pc)", "Shadow Set (1pc)"],
    }
    _render_build_detail(demo)


def _render_build_detail(data: Dict[str, Any]) -> None:
    """Render full build detail from a result dict."""
    # KPI row
    score = data.get("score", 0)
    cost = data.get("cost", 0)
    budget = data.get("budget", 0)
    n_items = len(data.get("items", []))
    n_synergies = len(data.get("synergies", []))

    kpi = {
        "Score": score,
        "Cost": cost,
        "Budget Left": max(budget - cost, 0),
        "Items": n_items,
        "Synergies": n_synergies,
    }
    st.markdown(create_kpi_row(kpi), unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Two-column layout: items table + stat radar
    col1, col2 = st.columns([3, 2])

    items = data.get("items", [])

    with col1:
        st.subheader("📦 Equipped Items")
        if items:
            for item in items:
                rarity = item.get("rarity", "COMMON")
                color = RARITY_COLORS.get(rarity, "#9e9e9e")
                cost_str = f"{item.get('cost', 0):,.0f}g"
                stats_str = ", ".join(f"{k}: {v}" for k, v in item.get("stats", {}).items())
                st.markdown(
                    f'<div style="padding: 8px 12px; margin: 4px 0; border-left: 4px solid {color}; '
                    f'background: rgba(0,0,0,0.02); border-radius: 6px;">'
                    f'<strong style="color: {color};">{item.get("name", "?")}</strong> '
                    f'<span style="color: #888; font-size: 12px;">({item.get("slot", "?")})</span><br/>'
                    f'<span style="font-size: 13px; color: #555;">{stats_str} — {cost_str}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("No items equipped.")

    with col2:
        st.subheader("📊 Stat Distribution")
        _render_stat_radar(items)

    # Synergies
    synergies = data.get("synergies", [])
    if synergies:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("✨ Active Synergies")
        for syn in synergies:
            st.markdown(
                f'<span class="status-pill info">{syn}</span>&nbsp;',
                unsafe_allow_html=True,
            )

    # Solver info
    solver = data.get("solver", "unknown")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(f"Solver: **{solver}**")


def _render_stat_radar(items: list) -> None:
    """Render a radar chart of aggregated build stats."""
    if not items:
        st.write("No stats to display.")
        return

    # Aggregate stats
    totals: Dict[str, float] = {}
    for item in items:
        for stat, val in item.get("stats", {}).items():
            totals[stat] = totals.get(stat, 0) + val

    if not totals:
        st.write("No stats to display.")
        return

    categories = list(totals.keys())
    values = list(totals.values())

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(26, 115, 232, 0.15)",
            line=dict(color="#1a73e8", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showline=False, gridcolor="#e8eaed"),
            angularaxis=dict(gridcolor="#e8eaed"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
