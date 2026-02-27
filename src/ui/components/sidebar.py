"""
Sidebar control panel for the Build-Optimization dashboard.
"""

from typing import Tuple

import streamlit as st


def render_mode_selector() -> str:
    """Render the main mode selector in the sidebar."""
    st.sidebar.title("⚔️ Build Optimizer")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "📊 Dashboard Mode",
        options=[
            "Build Explorer",
            "Solver Comparison",
            "Training Monitor",
            "Item Database",
        ],
        index=0,
        help="Switch between build visualization, solver comparison, training metrics, and item browsing",
    )

    mode_map = {
        "Build Explorer": "build_explorer",
        "Solver Comparison": "solver_comparison",
        "Training Monitor": "training_monitor",
        "Item Database": "item_database",
    }
    return mode_map.get(mode, "build_explorer")


def render_auto_refresh_toggle() -> Tuple[bool, int]:
    """Render auto-refresh controls."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Auto-Refresh")

    enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=2,
        max_value=30,
        value=5,
        disabled=not enabled,
    )
    return enabled, interval


def render_about_section() -> None:
    """Render an about section at the bottom of the sidebar."""
    from ui import __version__

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; color: #9aa0a6; font-size: 11px;">
            <p style="margin: 0;">Build Optimizer</p>
            <p style="margin: 2px 0 0 0;">Control Tower v{__version__}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
