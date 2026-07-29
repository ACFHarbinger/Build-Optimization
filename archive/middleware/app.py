"""
Build Optimizer Control Tower — Main Streamlit Entry Point.

A unified dashboard for exploring builds, comparing solvers,
monitoring RL training, and browsing item databases.

Usage:
    streamlit run src/ui/app.py
"""

import time

import streamlit as st

from .ui.components.sidebar import (
    render_about_section,
    render_auto_refresh_toggle,
    render_mode_selector,
)
from .ui.pages import (
    render_build_explorer,
    render_item_database,
    render_solver_comparison,
    render_training_monitor,
)
from .ui.styles.colors import get_page_config
from .ui.styles.css import CUSTOM_CSS


def main() -> None:
    """Main entry point for the dashboard."""
    st.set_page_config(**get_page_config())
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    mode = render_mode_selector()
    auto_refresh, refresh_interval = render_auto_refresh_toggle()
    render_about_section()

    # Page dispatch
    if mode == "build_explorer":
        with st.spinner("Loading Build Explorer..."):
            render_build_explorer()
    elif mode == "solver_comparison":
        with st.spinner("Loading Solver Comparison..."):
            render_solver_comparison()
    elif mode == "training_monitor":
        with st.spinner("Loading Training Monitor..."):
            render_training_monitor()
    elif mode == "item_database":
        with st.spinner("Loading Item Database..."):
            render_item_database()

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
