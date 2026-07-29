"""
Item Database page — browse and filter all game items.
"""

from typing import Any, Dict, List

import plotly.express as px
import streamlit as st

from ui.services.data_loader import discover_item_files, load_items_from_json
from ui.styles.colors import RARITY_COLORS


def render_item_database() -> None:
    """Render the Item Database page."""
    st.title("📚 Item Database")

    item_files = discover_item_files()

    if not item_files:
        st.info("No item data files found. Place item JSON files in `data/`.")
        _render_demo_database()
        return

    import os

    selected = st.selectbox(
        "Select Data File",
        options=item_files,
        format_func=lambda p: os.path.basename(p),
    )

    if selected:
        items = load_items_from_json(selected)
        if items:
            _render_item_table(items)
        else:
            st.warning("No items found in the selected file.")


def _render_demo_database() -> None:
    """Render a demo item database."""
    demo_items = [
        {
            "name": "Flame Sword",
            "slot": "WEAPON",
            "rarity": "EPIC",
            "level": 15,
            "cost": 1200,
            "stats": {"attack": 85, "critical_rate": 12},
            "tags": ["fire", "melee"],
        },
        {
            "name": "Dragon Helm",
            "slot": "HELMET",
            "rarity": "RARE",
            "level": 12,
            "cost": 800,
            "stats": {"defense": 45, "health": 60},
            "tags": ["fire", "armor"],
        },
        {
            "name": "Shadow Vest",
            "slot": "CHEST",
            "rarity": "LEGENDARY",
            "level": 20,
            "cost": 1500,
            "stats": {"defense": 70, "speed": 25},
            "tags": ["shadow", "armor"],
        },
        {
            "name": "Swift Boots",
            "slot": "BOOTS",
            "rarity": "UNCOMMON",
            "level": 5,
            "cost": 400,
            "stats": {"speed": 40, "defense": 15},
            "tags": ["speed"],
        },
        {
            "name": "Ruby Ring",
            "slot": "RING_1",
            "rarity": "RARE",
            "level": 10,
            "cost": 300,
            "stats": {"critical_damage": 30, "attack": 10},
            "tags": ["fire", "jewelry"],
        },
        {
            "name": "Iron Shield",
            "slot": "ACCESSORY_1",
            "rarity": "COMMON",
            "level": 1,
            "cost": 100,
            "stats": {"defense": 25},
            "tags": ["armor"],
        },
        {
            "name": "Emerald Amulet",
            "slot": "AMULET",
            "rarity": "EPIC",
            "level": 18,
            "cost": 950,
            "stats": {"health": 80, "speed": 10},
            "tags": ["nature", "jewelry"],
        },
        {
            "name": "Obsidian Gauntlets",
            "slot": "GLOVES",
            "rarity": "RARE",
            "level": 14,
            "cost": 650,
            "stats": {"attack": 30, "defense": 35, "critical_rate": 5},
            "tags": ["shadow", "melee"],
        },
    ]
    st.markdown("### Demo Items")
    _render_item_table(demo_items)


def _render_item_table(items: List[Dict[str, Any]]) -> None:
    """Render a filterable item table."""
    import pandas as pd

    # Filters
    col1, col2, col3 = st.columns(3)

    all_slots = sorted({item.get("slot", "?") for item in items})
    all_rarities = sorted({item.get("rarity", "COMMON") for item in items})
    all_tags = sorted({tag for item in items for tag in item.get("tags", [])})

    with col1:
        slot_filter = st.multiselect("Filter by Slot", options=all_slots, default=[])
    with col2:
        rarity_filter = st.multiselect("Filter by Rarity", options=all_rarities, default=[])
    with col3:
        tag_filter = st.multiselect("Filter by Tag", options=all_tags, default=[])

    # Level range
    levels = [item.get("level", 1) for item in items]
    min_lvl, max_lvl = min(levels), max(levels)
    if min_lvl < max_lvl:
        level_range = st.slider("Level Range", min_lvl, max_lvl, (min_lvl, max_lvl))
    else:
        level_range = (min_lvl, max_lvl)

    # Apply filters
    filtered = items
    if slot_filter:
        filtered = [i for i in filtered if i.get("slot") in slot_filter]
    if rarity_filter:
        filtered = [i for i in filtered if i.get("rarity") in rarity_filter]
    if tag_filter:
        filtered = [i for i in filtered if any(t in i.get("tags", []) for t in tag_filter)]
    filtered = [i for i in filtered if level_range[0] <= i.get("level", 1) <= level_range[1]]

    st.caption(f"Showing {len(filtered)} of {len(items)} items")

    # Build DataFrame
    rows = []
    for item in filtered:
        stats = item.get("stats", {})
        total = sum(stats.values())
        cost = item.get("cost", 0)
        rows.append(
            {
                "Name": item.get("name", "?"),
                "Slot": item.get("slot", "?"),
                "Rarity": item.get("rarity", "COMMON"),
                "Level": item.get("level", 1),
                "Cost": cost,
                "Total Stats": total,
                "Efficiency": round(total / max(cost, 1), 3),
                "Tags": ", ".join(item.get("tags", [])),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Charts
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Stats by Rarity")
        if not df.empty:
            fig = px.box(
                df,
                x="Rarity",
                y="Total Stats",
                color="Rarity",
                color_discrete_map=RARITY_COLORS,
                category_orders={"Rarity": ["COMMON", "UNCOMMON", "RARE", "EPIC", "LEGENDARY"]},
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📊 Cost vs Efficiency")
        if not df.empty:
            fig2 = px.scatter(
                df,
                x="Cost",
                y="Efficiency",
                color="Rarity",
                hover_name="Name",
                size="Total Stats",
                color_discrete_map=RARITY_COLORS,
                category_orders={"Rarity": ["COMMON", "UNCOMMON", "RARE", "EPIC", "LEGENDARY"]},
            )
            fig2.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)
