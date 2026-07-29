"""Dashboard styling: colors, CSS, and KPI card generators."""

from .colors import CHART_COLORS, KPI_COLORS, STATUS_COLORS, get_page_config
from .css import CUSTOM_CSS
from .kpi import create_kpi_html, create_kpi_row, create_kpi_row_with_deltas

__all__ = [
    "KPI_COLORS",
    "CHART_COLORS",
    "STATUS_COLORS",
    "get_page_config",
    "CUSTOM_CSS",
    "create_kpi_html",
    "create_kpi_row",
    "create_kpi_row_with_deltas",
]
