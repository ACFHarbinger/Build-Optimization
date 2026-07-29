"""
Custom CSS stylesheet for the Build-Optimization Streamlit dashboard.
"""

CUSTOM_CSS = """
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    h1 {
        color: #1a73e8;
        border-bottom: 3px solid #e8eaed;
        padding-bottom: 12px;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 { color: #202124; font-weight: 600; margin-top: 1.5rem; }
    h3 { color: #3c4043; font-weight: 600; }

    /* KPI Card styling */
    .kpi-card {
        border-radius: 14px;
        padding: 18px 20px;
        flex: 1 1 120px;
        min-width: 0;
        max-width: 220px;
        text-align: center;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }
    .kpi-card .label {
        font-size: 13px; opacity: 0.92; margin-bottom: 6px;
        font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase;
    }
    .kpi-card .value { font-size: 24px; font-weight: 700; }
    .kpi-card .delta { font-size: 12px; margin-top: 4px; opacity: 0.9; }
    .kpi-card .delta.positive { color: #c8e6c9; }
    .kpi-card .delta.negative { color: #ffcdd2; }
    .kpi-card .delta.neutral { color: rgba(255,255,255,0.7); }

    /* Status indicators */
    .status-good { color: #0d904f; font-weight: bold; }
    .status-warning { color: #e37400; font-weight: bold; }
    .status-error { color: #c5221f; font-weight: bold; }

    /* Status pills */
    .status-pill {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-size: 12px;
        font-weight: 600; letter-spacing: 0.3px;
    }
    .status-pill.good { background: #e8f5e9; color: #2e7d32; }
    .status-pill.warning { background: #fff3e0; color: #e65100; }
    .status-pill.error { background: #ffebee; color: #c62828; }
    .status-pill.info { background: #e3f2fd; color: #1565c0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f8f9fa; }

    /* Metric row */
    .metric-row {
        display: flex; flex-wrap: wrap; gap: 14px;
        margin-bottom: 20px; justify-content: center;
    }

    /* Chart container */
    .chart-container {
        background: white; border-radius: 10px;
        padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #e8eaed;
    }

    /* Data table styling */
    .dataframe { font-size: 14px; }
    .dataframe th { background-color: #f1f3f4; font-weight: 600; }

    /* Section divider */
    .section-divider {
        border: none; height: 1px;
        background: linear-gradient(to right, transparent, #dadce0, transparent);
        margin: 1.5rem 0;
    }

    /* Rarity badges */
    .rarity-common { color: #9e9e9e; font-weight: 600; }
    .rarity-uncommon { color: #4caf50; font-weight: 600; }
    .rarity-rare { color: #2196f3; font-weight: 600; }
    .rarity-epic { color: #9c27b0; font-weight: 600; }
    .rarity-legendary { color: #ff9800; font-weight: 600; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 500; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #1a73e8; font-weight: 600;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        h1 { color: #8ab4f8; border-bottom-color: #3c4043; }
        h2, h3 { color: #e8eaed; }
        .kpi-card { box-shadow: 0 2px 8px rgba(0,0,0,0.30); }
        .kpi-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.40); }
        .status-good { color: #34d058; }
        .status-warning { color: #ffdf5d; }
        .status-error { color: #f97583; }
        .status-pill.good { background: #1b3a1f; color: #34d058; }
        .status-pill.warning { background: #3a2e1b; color: #ffdf5d; }
        .status-pill.error { background: #3a1b1b; color: #f97583; }
        .status-pill.info { background: #1b2a3a; color: #58a6ff; }
        .chart-container { background: #292a2d; border-color: #3c4043; }
        .dataframe th { background-color: #3c4043; }
        section[data-testid="stSidebar"] { background-color: #1e1e1e; }
        .section-divider {
            background: linear-gradient(to right, transparent, #3c4043, transparent);
        }
    }
</style>
"""
