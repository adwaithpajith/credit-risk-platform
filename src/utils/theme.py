"""
Shared visual theme for the Streamlit app: a black-and-silver, console/tech
aesthetic, no emoji anywhere.

Usage in any page:

    from src.utils.theme import apply_theme, page_header, style_fig

    st.set_page_config(page_title="EDA", layout="wide")
    apply_theme()
    page_header("Data Understanding & Exploratory Analysis")
    ...
    fig = px.bar(...)
    st.plotly_chart(style_fig(fig), use_container_width=True)
"""
from __future__ import annotations

import streamlit as st
from textwrap import dedent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#050607"
PANEL = "#101214"
BORDER = "#2a2d31"
SILVER = "#c9cdd3"
SILVER_BRIGHT = "#eef0f2"
SILVER_DIM = "#7d838c"
ACCENT_LINE = "linear-gradient(90deg, transparent, #9aa1ab, transparent)"

# Risk-band colors are kept semantically distinct (red/amber/green) even
# inside an otherwise monochrome theme -- they carry real decision meaning
# and shouldn't be flattened to silver.
RISK_COLORS = {"Low": "#4ADE80", "Medium": "#F5B84B", "High": "#F0616B"}

PLOTLY_LAYOUT = {
    "paper_bgcolor": PANEL,
    "plot_bgcolor": PANEL,
    "font": {"color": SILVER, "family": "IBM Plex Mono, monospace"},
    "title": {"font": {"color": SILVER_BRIGHT, "family": "IBM Plex Mono, monospace", "size": 16}},
    "xaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER, "linecolor": BORDER},
    "yaxis": {"gridcolor": BORDER, "zerolinecolor": BORDER, "linecolor": BORDER},
    "colorway": ["#c9cdd3", "#7d838c", "#eef0f2", "#4ADE80", "#F5B84B", "#F0616B"],
    "legend": {"font": {"color": SILVER}},
    "margin": {"t": 60},
}


def style_fig(fig):
    """Apply the dark/silver template to a plotly figure and return it (for chaining)."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def apply_theme() -> None:
    """Inject the black/silver CSS. Call once near the top of every page,
    right after st.set_page_config()."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Orbitron:wght@600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Mono', monospace;
        }}

        .stApp {{
            background: radial-gradient(circle at 20% 0%, #0c0d0f 0%, {BG} 45%, #030304 100%);
            color: {SILVER};
        }}

        [data-testid="stSidebar"] {{
            background: #08090a;
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] * {{
            color: {SILVER} !important;
        }}

        h1, h2, h3 {{
            font-family: 'Orbitron', 'IBM Plex Mono', monospace !important;
            color: {SILVER_BRIGHT} !important;
            letter-spacing: 0.03em;
        }}

        p, span, label, li {{
            color: {SILVER};
        }}

        hr {{
            border: none;
            height: 1px;
            background: {ACCENT_LINE};
            margin: 1.6rem 0;
        }}

        [data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-left: 3px solid {SILVER_DIM};
            border-radius: 4px;
            padding: 0.9rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {SILVER_DIM} !important;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}
        [data-testid="stMetricValue"] {{
            color: {SILVER_BRIGHT} !important;
            font-family: 'Orbitron', monospace !important;
        }}

        .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {{
            background: linear-gradient(180deg, #1c1e21, #101112);
            color: {SILVER_BRIGHT};
            border: 1px solid {SILVER_DIM};
            border-radius: 3px;
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.8rem;
            transition: all 0.15s ease;
        }}
        .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {{
            border-color: {SILVER_BRIGHT};
            box-shadow: 0 0 12px rgba(200,205,211,0.25);
            color: {SILVER_BRIGHT};
        }}

        [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {BORDER};
        }}
        [data-baseweb="tab"] {{
            background: {PANEL};
            color: {SILVER_DIM} !important;
            border: 1px solid {BORDER};
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
        }}
        [aria-selected="true"] {{
            color: {SILVER_BRIGHT} !important;
            border-color: {SILVER_DIM} !important;
        }}

        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid {BORDER};
            border-radius: 4px;
        }}

        .streamlit-expanderHeader {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 4px;
            color: {SILVER} !important;
            font-family: 'IBM Plex Mono', monospace;
        }}

        div[data-testid="stForm"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 1.4rem;
        }}

        [data-testid="stChatMessage"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 6px;
        }}

        [data-testid="stAlert"] {{
            background: {PANEL} !important;
            border: 1px solid {BORDER} !important;
        }}

        code, pre {{
            background: #0a0b0c !important;
            border: 1px solid {BORDER} !important;
            color: {SILVER_BRIGHT} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None) -> None:
    """Console-style page header: bracketed uppercase title + thin silver rule."""
    subtitle_html = (
        f'<div style="color:{SILVER_DIM}; font-size:0.85rem; margin-top:0.25rem;">{subtitle}</div>'
        if subtitle else ""
    )
    html = (
        f'<div style="margin-bottom:0.6rem;">'
        f'<div style="font-family:\'Orbitron\',monospace; font-size:1.7rem; '
        f'font-weight:700; letter-spacing:0.06em; color:{SILVER_BRIGHT}; '
        f'text-transform:uppercase;">{title}</div>'
        f'{subtitle_html}'
        f'<div style="height:1px; background:{ACCENT_LINE}; margin-top:0.8rem;"></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def status_badge(label: str, status: str = "neutral") -> str:
    """Return an HTML pill badge. Render it with st.markdown(html, unsafe_allow_html=True)."""
    colors = {
        "ok": "#4ADE80", "warn": "#F5B84B", "error": "#F0616B",
        "info": SILVER_DIM, "neutral": SILVER,
    }
    color = colors.get(status, SILVER)
    return (
        f'<span style="border:1px solid {color}; color:{color}; padding:0.15rem 0.6rem;'
        f' border-radius:3px; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;">'
        f'{label}</span>'
    )


def risk_band_html(band: str, size: str = "1rem") -> str:
    """Return an HTML pill for a Low/Medium/High risk band."""
    color = RISK_COLORS.get(band, SILVER)
    return (
        f'<span style="border:1px solid {color}; color:{color}; padding:0.3rem 0.9rem;'
        f' border-radius:3px; font-size:{size}; letter-spacing:0.08em; text-transform:uppercase;'
        f' font-family:\'Orbitron\',monospace;">{band}</span>'
    )
