"""Utility functions, formatters, and custom Plotly styling for FairLens."""

from typing import Any, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Brand & Status Color Palette
COLOR_FAIR = "#10b981"      # Green
COLOR_CAUTION = "#f59e0b"   # Amber
COLOR_BIASED = "#ef4444"    # Red
COLOR_PRIMARY = "#6366f1"   # Indigo Accent
COLOR_CYAN = "#06b6d4"      # Cyan Accent
COLOR_BG_CARD = "#131b2e"   # Card background
COLOR_TEXT = "#f8fafc"      # Light text
COLOR_GRID = "rgba(255, 255, 255, 0.08)"

PLOTLY_DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.5)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=COLOR_TEXT, size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            showgrid=True,
            tickfont=dict(color="#94a3b8"),
            title=dict(font=dict(color="#cbd5e1")),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            showgrid=True,
            tickfont=dict(color="#94a3b8"),
            title=dict(font=dict(color="#cbd5e1")),
        ),
        legend=dict(
            bgcolor="rgba(19, 27, 46, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            bordercolor="rgba(255, 255, 255, 0.15)",
        ),
    )
)


def format_percent(val: Optional[float], decimals: int = 1) -> str:
    """Format float into clean percentage string."""
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_ratio(val: Optional[float], decimals: int = 3) -> str:
    """Format float into ratio string."""
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    return f"{val:.{decimals}f}"


def format_score(val: Optional[float]) -> str:
    """Format score out of 100."""
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    return f"{round(val):d}/100"


def get_metric_status(
    metric_name: str,
    value: float
) -> Tuple[str, str, str]:
    """
    Determine whether a metric is Fair, Caution, or Biased.
    
    Returns:
        (status_key, badge_label, tooltip_text)
        status_key is in ('fair', 'caution', 'biased')
    """
    # Disparate Impact Ratio (DIR): 4/5ths rule (0.80 - 1.25)
    if "disparate_impact" in metric_name.lower():
        if value >= 0.80 and value <= 1.25:
            return "fair", "FAIR (80% Rule)", "Complies with the EEOC four-fifths (0.80) guideline."
        elif (value >= 0.65 and value < 0.80) or (value > 1.25 and value <= 1.40):
            return "caution", "CAUTION", "Approaching unlawful disparity threshold (below 0.80 or above 1.25)."
        else:
            return "biased", "BIASED", "Severe disparate impact violating the 80% rule."

    # Demographic Parity Difference (DPD): Ideal is 0.0
    elif "demographic_parity" in metric_name.lower() or "dpd" in metric_name.lower():
        if abs(value) <= 0.10:
            return "fair", "FAIR (< 10%)", "Selection rates across groups differ by less than 10%."
        elif abs(value) <= 0.20:
            return "caution", "CAUTION (10-20%)", "Moderate selection rate divergence between demographic groups."
        else:
            return "biased", "BIASED (> 20%)", "Substantial demographic selection gap exceeding 20%."

    # Equal Opportunity Difference (EOD): Ideal is 0.0
    elif "equal_opportunity" in metric_name.lower() or "equalized_odds" in metric_name.lower():
        if abs(value) <= 0.10:
            return "fair", "FAIR (< 10%)", "True positive rates across groups are closely balanced."
        elif abs(value) <= 0.20:
            return "caution", "CAUTION (10-20%)", "Moderate divergence in qualified applicant approval rates."
        else:
            return "biased", "BIASED (> 20%)", "Large gap in approval rates for qualified members of different groups."

    # General Fairness Score (0 - 100)
    elif "fairness_score" in metric_name.lower() or "score" in metric_name.lower():
        if value >= 80.0:
            return "fair", "HIGH FAIRNESS", "Overall model complies with primary equity criteria."
        elif value >= 60.0:
            return "caution", "MODERATE RISK", "Audited disparities detected; requires targeted mitigation."
        else:
            return "biased", "CRITICAL BIAS", "Substantial disparate impact and parity violations found."

    return "caution", "AUDIT", "Metric value computed from session dataset."


def apply_theme_to_figure(fig: go.Figure) -> go.Figure:
    """Apply FairLens consistent dark glassmorphism styling to any Plotly figure."""
    fig.update_layout(template=PLOTLY_DARK_TEMPLATE)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.4)",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans, sans-serif"),
    )
    return fig
