"""Reusable UI components for FairLens: cards, gauges, empty states, and badges."""

from typing import Optional, List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
from core.utils import COLOR_FAIR, COLOR_CAUTION, COLOR_BIASED, COLOR_PRIMARY, apply_theme_to_figure


def load_custom_css() -> None:
    """Inject FairLens glassmorphism styling into Streamlit DOM."""
    import os
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_empty_state(
    title: str = "Upload a dataset to begin",
    description: str = "No data has been loaded in this session yet. Upload a CSV or Excel file to get started.",
    action_label: str = "Go to Upload & Setup",
    target_page: str = "pages/2_Upload_and_Setup.py",
) -> None:
    """Render an authoritative empty state card with clean guidance and navigation."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-title">{title}</div>
            <div class="empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(f"🚀 {action_label}", use_container_width=True, type="primary"):
            st.switch_page(target_page)


def render_badge(label: str, status: str = "fair") -> str:
    """Generate HTML markup for a status badge."""
    cls_map = {
        "fair": "badge-fair",
        "caution": "badge-caution",
        "biased": "badge-biased",
    }
    cls = cls_map.get(status.lower(), "badge-caution")
    return f'<span class="{cls}">{label}</span>'


def render_metric_card(
    title: str,
    value: str,
    subtext: str,
    status: str = "fair",
    tooltip: Optional[str] = None,
) -> None:
    """Render a glassmorphic metric card with status color and optional tooltip."""
    border_color_map = {
        "fair": "rgba(16, 185, 129, 0.4)",
        "caution": "rgba(245, 158, 11, 0.4)",
        "biased": "rgba(239, 68, 68, 0.4)",
    }
    val_color_map = {
        "fair": "#34d399",
        "caution": "#fbbf24",
        "biased": "#f87171",
    }
    border_color = border_color_map.get(status.lower(), "rgba(255, 255, 255, 0.1)")
    val_color = val_color_map.get(status.lower(), "#ffffff")
    badge_html = render_badge(status.upper(), status=status)
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""

    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 3px solid {border_color};" {tooltip_attr}>
            <div class="metric-header">
                <span>{title}</span>
                {badge_html}
            </div>
            <div class="metric-value" style="color: {val_color};">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fairness_gauge(score: float, title: str = "Overall Fairness Score") -> go.Figure:
    """
    Generate an interactive animated Plotly gauge chart (0-100).
    Color zones: Red (0-59), Amber (60-79), Green (80-100).
    """
    bar_color = COLOR_FAIR if score >= 80 else (COLOR_CAUTION if score >= 60 else COLOR_BIASED)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:#94a3b8'>Scale 0 (Biased) to 100 (Equitable)</span>", "font": {"size": 18, "color": "#f8fafc"}},
            number={"suffix": "/100", "font": {"size": 36, "color": bar_color, "family": "JetBrains Mono, monospace"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(19, 27, 46, 0.8)",
                "borderwidth": 2,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 60], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [60, 80], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [80, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": 80,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc", "family": "Plus Jakarta Sans, sans-serif"},
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def render_bias_story_cards(stories: List[Dict[str, str]]) -> None:
    """Render 5 narrative Bias Story cards generated dynamically from computed user data."""
    st.markdown("### 📖 Dynamic Bias Story")
    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.25rem;'>Automated plain-English insights synthesized strictly from this session's dataset and model predictions.</p>", unsafe_allow_html=True)
    
    for story in stories:
        accent_color = story.get("color", COLOR_PRIMARY)
        icon = story.get("icon", "🔍")
        st.markdown(
            f"""
            <div class="story-card" style="border-left-color: {accent_color};">
                <div class="story-header" style="display:flex; align-items:center; gap: 8px;">
                    <span>{icon}</span>
                    <span>{story.get('title', 'Observation')}</span>
                </div>
                <div class="story-body">{story.get('body', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_small_group_warning(group_name: str, count: int, threshold: int = 30) -> None:
    """Display warning for groups under threshold size."""
    st.warning(
        f"⚠️ **Small Sample Alert for Group `{group_name}` ({count} rows)**: "
        f"Groups with fewer than {threshold} records have high statistical uncertainty. "
        "Fairness metrics for this group may fluctuate significantly.",
        icon="⚠️"
    )

