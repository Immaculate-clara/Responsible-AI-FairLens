"""Page 5: Fairness Report — Algorithmic Equity Gauges, Metric Cards, and Intersectional Disparity."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.state import (
    init_session_state, render_sidebar_progress, require_model_guard
)
from core.fairness_metrics import (
    compute_fairness_metrics, compute_intersectional_fairness
)
from core.utils import (
    apply_theme_to_figure, get_metric_status, COLOR_FAIR, COLOR_CAUTION, COLOR_BIASED, COLOR_PRIMARY
)
from ui.components import (
    load_custom_css, render_metric_card, render_fairness_gauge, render_bias_story_cards
)

st.set_page_config(page_title="Fairness Report | FairLens", page_icon="⚖️", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_model_guard("Fairness Report")

active_model_name = st.session_state.get("active_model_name", "Evaluated Model")
prepared_data = st.session_state["prepared_data"]
y_test = st.session_state["y_test"]
y_pred = st.session_state["model_predictions"][active_model_name]
sens_test_df = st.session_state["sensitive_test"]

st.title("⚖️ Comprehensive Algorithmic Fairness Report")
st.markdown(
    f"<p style='color: #94a3b8;'>Quantitative equity audit for active model: <b>{active_model_name}</b>. "
    f"Evaluates statistical parity, error rate balance, and statutory EEOC guidelines.</p>",
    unsafe_allow_html=True,
)

# Select Demographic Attribute to Evaluate
selected_sens = st.selectbox(
    "Select Demographic Attribute for Fairness Audit:",
    options=st.session_state["sensitive_cols"],
)

# Compute live fairness metrics for chosen attribute
with st.spinner("Computing statistical parity and subgroup disparity metrics..."):
    fairness = compute_fairness_metrics(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sens_test_df[selected_sens],
        attribute_name=selected_sens,
    )
    st.session_state["fairness_results"] = fairness

score = fairness["fairness_score"]
dir_val = fairness["disparate_impact_ratio"]
dpd_val = fairness["demographic_parity_difference"]
eod_val = fairness["equal_opportunity_difference"]
eq_odds_val = fairness["equalized_odds_difference"]

# Top Row: Fairness Gauge + Executive Summary Card
col_gauge, col_summary = st.columns([1, 1.2])

with col_gauge:
    fig_gauge = render_fairness_gauge(score, title="FairLens Equity Index")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_summary:
    status_key, badge_label, _ = get_metric_status("fairness_score", score)
    border_col = COLOR_FAIR if status_key == "fair" else (COLOR_CAUTION if status_key == "caution" else COLOR_BIASED)
    
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {border_col}; height: 90%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">Audit Synthesis</span>
                <span style="font-size: 0.8rem; font-weight: 800; color: {border_col};">{badge_label}</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.6;">
                {fairness['summary_text']}
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #94a3b8;">
                Audited against <b>{len(fairness['unique_groups'])} demographic groups</b>: {', '.join([f'`{g}`' for g in fairness['unique_groups']])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Metric Cards Grid
st.markdown("### 📊 Statutory & Statistical Parity Metrics")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    s_key, b_lbl, t_tip = get_metric_status("disparate_impact", dir_val)
    render_metric_card(
        title="Disparate Impact (4/5ths)",
        value=f"{dir_val:.3f}",
        subtext=b_lbl,
        status=s_key,
        tooltip=t_tip,
    )

with col_m2:
    s_key, b_lbl, t_tip = get_metric_status("demographic_parity", dpd_val)
    render_metric_card(
        title="Demographic Parity Diff",
        value=f"{dpd_val:.1%}",
        subtext=b_lbl,
        status=s_key,
        tooltip=t_tip,
    )

with col_m3:
    s_key, b_lbl, t_tip = get_metric_status("equal_opportunity", eod_val)
    render_metric_card(
        title="Equal Opportunity Diff",
        value=f"{eod_val:.1%}",
        subtext=b_lbl,
        status=s_key,
        tooltip=t_tip,
    )

with col_m4:
    s_key, b_lbl, t_tip = get_metric_status("equal_opportunity", eq_odds_val)
    render_metric_card(
        title="Equalized Odds Diff",
        value=f"{eq_odds_val:.1%}",
        subtext=b_lbl,
        status=s_key,
        tooltip="Max disparity between TPR and FPR differences across subgroups.",
    )

st.markdown("---")

# Group Selection Rates & Disparity Chart
st.markdown("### 📈 Group Selection Rate Divergence")

sr_data = fairness["selection_rates"]
groups = list(sr_data.keys())
rates = [sr_data[g] * 100 for g in groups]
bar_colors = [COLOR_PRIMARY if r == max(rates) else (COLOR_BIASED if r == min(rates) and min(rates) < 0.8 * max(rates) else COLOR_CYAN) for r in rates]

fig_sr = go.Figure(
    data=[
        go.Bar(
            x=groups,
            y=rates,
            text=[f"{r:.1f}%" for r in rates],
            textposition="auto",
            marker_color=bar_colors,
            hovertext=[f"Group: <b>{g}</b><br>Predicted Approval Rate: {r:.1f}%" for g, r in zip(groups, rates)],
            hoverinfo="text",
        )
    ]
)
fig_sr.update_layout(
    title=f"Predicted Favorable Decision Rate by <b>{selected_sens}</b>",
    xaxis_title=selected_sens,
    yaxis_title="Selection Rate (% Predicted Positive)",
    yaxis_range=[0, 105],
    height=340,
)
st.plotly_chart(apply_theme_to_figure(fig_sr), use_container_width=True)

st.markdown("---")

# Intersectional Analysis Heatmap
st.markdown("### 🔀 Intersectional Demographic Analysis")
st.markdown(
    "<p style='color: #94a3b8; font-size: 0.9rem;'>Intersectional fairness evaluates whether compounding demographic "
    "identities (e.g. Gender combined with Age or Race) experience compounded disadvantage.</p>",
    unsafe_allow_html=True,
)

all_sens = st.session_state["sensitive_cols"]
if len(all_sens) >= 2:
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        attr1 = st.selectbox("Intersectional Dimension 1:", options=all_sens, index=0)
    with col_i2:
        attr2 = st.selectbox("Intersectional Dimension 2:", options=[a for a in all_sens if a != attr1], index=0)

    raw_test_df = st.session_state["raw_df"].loc[sens_test_df.index]
    pivot_rates, pivot_counts = compute_intersectional_fairness(
        df=raw_test_df,
        y_true=y_test,
        y_pred=y_pred,
        attr1=attr1,
        attr2=attr2,
    )

    # Plotly interactive heatmap
    fig_heat = px.imshow(
        pivot_rates * 100,
        labels=dict(x=attr2, y=attr1, color="Selection Rate (%)"),
        x=pivot_rates.columns,
        y=pivot_rates.index,
        color_continuous_scale="Viridis",
        text_auto=".1f",
    )
    fig_heat.update_layout(
        title=f"Compounded Selection Rate: {attr1} × {attr2}",
        height=380,
    )
    st.plotly_chart(apply_theme_to_figure(fig_heat), use_container_width=True)

else:
    st.info(
        "💡 To unlock 2D Intersectional Heatmaps, configure at least two sensitive attributes in Setup.",
        icon="ℹ️"
    )

st.markdown("---")

# Dynamic Bias Story Mode (5 Narrative Cards Generated Dynamically)
best_g = fairness["best_group"]
worst_g = fairness["worst_group"]
best_rate = sr_data[best_g]
worst_rate = sr_data[worst_g]

bias_stories = [
    {
        "icon": "⚖️",
        "title": "Statutory 80% Rule Compliance",
        "color": COLOR_FAIR if dir_val >= 0.80 else COLOR_BIASED,
        "body": (
            f"The model's disparate impact ratio for {selected_sens} is <b>{dir_val:.3f}</b>. "
            + ("This complies with the US Equal Employment Opportunity Commission (EEOC) 80% threshold."
               if dir_val >= 0.80 else
               f"This is a violation of the EEOC 80% rule. Subgroup <code>{worst_g}</code> receives favorable decisions at only <b>{dir_val:.1%}</b> the rate of subgroup <code>{best_g}</code>.")
        ),
    },
    {
        "icon": "🎯",
        "title": "Demographic Parity Gap",
        "color": COLOR_FAIR if dpd_val <= 0.10 else (COLOR_CAUTION if dpd_val <= 0.20 else COLOR_BIASED),
        "body": (
            f"The absolute demographic parity difference is <b>{dpd_val:.1%}</b>. "
            f"Subgroup <code>{best_g}</code> has an acceptance rate of {best_rate:.1%}, compared to {worst_rate:.1%} for <code>{worst_g}</code>."
        ),
    },
    {
        "icon": "🛡️",
        "title": "Equal Opportunity in Positive Outcomes",
        "color": COLOR_FAIR if eod_val <= 0.10 else COLOR_BIASED,
        "body": (
            f"The Equal Opportunity Difference is <b>{eod_val:.1%}</b>. This measures the difference in True Positive Rates (Recall) "
            "among genuinely qualified candidates across demographic groups."
        ),
    },
    {
        "icon": "⚠️",
        "title": "Subgroup Sample Reliability",
        "color": COLOR_FAIR if not fairness.get("small_group_warnings") else COLOR_CAUTION,
        "body": (
            f"FairLens evaluated {len(y_test):,} held-out records. "
            + ("All demographic subgroups have adequate representation (N ≥ 30) for reliable statistical inference."
               if not st.session_state.get("data_audit_results", {}).get("small_group_warnings") else
               "Some demographic subgroups have fewer than 30 observations; their fairness metrics should be treated with caution due to wide confidence intervals.")
        ),
    },
    {
        "icon": "🔍",
        "title": "Recommended Action",
        "color": COLOR_PRIMARY,
        "body": (
            "If disparate impact or equal opportunity differences are unacceptable for your domain, proceed to the "
            "<b>'Why Is It Biased?'</b> module to diagnose proxy features, followed by <b>'Fix It'</b> to apply Fairlearn mitigation."
        ),
    },
]

render_bias_story_cards(bias_stories)

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Explore Root Cause (Why Is It Biased?) ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/6_Why_Is_It_Biased.py")

