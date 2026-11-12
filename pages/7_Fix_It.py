"""Page 7: Fix It — Algorithmic Bias Mitigation and Fairness-Accuracy Pareto Trade-offs."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.state import (
    init_session_state, render_sidebar_progress, require_model_guard
)
from core.mitigation import (
    run_reweighing_mitigation, run_exponentiated_gradient_mitigation,
    run_threshold_optimizer_mitigation, create_tradeoff_chart
)
from core.utils import (
    apply_theme_to_figure, COLOR_FAIR, COLOR_PRIMARY, COLOR_BIASED, COLOR_CAUTION, COLOR_CYAN
)
from ui.components import (
    load_custom_css, render_metric_card, render_fairness_gauge
)

st.set_page_config(page_title="Fix It | FairLens", page_icon="🛠️", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_model_guard("Mitigation Engine")

active_model_name = st.session_state.get("active_model_name", "Random Forest")
base_model = st.session_state["trained_models"][active_model_name]
prepared_data = st.session_state["prepared_data"]
base_fairness = st.session_state["fairness_results"]
primary_sens = st.session_state["sensitive_cols"][0]

st.title("🛠️ Fix It: Algorithmic Bias Mitigation")
st.markdown(
    "<p style='color: #94a3b8;'>Apply state-of-the-art fairness interventions across the machine learning lifecycle. "
    "Observe empirical trade-offs between predictive accuracy and statistical parity.</p>",
    unsafe_allow_html=True,
)

# Terminology notice
st.info(
    "💡 **Scientific Integrity Notice**: In accordance with responsible AI standards, FairLens reports "
    "that **bias is reduced**, never 'bias removed'. Real-world datasets contain structural complexities "
    "that no single mathematical constraint can eliminate entirely.",
    icon="ℹ️",
)

st.markdown("### ⚙️ Configure Mitigation Strategy")

col_strat1, col_strat2, col_strat3 = st.columns(3)

with col_strat1:
    mitigation_family = st.selectbox(
        "Intervention Stage:",
        options=[
            "1. Pre-processing (Sample Reweighing)",
            "2. In-processing (Fairlearn Exponentiated Gradient)",
            "3. Post-processing (Threshold Optimizer)",
        ],
        help="Pre-processing balances historical weights; in-processing constrains model optimization; post-processing adjusts group cutoffs.",
    )

with col_strat2:
    constraint_choice = st.selectbox(
        "Fairness Constraint:",
        options=["DemographicParity", "EqualizedOdds"],
        help="DemographicParity equalizes positive outcome rates; EqualizedOdds equalizes True Positive and False Positive rates.",
    )

with col_strat3:
    target_sens = st.selectbox(
        "Protected Demographic Attribute:",
        options=st.session_state["sensitive_cols"],
        help="The sensitive feature to optimize fairness across.",
    )

if st.button("🚀 Apply Mitigation Algorithm", type="primary", use_container_width=True):
    with st.spinner("Executing fairness mitigation algorithm and re-evaluating test split..."):
        if "Pre-processing" in mitigation_family:
            res = run_reweighing_mitigation(
                prepared_data=prepared_data,
                sensitive_col=target_sens,
                base_model_type="Logistic Regression" if "Logistic" in active_model_name else "Random Forest",
            )
        elif "In-processing" in mitigation_family:
            res = run_exponentiated_gradient_mitigation(
                prepared_data=prepared_data,
                sensitive_col=target_sens,
                constraint_type=constraint_choice,
            )
        else:
            res = run_threshold_optimizer_mitigation(
                base_model=base_model,
                prepared_data=prepared_data,
                sensitive_col=target_sens,
                constraint_type=constraint_choice,
            )

        # Store in session state
        st.session_state["selected_mitigated_model"] = res
        
        # Add to history if not duplicate
        history = st.session_state.get("mitigation_results", [])
        history.append({
            "name": res["method"],
            "accuracy": res["accuracy"],
            "fairness_score": res["fairness"]["fairness_score"],
            "dir": res["fairness"]["disparate_impact_ratio"],
            "dpd": res["fairness"]["demographic_parity_difference"],
            "eod": res["fairness"]["equal_opportunity_difference"],
            "model": res["model"],
            "fairness": res["fairness"],
        })
        st.session_state["mitigation_results"] = history

    st.success(f"Successfully executed: {res['method']}! Review the Before vs After comparison below.")

# BEFORE vs AFTER Comparison
mitigated_res = st.session_state.get("selected_mitigated_model")
if mitigated_res is not None:
    st.markdown("---")
    st.markdown("### ⚖️ Before vs. After Mitigation Comparison")

    before_score = base_fairness["fairness_score"]
    after_score = mitigated_res["fairness"]["fairness_score"]
    score_delta = after_score - before_score

    before_dir = base_fairness["disparate_impact_ratio"]
    after_dir = mitigated_res["fairness"]["disparate_impact_ratio"]

    before_dpd = base_fairness["demographic_parity_difference"]
    after_dpd = mitigated_res["fairness"]["demographic_parity_difference"]

    before_acc = st.session_state["model_performance_results"][active_model_name]["overall"]["accuracy"]
    after_acc = mitigated_res["accuracy"]

    col_bva1, col_bva2, col_bva3, col_bva4 = st.columns(4)

    with col_bva1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #818cf8;">
                <div class="metric-header">Fairness Score (0-100)</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff;">
                    {round(before_score):d} ➔ <span style="color: #34d399;">{round(after_score):d}</span>
                </div>
                <div class="metric-sub">Delta: {'+' if score_delta >= 0 else ''}{score_delta:.1f} pts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bva2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #38bdf8;">
                <div class="metric-header">Disparate Impact Ratio</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff;">
                    {before_dir:.3f} ➔ <span style="color: {'#34d399' if after_dir >= 0.8 else '#fbbf24'};">{after_dir:.3f}</span>
                </div>
                <div class="metric-sub">Target: ≥ 0.800 (4/5ths Rule)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bva3:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #f59e0b;">
                <div class="metric-header">Demographic Parity Gap</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff;">
                    {before_dpd:.1%} ➔ <span style="color: #34d399;">{after_dpd:.1%}</span>
                </div>
                <div class="metric-sub">Selection Rate Disparity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bva4:
        acc_delta = (after_acc - before_acc) * 100
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #a855f7;">
                <div class="metric-header">Accuracy Impact</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff;">
                    {before_acc:.1%} ➔ <span style="color: #e2e8f0;">{after_acc:.1%}</span>
                </div>
                <div class="metric-sub">Trade-off: {'+' if acc_delta >= 0 else ''}{acc_delta:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Before vs After Selection Rates Comparison Chart
    st.markdown("#### Selection Rate Divergence: Before vs After")
    before_sr = base_fairness["selection_rates"]
    after_sr = mitigated_res["fairness"]["selection_rates"]
    groups = list(before_sr.keys())

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=groups,
        y=[before_sr[g] * 100 for g in groups],
        name=f"Baseline ({active_model_name})",
        marker_color="rgba(239, 68, 68, 0.7)",
        text=[f"{before_sr[g]*100:.1f}%" for g in groups],
        textposition="auto",
    ))
    fig_comp.add_trace(go.Bar(
        x=groups,
        y=[after_sr.get(g, 0.0) * 100 for g in groups],
        name=f"Mitigated ({mitigated_res['method'].split(':')[0]})",
        marker_color="rgba(16, 185, 129, 0.8)",
        text=[f"{after_sr.get(g, 0.0)*100:.1f}%" for g in groups],
        textposition="auto",
    ))
    fig_comp.update_layout(
        title=f"Subgroup Selection Rate Comparison: Before vs. After Mitigation ({target_sens})",
        barmode="group",
        xaxis_title=target_sens,
        yaxis_title="Selection Rate (% Positive)",
        yaxis_range=[0, 105],
        height=340,
    )
    st.plotly_chart(apply_theme_to_figure(fig_comp), use_container_width=True)

st.markdown("---")

# Fairness vs Accuracy Pareto Trade-off Frontier
st.markdown("### 📉 Fairness vs. Accuracy Pareto Trade-off Frontier")
st.markdown(
    "<p style='color: #94a3b8; font-size: 0.9rem;'>Track all model iterations across this session to evaluate whether gains in fairness "
    "incur an acceptable or negligible trade-off in overall classification accuracy.</p>",
    unsafe_allow_html=True,
)

# Collect baseline models + mitigated models
all_runs = []
for m_name in st.session_state["model_performance_results"]:
    m_acc = st.session_state["model_performance_results"][m_name]["overall"]["accuracy"]
    # Re-evaluate baseline score if needed
    if m_name == active_model_name:
        all_runs.append({
            "name": f"Baseline: {m_name}",
            "accuracy": m_acc,
            "fairness_score": base_fairness["fairness_score"],
        })
    else:
        all_runs.append({
            "name": f"Baseline: {m_name}",
            "accuracy": m_acc,
            "fairness_score": base_fairness["fairness_score"],
        })

for m_run in st.session_state.get("mitigation_results", []):
    all_runs.append({
        "name": m_run["name"].split(":")[0] + " (Fix)",
        "accuracy": m_run["accuracy"],
        "fairness_score": m_run["fairness_score"],
    })

fig_tradeoff = create_tradeoff_chart(all_runs)
st.plotly_chart(fig_tradeoff, use_container_width=True)

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Explore What-If Simulator ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/8_What_If_Simulator.py")

