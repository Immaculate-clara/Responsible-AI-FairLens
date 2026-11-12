"""Page 3: Data Audit — Pre-Modeling Demographic Distribution and Baseline Disparity Analysis."""

import streamlit as st
from core.state import (
    init_session_state, render_sidebar_progress, require_data_guard,
    is_data_loaded, is_setup_configured
)
from core.audit import (
    compute_data_audit, create_group_size_chart, create_raw_outcome_chart,
    create_missing_value_heatmap
)
from ui.components import (
    load_custom_css, render_metric_card, render_small_group_warning
)

st.set_page_config(page_title="Data Audit | FairLens", page_icon="📊", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

# Empty State Guard
require_data_guard("Data Audit")

audit = st.session_state.get("data_audit_results")
if audit is None:
    # Compute if not already in state
    audit = compute_data_audit(
        df=st.session_state["raw_df"],
        target_col=st.session_state["target_col"],
        positive_label=st.session_state["positive_label"],
        sensitive_cols=st.session_state["sensitive_cols"],
    )
    st.session_state["data_audit_results"] = audit

st.title("📊 Pre-Training Data & Demographic Audit")
st.markdown(
    f"<p style='color: #94a3b8;'>Baseline assessment of <b>{st.session_state.get('dataset_name', 'uploaded data')}</b> "
    f"before training any machine learning models. Identifies historical outcome imbalances and representation risks.</p>",
    unsafe_allow_html=True,
)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card(
        title="Total Records",
        value=f"{audit['total_rows']:,}",
        subtext=f"{audit['total_cols']} features ingested",
        status="fair",
    )
with col2:
    pos_rate = audit["overall_positive_rate"]
    pos_status = "fair" if 0.20 <= pos_rate <= 0.80 else "caution"
    render_metric_card(
        title=f"Target: '{audit['positive_label']}' Rate",
        value=f"{pos_rate:.1%}",
        subtext=f"{audit['positive_count']:,} of {audit['total_rows']:,} rows",
        status=pos_status,
        tooltip="Overall percentage of records with the favorable outcome.",
    )
with col3:
    total_missing = sum(x["missing_count"] for x in audit["missing_summary"])
    miss_status = "fair" if total_missing == 0 else ("caution" if total_missing < (0.05 * audit['total_rows']) else "biased")
    render_metric_card(
        title="Missing Values",
        value=f"{total_missing:,}",
        subtext=f"Across {len(audit['missing_summary'])} columns",
        status=miss_status,
    )
with col4:
    n_warnings = len(audit["small_group_warnings"])
    warn_status = "fair" if n_warnings == 0 else "biased"
    render_metric_card(
        title="Small Group Alerts (<30)",
        value=f"{n_warnings}",
        subtext="Demographic groups with N < 30",
        status=warn_status,
        tooltip="Groups below 30 samples yield high statistical variance in fairness metrics.",
    )

# Small Group Warnings Banner
if audit["small_group_warnings"]:
    st.markdown("<br>", unsafe_allow_html=True)
    for w in audit["small_group_warnings"]:
        render_small_group_warning(
            group_name=f"{w['attribute']} = '{w['group']}'",
            count=w["count"],
            threshold=w["threshold"],
        )

st.markdown("---")

# Demographic Breakdown & Raw Historical Outcome Rate
st.markdown("### 👥 Demographic Representation & Raw Disparity")

selected_scol = st.selectbox(
    "Select Sensitive Attribute to Inspect:",
    options=st.session_state["sensitive_cols"],
    help="Toggle between configured sensitive demographic variables.",
)

if selected_scol in audit["group_audits"]:
    group_data = audit["group_audits"][selected_scol]
    breakdown = group_data["breakdown"]

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_size = create_group_size_chart(breakdown, selected_scol)
        st.plotly_chart(fig_size, use_container_width=True)

    with col_chart2:
        fig_outcome = create_raw_outcome_chart(breakdown, selected_scol, audit["positive_label"])
        st.plotly_chart(fig_outcome, use_container_width=True)

    # Disparity Insights
    raw_di = group_data["raw_disparate_impact"]
    raw_gap = group_data["raw_gap"]
    
    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 1rem;">
            <div style="font-size: 1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                Historical Baseline Disparity for <code>{selected_scol}</code>
            </div>
            <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
                Before any model is trained, the historical raw favorable outcome rate ranges from 
                <b>{group_data['min_rate']:.1%}</b> to <b>{group_data['max_rate']:.1%}</b> 
                (an absolute gap of <b>{raw_gap:.1%}</b>).<br>
                The pre-training Disparate Impact Ratio is <b>{raw_di:.3f}</b> 
                ({('meets' if raw_di >= 0.80 else 'violates')} the 80% rule).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Missing Values Breakdown
st.markdown("### 🧹 Data Quality & Missingness")
fig_missing = create_missing_value_heatmap(st.session_state["raw_df"])
st.plotly_chart(fig_missing, use_container_width=True)

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Train Models Side-by-Side ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/4_Model_Performance.py")

