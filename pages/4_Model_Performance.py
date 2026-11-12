"""Page 4: Model Performance — Side-by-Side Model Training & Disaggregated Group Accuracy."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.state import (
    init_session_state, render_sidebar_progress, require_data_guard,
    is_model_trained
)
from core.model_trainer import prepare_data_pipeline, train_baseline_models
from core.fairness_metrics import compute_fairness_metrics
from core.utils import apply_theme_to_figure, COLOR_PRIMARY, COLOR_CYAN, COLOR_FAIR, COLOR_BIASED
from ui.components import load_custom_css, render_metric_card

st.set_page_config(page_title="Model Performance | FairLens", page_icon="⚡", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_data_guard("Model Performance")

st.title("⚡ Model Performance & Disaggregated Accuracy")
st.markdown(
    "<p style='color: #94a3b8;'>Train benchmark models side by side. Inspect overall accuracy against "
    "per-group accuracy to reveal where standard metrics obscure performance disparities.</p>",
    unsafe_allow_html=True,
)

# Training control panel
col_ctl1, col_ctl2, col_ctl3 = st.columns([2, 1, 1])
with col_ctl1:
    test_split_pct = st.slider(
        "Test Split Ratio",
        min_value=0.15,
        max_value=0.40,
        value=0.30,
        step=0.05,
        help="Proportion of the dataset held out strictly for independent performance and fairness testing.",
    )
with col_ctl2:
    train_button = st.button("🚀 Train Models Now", type="primary", use_container_width=True)

# Train or load cached models
if train_button or not is_model_trained():
    with st.spinner("Preparing feature pipelines, encoding data, and training benchmark classifiers..."):
        # 1. Prepare data splits and preprocessor
        prep = prepare_data_pipeline(
            df=st.session_state["raw_df"],
            target_col=st.session_state["target_col"],
            positive_label=st.session_state["positive_label"],
            sensitive_cols=st.session_state["sensitive_cols"],
            exclude_cols=st.session_state["exclude_cols"],
            test_size=test_split_pct,
            random_state=42,
        )
        st.session_state["prepared_data"] = prep
        st.session_state["X_train"] = prep["X_train_proc"]
        st.session_state["X_test"] = prep["X_test_proc"]
        st.session_state["y_train"] = prep["y_train"]
        st.session_state["y_test"] = prep["y_test"]
        st.session_state["sensitive_test"] = prep["sensitive_test_df"]
        st.session_state["preprocessor"] = prep["preprocessor"]
        st.session_state["feature_names"] = prep["feature_names"]

        # 2. Train baseline models
        training_res = train_baseline_models(prep, threshold=0.5, random_state=42)
        st.session_state["trained_models"] = training_res["trained_models"]
        st.session_state["model_predictions"] = training_res["model_predictions"]
        st.session_state["model_probabilities"] = training_res["model_probabilities"]
        st.session_state["model_performance_results"] = training_res["performance_results"]
        
        # Set default active model
        if not st.session_state.get("active_model_name"):
            st.session_state["active_model_name"] = "Random Forest"

        # Auto-compute initial fairness assessment for the active model
        active_m = st.session_state["active_model_name"]
        primary_sens = st.session_state["sensitive_cols"][0]
        fairness_res = compute_fairness_metrics(
            y_true=prep["y_test"],
            y_pred=training_res["model_predictions"][active_m],
            sensitive_features=prep["sensitive_test_df"][primary_sens],
            attribute_name=primary_sens,
        )
        st.session_state["fairness_results"] = fairness_res

    st.success("Models successfully trained and disaggregated metrics computed!")

# Display Side-by-Side Model Performance
perf_dict = st.session_state["model_performance_results"]
model_names = list(perf_dict.keys())

st.markdown("### 🏆 Overall Model Comparison")

col_lr, col_rf = st.columns(2)
for idx, m_name in enumerate(model_names):
    col = col_lr if idx == 0 else col_rf
    m_data = perf_dict[m_name]["overall"]
    with col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.75rem;">
                    {m_name}
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; text-align: center;">
                    <div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Accuracy</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #34d399;">{m_data['accuracy']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Precision</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #38bdf8;">{m_data['precision']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Recall</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #818cf8;">{m_data['recall']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">F1-Score</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #f59e0b;">{m_data['f1']:.3f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Confusion Matrix Comparison
st.markdown("#### Confusion Matrix (Held-out Test Split)")
col_cm1, col_cm2 = st.columns(2)
for idx, m_name in enumerate(model_names):
    col = col_cm1 if idx == 0 else col_cm2
    cm_vals = perf_dict[m_name]["overall"]["confusion_matrix"]
    z = [[cm_vals["tn"], cm_vals["fp"]], [cm_vals["fn"], cm_vals["tp"]]]
    x = ["Predicted 0", "Predicted 1"]
    y = ["Actual 0", "Actual 1"]
    
    fig_cm = px.imshow(
        z,
        text_auto=True,
        x=x,
        y=y,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_cm.update_coloraxes(showscale=False)
    fig_cm.update_layout(
        title=f"Confusion Matrix: {m_name}",
        height=260,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    with col:
        st.plotly_chart(apply_theme_to_figure(fig_cm), use_container_width=True)

st.markdown("---")

# Per-Group Disaggregated Performance Table
st.markdown("### 🔍 Disaggregated Performance by Demographic Group")
st.markdown(
    "<p style='color: #cbd5e1; font-size: 0.9rem;'><b>Warning:</b> High overall accuracy frequently conceals large error rate gaps between demographic subgroups.</p>",
    unsafe_allow_html=True,
)

inspect_model = st.radio(
    "Select Model to Inspect Disaggregated Performance:",
    options=model_names,
    index=model_names.index(st.session_state.get("active_model_name", model_names[0])),
    horizontal=True,
)
st.session_state["active_model_name"] = inspect_model

# Recompute fairness results for newly selected active model
prep = st.session_state["prepared_data"]
primary_sens = st.session_state["sensitive_cols"][0]
st.session_state["fairness_results"] = compute_fairness_metrics(
    y_true=prep["y_test"],
    y_pred=st.session_state["model_predictions"][inspect_model],
    sensitive_features=prep["sensitive_test_df"][primary_sens],
    attribute_name=primary_sens,
)

inspect_sens = st.selectbox(
    "Select Demographic Attribute for Group Breakdown:",
    options=st.session_state["sensitive_cols"],
)

group_perf = perf_dict[inspect_model]["per_group"].get(inspect_sens, {})
if group_perf:
    table_rows = []
    for g_name, g_metrics in group_perf.items():
        table_rows.append({
            "Subgroup": g_name,
            "Sample Count": g_metrics["count"],
            "Selection Rate": f"{g_metrics['selection_rate']:.1%}",
            "Accuracy": f"{g_metrics['accuracy']:.1%}",
            "Precision": f"{g_metrics['precision']:.1%}",
            "Recall (TPR)": f"{g_metrics['recall']:.1%}",
            "False Positive Rate": f"{g_metrics['fpr']:.1%}",
            "F1-Score": f"{g_metrics['f1']:.3f}",
        })
    df_perf_table = pd.DataFrame(table_rows)
    try:
        st.dataframe(df_perf_table, use_container_width=True, hide_index=True)
    except Exception:
        st.table(df_perf_table)

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Inspect Fairness Report ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/5_Fairness_Report.py")
