"""Page 8: What-If Simulator — Interactive Record Counterfactual Flip Testing & Threshold Tuning."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.state import (
    init_session_state, render_sidebar_progress, require_model_guard
)
from core.counterfactual import (
    predict_single_instance, run_counterfactual_flip_test, compute_live_threshold_metrics
)
from core.utils import (
    apply_theme_to_figure, COLOR_FAIR, COLOR_BIASED, COLOR_CAUTION, COLOR_PRIMARY, COLOR_CYAN
)
from ui.components import (
    load_custom_css, render_metric_card
)

st.set_page_config(page_title="What-If Simulator | FairLens", page_icon="🧪", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_model_guard("What-If Simulator")

active_model_name = st.session_state.get("active_model_name", "Random Forest")
model = st.session_state["trained_models"][active_model_name]
preprocessor = st.session_state["preprocessor"]
raw_df = st.session_state["raw_df"]
target_col = st.session_state["target_col"]
positive_label = st.session_state["positive_label"]
sensitive_cols = st.session_state["sensitive_cols"]
exclude_cols = st.session_state.get("exclude_cols", [])

st.title("🧪 What-If Simulator & Counterfactual Flip Tester")
st.markdown(
    f"<p style='color: #94a3b8;'>Simulate real-time decisions for individual candidate profiles on <b>{active_model_name}</b>. "
    f"Test counterfactual perturbations by toggling sensitive attributes in isolation to detect discriminatory flips.</p>",
    unsafe_allow_html=True,
)

tab_cf, tab_thresh = st.tabs(["1. Counterfactual Profile Simulator", "2. Dynamic Decision Threshold Tuner"])

with tab_cf:
    st.markdown("### 👤 Build Candidate Profile")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.9rem;'>Enter feature values for a specific record. Default values are pre-filled using medians and modes from your uploaded dataset.</p>",
        unsafe_allow_html=True,
    )

    feature_cols = [c for c in raw_df.columns if c != target_col and c not in exclude_cols]

    # Dynamically build form inputs
    user_inputs = {}
    col_a, col_b = st.columns(2)

    for idx, col in enumerate(feature_cols):
        container = col_a if idx % 2 == 0 else col_b
        col_series = raw_df[col].dropna()

        with container:
            if pd.api.types.is_numeric_dtype(col_series):
                min_v = float(col_series.min())
                max_v = float(col_series.max())
                med_v = float(col_series.median())
                step = 1.0 if pd.api.types.is_integer_dtype(col_series) else (max_v - min_v) / 100.0
                step = max(0.01, step)
                user_inputs[col] = st.number_input(
                    f"{col} ({'Sensitive' if col in sensitive_cols else 'Feature'})",
                    min_value=min_v,
                    max_value=max_v,
                    value=med_v,
                    step=step,
                )
            else:
                uniques = [str(x) for x in col_series.unique()]
                mode_v = str(col_series.mode()[0]) if not col_series.empty else uniques[0]
                user_inputs[col] = st.selectbox(
                    f"{col} ({'Sensitive' if col in sensitive_cols else 'Feature'})",
                    options=uniques,
                    index=uniques.index(mode_v) if mode_v in uniques else 0,
                )

    test_sens_col = st.selectbox(
        "Select Protected Demographic Attribute to Stress-Test:",
        options=sensitive_cols,
    )

    # Prediction & Counterfactual Run
    if st.button("⚡ Test Profile & Run Counterfactual Flip Test", type="primary", use_container_width=True):
        current_thresh = st.session_state.get("decision_threshold", 0.5)
        
        # 1. Base Prediction
        base_res = predict_single_instance(model, preprocessor, user_inputs, threshold=current_thresh)
        
        # 2. Counterfactual test
        all_groups = [str(x) for x in raw_df[test_sens_col].dropna().unique()]
        cf_res = run_counterfactual_flip_test(
            model=model,
            preprocessor=preprocessor,
            base_instance=user_inputs,
            sensitive_col=test_sens_col,
            all_groups=all_groups,
            threshold=current_thresh,
        )

        st.markdown("---")
        st.markdown("### 🎯 Decision & Counterfactual Audit Results")

        col_dec1, col_dec2, col_dec3 = st.columns(3)
        dec_color = COLOR_FAIR if base_res["decision"] == 1 else COLOR_BIASED
        dec_text = f"APPROVED ({positive_label})" if base_res["decision"] == 1 else "DENIED (Alternative)"

        with col_dec1:
            render_metric_card(
                title="Model Decision",
                value=dec_text,
                subtext=f"Cutoff Threshold: {current_thresh:.2f}",
                status="fair" if base_res["decision"] == 1 else "caution",
            )
        with col_dec2:
            render_metric_card(
                title="Predicted Favorable Probability",
                value=f"{base_res['probability']:.1%}",
                subtext="P(Outcome = Favorable)",
                status="fair" if base_res["probability"] >= current_thresh else "biased",
            )
        with col_dec3:
            cf_status = "biased" if cf_res["has_decision_flip"] else "fair"
            cf_text = "DECISION FLIPPED ⚠️" if cf_res["has_decision_flip"] else "ROBUST (NO FLIP) ✅"
            render_metric_card(
                title="Counterfactual Flip Test",
                value=cf_text,
                subtext=f"Tested across {len(all_groups)} groups",
                status=cf_status,
                tooltip="Warns if changing ONLY the sensitive attribute flips the model decision.",
            )

        # Counterfactual Flip Alert
        if cf_res["has_decision_flip"]:
            st.error(
                f"🚨 **Direct Counterfactual Bias Detected!** Keeping all qualifications identical and changing "
                f"only the demographic group for `{test_sens_col}` caused the model to flip its decision. "
                "This indicates individual discrimination for this profile.",
                icon="🚨",
            )
        else:
            st.success(
                f"✅ **Individual Counterfactual Consistency Confirmed!** Changing `{test_sens_col}` across all "
                f"{len(all_groups)} demographic groups maintained the identical decision for this candidate profile.",
                icon="✅",
            )

        # Counterfactual Variations Table
        st.markdown("#### Subgroup Perturbation Matrix (All other features identical)")
        var_rows = []
        for v in cf_res["variations"]:
            var_rows.append({
                "Demographic Group": v["group"],
                "Is Current Input": "✓ Original" if v["is_original"] else "",
                "Model Decision": "APPROVED" if v["decision"] == 1 else "DENIED",
                "Probability": f"{v['probability']:.1%}",
                "Probability Delta": f"{v['prob_delta']:+.2%}",
                "Decision Flipped?": "⚠️ FLIP" if v["flipped"] else "Consistent",
            })
        df_var_tab = pd.DataFrame(var_rows)
        try:
            st.dataframe(df_var_tab, use_container_width=True, hide_index=True)
        except Exception:
            st.table(df_var_tab)

with tab_thresh:
    st.markdown("### 🎚️ Live Decision Threshold Optimizer")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.9rem;'>Moving the classification threshold dynamically changes both accuracy and fairness metrics. "
        "Find the threshold that balances acceptance rates across demographic groups.</p>",
        unsafe_allow_html=True,
    )

    y_test = st.session_state["y_test"]
    y_prob = st.session_state["model_probabilities"][active_model_name]
    sens_series = st.session_state["sensitive_test"][test_sens_col]

    threshold_slider = st.slider(
        "Decision Cutoff Threshold (Default 0.50)",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.get("decision_threshold", 0.50)),
        step=0.02,
        help="Adjust the minimum predicted probability required to assign a favorable decision.",
    )
    st.session_state["decision_threshold"] = threshold_slider

    # Compute live threshold metrics
    live_m = compute_live_threshold_metrics(
        y_test=y_test,
        y_prob=y_prob,
        sensitive_series=sens_series,
        threshold=threshold_slider,
        attribute_name=test_sens_col,
    )

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        render_metric_card(
            title="Accuracy at Threshold",
            value=f"{live_m['accuracy']:.1%}",
            subtext=f"Threshold = {threshold_slider:.2f}",
            status="fair",
        )
    with col_t2:
        render_metric_card(
            title="Fairness Score",
            value=f"{round(live_m['fairness_score']):d}/100",
            subtext="Live composite equity",
            status="fair" if live_m['fairness_score'] >= 80 else ("caution" if live_m['fairness_score'] >= 60 else "biased"),
        )
    with col_t3:
        render_metric_card(
            title="Disparate Impact Ratio",
            value=f"{live_m['disparate_impact_ratio']:.3f}",
            subtext="Target ≥ 0.800",
            status="fair" if live_m['disparate_impact_ratio'] >= 0.80 else "biased",
        )
    with col_t4:
        render_metric_card(
            title="Demographic Parity Diff",
            value=f"{live_m['demographic_parity_diff']:.1%}",
            subtext="Selection gap",
            status="fair" if live_m['demographic_parity_diff'] <= 0.10 else "biased",
        )

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Generate Audit Report & Export ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/9_Report_and_Export.py")
