"""Page 6: Why Is It Biased? — SHAP Attribution, Subgroup Comparison, and Proxy Feature Detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.state import (
    init_session_state, render_sidebar_progress, require_model_guard
)
from core.explain import (
    compute_shap_explanations, detect_proxy_features, create_shap_summary_chart
)
from core.utils import (
    apply_theme_to_figure, COLOR_PRIMARY, COLOR_CYAN, COLOR_BIASED, COLOR_CAUTION, COLOR_FAIR
)
from ui.components import load_custom_css, render_metric_card

st.set_page_config(page_title="Why Is It Biased? | FairLens", page_icon="🔍", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_model_guard("Explainability & Bias Root Cause")

active_model_name = st.session_state.get("active_model_name", "Evaluated Model")
model = st.session_state["trained_models"][active_model_name]
prepared_data = st.session_state["prepared_data"]
feature_names = st.session_state["feature_names"]
sens_test_df = st.session_state["sensitive_test"]

st.title("🔍 Why Is It Biased? Root Cause Diagnostics")
st.markdown(
    f"<p style='color: #94a3b8;'>Examine feature importance via SHAP values, compare how features impact different demographic groups, "
    f"and uncover hidden proxy variables for <b>{active_model_name}</b>.</p>",
    unsafe_allow_html=True,
)

# Crucial Regulatory & Technical Alert: Fairness Through Unawareness Fallacy
st.warning(
    "⚠️ **The 'Fairness Through Unawareness' Fallacy**: Simply deleting the protected demographic feature "
    "(e.g. Race or Gender) from your training set **does NOT remove bias**. Machine learning algorithms easily "
    "reconstruct protected attributes from correlated proxy variables (e.g. zip code, education, browsing history). "
    "Inspect the Proxy Detector below to identify potential proxies in your data.",
    icon="⚠️",
)

tab_shap, tab_proxy = st.tabs(["1. SHAP Feature Attribution", "2. Proxy Feature Detector"])

with tab_shap:
    st.markdown("### 🌐 Global Feature Importance (SHAP)")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.9rem;'>Measures each feature's average absolute impact on pushing the model toward or away from a positive outcome.</p>",
        unsafe_allow_html=True,
    )

    # Compute or retrieve cached SHAP results
    primary_sens = st.session_state["sensitive_cols"][0]
    if st.session_state.get("shap_results") is None:
        with st.spinner("Computing SHAP explainer values across evaluation records..."):
            shap_res = compute_shap_explanations(
                model=model,
                X_train_proc=st.session_state["X_train"],
                X_test_proc=st.session_state["X_test"],
                feature_names=feature_names,
                sensitive_test_series=sens_test_df[primary_sens],
            )
            st.session_state["shap_results"] = shap_res
    else:
        shap_res = st.session_state["shap_results"]

    # Global SHAP Chart
    fig_shap = create_shap_summary_chart(shap_res["global_importance"], top_n=10)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # Subgroup SHAP Comparison
    st.markdown("### 👥 Subgroup Feature Impact Divergence")
    st.markdown(
        f"<p style='color: #94a3b8; font-size: 0.9rem;'>Compare how the top features drive decisions differently for each subgroup of <code>{primary_sens}</code>.</p>",
        unsafe_allow_html=True,
    )

    top_feats = shap_res["top_features"][:5]
    groups = shap_res["unique_groups"]

    if len(groups) >= 2:
        group_comp_data = []
        for feat in top_feats:
            row = {"Feature": feat}
            for g in groups:
                # Find importance for this feature in this group
                g_attrs = shap_res["group_attributions"].get(g, [])
                val = next((item["importance"] for item in g_attrs if item["feature"] == feat), 0.0)
                row[f"Group '{g}' Impact"] = f"{val:.4f}"
            group_comp_data.append(row)

        df_group_comp = pd.DataFrame(group_comp_data)
        try:
            st.dataframe(df_group_comp, use_container_width=True, hide_index=True)
        except Exception:
            st.table(df_group_comp)

        # Plotly grouped bar chart
        fig_subgroup = go.Figure()
        for idx, g in enumerate(groups):
            g_attrs = shap_res["group_attributions"].get(g, [])
            vals = [next((item["importance"] for item in g_attrs if item["feature"] == f), 0.0) for f in top_feats]
            fig_subgroup.add_trace(go.Bar(
                x=top_feats,
                y=vals,
                name=f"Subgroup: {g}",
            ))

        fig_subgroup.update_layout(
            title=f"Feature Impact Comparison Across Subgroups of {primary_sens}",
            barmode="group",
            xaxis_title="Top Features",
            yaxis_title="Mean |SHAP Value|",
            height=360,
        )
        st.plotly_chart(apply_theme_to_figure(fig_subgroup), use_container_width=True)

with tab_proxy:
    st.markdown("### 🕵️ Proxy Feature Detector")
    st.markdown(
        "<p style='color: #cbd5e1; font-size: 0.9rem;'>FairLens tests statistical mutual dependency between non-sensitive features "
        "and protected demographic columns. Features with high correlation scores allow the model to infer protected status indirectly.</p>",
        unsafe_allow_html=True,
    )

    raw_df = st.session_state["raw_df"]
    target_col = st.session_state["target_col"]
    sensitive_cols = st.session_state["sensitive_cols"]

    scol_for_proxy = st.selectbox(
        "Select Sensitive Attribute to Scan Against:",
        options=sensitive_cols,
    )

    candidate_cols = [c for c in raw_df.columns if c != target_col and c != scol_for_proxy]

    with st.spinner(f"Scanning {len(candidate_cols)} candidate features for proxy associations..."):
        proxies = detect_proxy_features(
            df=raw_df,
            sensitive_col=scol_for_proxy,
            feature_cols=candidate_cols,
        )
        st.session_state["proxy_features"] = proxies

    # Display Top Proxy Results
    col_p1, col_p2, col_p3 = st.columns(3)
    high_proxies = [p for p in proxies if p["status"] == "biased"]
    mod_proxies = [p for p in proxies if p["status"] == "caution"]

    with col_p1:
        render_metric_card(
            title="High Proxy Risk Features",
            value=f"{len(high_proxies)}",
            subtext="Score ≥ 0.50 with protected attribute",
            status="biased" if high_proxies else "fair",
        )
    with col_p2:
        render_metric_card(
            title="Moderate Proxies",
            value=f"{len(mod_proxies)}",
            subtext="Score 0.25 - 0.49 with protected attribute",
            status="caution" if mod_proxies else "fair",
        )
    with col_p3:
        render_metric_card(
            title="Total Scanned Features",
            value=f"{len(proxies)}",
            subtext=f"Tested against {scol_for_proxy}",
            status="fair",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Proxy Dependency Table")
    
    proxy_table = [
        {
            "Feature Name": p["feature"],
            "Data Type": p["dtype"],
            "Association Score (0-1)": f"{p['correlation_score']:.3f}",
            "Risk Assessment": p["risk"],
        }
        for p in proxies
    ]
    df_proxy_tab = pd.DataFrame(proxy_table)
    try:
        st.dataframe(df_proxy_tab, use_container_width=True, hide_index=True)
    except Exception:
        st.table(df_proxy_tab)

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Fix It (Mitigate Bias) ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/7_Fix_It.py")
