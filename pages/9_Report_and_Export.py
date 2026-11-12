"""Page 9: Report & Export — Certified PDF Audit Report, Model Card, and Artifact Downloads."""

import streamlit as st
import pandas as pd
import io
import joblib
from core.state import (
    init_session_state, render_sidebar_progress, require_model_guard
)
from core.report import generate_pdf_report, generate_markdown_model_card
from ui.components import load_custom_css, render_metric_card

st.set_page_config(page_title="Report & Export | FairLens", page_icon="📑", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

require_model_guard("Report & Export")

active_model_name = st.session_state.get("active_model_name", "Random Forest")
model = st.session_state["trained_models"][active_model_name]
dataset_name = st.session_state.get("dataset_name", "Uploaded Dataset")
fairness = st.session_state["fairness_results"]
score = fairness["fairness_score"]

st.title("📑 Governance Report & Artifact Export")
st.markdown(
    f"<p style='color: #94a3b8;'>Generate certified compliance audit reports and model cards for <b>{active_model_name}</b>. "
    f"All artifacts are computed strictly from this session's real data. Zero fake outputs.</p>",
    unsafe_allow_html=True,
)

tab_pdf, tab_card, tab_download = st.tabs(["1. PDF Audit Report", "2. Model Card (Markdown)", "3. Export Model & Data"])

with tab_pdf:
    st.markdown("### 📄 One-Click PDF Fairness Audit Report")
    st.markdown(
        "Generate a publication-ready PDF containing executive summaries, quantitative fairness metrics, "
        "disaggregated demographic breakdowns, mitigation summaries, and legal recommendations."
    )

    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #6366f1;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">
                    Report Scope & Certification
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    • <b>Evaluated Model:</b> {active_model_name}<br>
                    • <b>Audited Dataset:</b> {dataset_name}<br>
                    • <b>Protected Attributes:</b> {', '.join(st.session_state.get('sensitive_cols', []))}<br>
                    • <b>Fairness Score:</b> {round(score):d}/100<br>
                    • <b>Disparate Impact:</b> {fairness['disparate_impact_ratio']:.3f} (80% Rule Compliance)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pdf2:
        try:
            pdf_bytes = generate_pdf_report(st.session_state)
            st.download_button(
                label="📥 Download PDF Audit Report",
                data=pdf_bytes,
                file_name=f"FairLens_Audit_Report_{active_model_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.caption("Generated with ReportLab (Strictly Session Data)")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

with tab_card:
    st.markdown("### 📇 Automated Model Card (Mitchell et al., 2019)")
    st.markdown(
        "Standardized model documentation detailing intended use, quantitative performance, "
        "ethical boundaries, and subgroup disparities."
    )

    model_card_md = generate_markdown_model_card(st.session_state)

    st.download_button(
        label="📥 Download Model Card (.md)",
        data=model_card_md,
        file_name=f"Model_Card_{active_model_name.replace(' ', '_')}.md",
        mime="text/markdown",
    )

    st.markdown("#### Model Card Preview")
    st.markdown(
        f"""
        <div class="glass-card" style="max-height: 480px; overflow-y: auto;">
            {model_card_md}
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_download:
    st.markdown("### 💾 Export Model & Evaluation Records")
    st.markdown("Download audited serialized models and evaluation datasets for independent auditing.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 1. Trained Model Pipeline (.pkl)")
        # Serialize model + preprocessor
        pipeline_export = {
            "model_name": active_model_name,
            "model": model,
            "preprocessor": st.session_state["preprocessor"],
            "feature_names": st.session_state["feature_names"],
            "target_col": st.session_state["target_col"],
            "positive_label": st.session_state["positive_label"],
            "sensitive_cols": st.session_state["sensitive_cols"],
            "fairness_score": score,
        }
        model_buffer = io.BytesIO()
        joblib.dump(pipeline_export, model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label=f"💾 Download {active_model_name} (.pkl)",
            data=model_buffer,
            file_name=f"fairlens_model_{active_model_name.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    with col_d2:
        st.markdown("#### 2. Evaluation Predictions (.csv)")
        # Build evaluation dataframe
        prep = st.session_state["prepared_data"]
        eval_df = prep["X_test_df"].copy()
        eval_df["Ground_Truth"] = st.session_state["y_test"]
        eval_df["Predicted_Outcome"] = st.session_state["model_predictions"][active_model_name]
        eval_df["Predicted_Probability"] = st.session_state["model_probabilities"][active_model_name]

        csv_buffer = eval_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📊 Download Audited Predictions (.csv)",
            data=csv_buffer,
            file_name=f"fairlens_evaluation_predictions_{active_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Ethical Limitations & Recommendations Strip
st.markdown("---")
st.markdown("### 🛡️ Ethical Limitations & Regulatory Recommendations")

col_l1, col_l2 = st.columns(2)
with col_l1:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">Auditing Limitations</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                • <b>Observational Bias:</b> Statistical parity does not resolve historical societal discrimination embedded in target labels.<br>
                • <b>Subgroup Sample Sizes:</b> Demographic categories with small counts (< 30) exhibit higher statistical variance.<br>
                • <b>Static Snapshot:</b> FairLens audits a point-in-time test split; production drift requires continual monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_l2:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">Actionable Recommendations</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                • <b>Human-in-the-Loop:</b> Implement manual review procedures for borderline scores near decision thresholds.<br>
                • <b>Proxy Mitigation:</b> Eliminate features displaying high proxy association with protected characteristics.<br>
                • <b>Continuous Auditing:</b> Re-run FairLens audits when retraining models or expanding demographic domains.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Next Action
st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns([3, 1])
with col_nav2:
    if st.button("Read About Responsible AI ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/10_About_Responsible_AI.py")

