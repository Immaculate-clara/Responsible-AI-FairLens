"""Responsible AI: FairLens — Main Entry Point

Tagline: 'See the bias before it decides.'
Production-quality Bias & Fairness Checker for AI models.
"""

import streamlit as st
from core.state import init_session_state, render_sidebar_progress, is_data_loaded
from ui.components import load_custom_css

st.set_page_config(
    page_title="Responsible AI: FairLens",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load global styling and initialize state
load_custom_css()
init_session_state()
render_sidebar_progress()

# Main landing display if executed directly
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Responsible AI: FairLens</div>
        <div class="hero-tagline">See the bias before it decides.</div>
        <div class="hero-pitch">
            An enterprise-grade fairness audit and mitigation platform. 
            Upload your dataset or model to inspect demographic disparities, 
            diagnose proxy variables, simulate counterfactuals, and generate compliant audit reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    if st.button("🚀 Launch Audit Suite", type="primary", use_container_width=True):
        st.switch_page("pages/2_Upload_and_Setup.py")

# Quick Overview Strip
st.markdown(
    """
    <div class="step-container">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Upload & Detect</div>
            <div class="step-desc">
                Bring your CSV or Excel dataset. FairLens maps demographic attributes, 
                flags small sample groups, and tests raw baseline disparities.
            </div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Audit & Explain</div>
            <div class="step-desc">
                Evaluate models across standard metrics (Demographic Parity, 80% Rule, Equal Opportunity). 
                Uncover root causes with SHAP and proxy detectors.
            </div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Mitigate & Report</div>
            <div class="step-desc">
                Reduce bias through pre-, in-, or post-processing algorithms. 
                Simulate counterfactuals and export a certified PDF audit report.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cardinal Rule Notice
st.info(
    "🔒 **Data Privacy & Integrity**: FairLens contains **zero sample or pre-loaded demo data**. "
    "Every metric, chart, and report is computed dynamically from data you upload in this private session.",
    icon="ℹ️"
)

