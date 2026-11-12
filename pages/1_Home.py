"""Page 1: Home — Landing Page & Platform Overview."""

import streamlit as st
from core.state import init_session_state, render_sidebar_progress, is_data_loaded
from ui.components import load_custom_css

st.set_page_config(page_title="Home | FairLens", page_icon="⚖️", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

# Hero Section
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Responsible AI: FairLens</div>
        <div class="hero-tagline">"See the bias before it decides."</div>
        <div class="hero-pitch">
            An open, principled audit suite designed to evaluate, diagnose, and mitigate algorithmic bias in predictive machine learning models. 
            Ensure demographic fairness, legal compliance (EEOC 80% Rule), and explainable transparency for mission-critical AI systems.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🚀 Start Bias Audit Now", type="primary", use_container_width=True):
        st.switch_page("pages/2_Upload_and_Setup.py")

st.markdown("<br>", unsafe_allow_html=True)

# 3-Step "How It Works" Strip
st.markdown("### 🛠️ How FairLens Works")
st.markdown(
    """
    <div class="step-container">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Upload & Map</div>
            <div class="step-desc">
                Provide your historical tabular dataset (CSV or Excel) or pre-trained model. 
                Identify protected demographic attributes (e.g., gender, age, race, region) 
                and your binary favorable target decision.
            </div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Audit & Diagnose</div>
            <div class="step-desc">
                Inspect raw historical disparities and evaluate model predictions against statutory standards 
                (Disparate Impact Ratio, Demographic Parity, Equalized Odds). 
                Detect covert proxy variables and explain disparities via SHAP.
            </div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Mitigate & Govern</div>
            <div class="step-desc">
                Apply pre-processing (reweighing), in-processing (exponentiated gradient), 
                or post-processing (threshold optimization). Verify trade-offs, run counterfactual tests, 
                and export audit-ready PDF reports and Model Cards.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Why Fairness Matters in Machine Learning
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### ⚖️ Why Algorithmic Fairness Matters")
    st.markdown(
        """
        Automated decision-making systems increasingly govern life-altering opportunities—such as 
        loan approvals, hiring selections, insurance underwriting, and criminal risk assessments.
        
        When models are trained on historical data, they often inherit and amplify existing societal disparities. 
        Unchecked, this leads to:
        - **Systemic Disparate Impact:** Unlawful discrimination against protected demographic groups.
        - **Reputational & Regulatory Liability:** Non-compliance with EEOC guidelines, the EU AI Act, and FTC enforcement.
        - **The Proxy Trap:** Simply hiding demographic features does not fix bias; models easily learn proxy correlations from other attributes.
        """
    )

with col_right:
    st.markdown("### 🔒 Zero Demo Data Commitment")
    st.markdown(
        """
        FairLens operates under a strict **Zero Synthetic/Demo Data** standard:
        - We do **not** ship with pre-loaded toy datasets or placeholder figures.
        - Every statistic, chart, SHAP summary, and PDF line is computed dynamically from **your uploaded data**.
        - If a group has fewer than 30 records, FairLens alerts you to small-sample uncertainty rather than faking confidence.
        
        *Your data remains in your local browser session and is never transmitted to third-party APIs.*
        """
    )

