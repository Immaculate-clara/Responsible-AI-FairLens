"""Page 10: About Responsible AI — Educational Reference on Ethics, Accountability, and AI Governance."""

import streamlit as st
from core.state import init_session_state, render_sidebar_progress
from ui.components import load_custom_css

st.set_page_config(page_title="About Responsible AI | FairLens", page_icon="📚", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

st.title("📚 Principles of Responsible & Equitable AI")
st.markdown(
    "<p style='color: #94a3b8;'>A foundational reference on fairness definitions, transparency guidelines, "
    "and statutory frameworks governing algorithmic decision systems.</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["1. Core Pillars of Responsible AI", "2. Statutory & Regulatory Landscape", "3. Fairness Metrics Glossary"])

with tab1:
    st.markdown("### The Four Pillars of Responsible AI")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #10b981;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">
                    1. Fairness & Equity
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    AI systems should treat individuals and demographic subgroups equitably. Models must not propagate 
                    unjustified disparities across legally protected characteristics such as race, sex, age, disability, 
                    or religion. True fairness entails proactive auditing across both representation and error parity.
                </div>
            </div>
            
            <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">
                    2. Transparency & Explainability
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    Stakeholders have a fundamental right to understand the factors driving algorithmic decisions. 
                    Employing feature attribution tools (such as SHAP values) and standardized documentation (Model Cards) 
                    ensures decisions are interpretable, contestable, and reproducible.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #f59e0b;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">
                    3. Accountability & Human Agency
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    Autonomous algorithms must never operate in a vacuum. Organizations deploying predictive models 
                    must retain clear human oversight, establish audit trails, and maintain escalation pathways 
                    when automated systems make erroneous or discriminatory calls.
                </div>
            </div>
            
            <div class="glass-card" style="border-left: 4px solid #a855f7;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">
                    4. Privacy & Data Governance
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    Auditing models for fairness requires handling sensitive demographic information. 
                    Responsible engineering demands local computation, strict data minimization, secure parameterization, 
                    and safeguarding user privacy throughout the testing lifecycle.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown("### Statutory & Regulatory Frameworks")
    st.markdown(
        """
        Global regulators have established clear enforcement mechanisms regarding algorithmic bias:
        
        #### 1. United States: EEOC Four-Fifths (80%) Rule
        Under the US Equal Employment Opportunity Commission (EEOC) Uniform Guidelines on Employee Selection Procedures:
        > *"A selection rate for any race, sex, or ethnic group which is less than four-fifths (4/5) or 80 percent of the rate for the group with the highest rate will generally be regarded by the Federal enforcement agencies as evidence of adverse impact."*
        
        #### 2. European Union: The EU AI Act (2024)
        The EU AI Act classifies predictive models in employment, education, credit scoring, law enforcement, and critical infrastructure as **High-Risk AI Systems**. High-risk systems face mandatory requirements:
        - Robust risk management and governance systems.
        - Data quality criteria to mitigate discriminatory biases.
        - Detailed technical documentation and post-market monitoring.
        - Continuous human oversight.
        
        #### 3. US Federal Trade Commission (FTC) Enforcement
        The FTC enforces Section 5 of the FTC Act against unfair or deceptive algorithmic practices, warning organizations that claiming an algorithm is an objective 'black box' does not exempt it from anti-discrimination statutes.
        """
    )

with tab3:
    st.markdown("### Algorithmic Fairness Metrics Glossary")
    st.markdown(
        """
        | Metric Name | Mathematical Concept | Regulatory Interpretation |
        | :--- | :--- | :--- |
        | **Disparate Impact Ratio (DIR)** | $\\frac{\\min_g P(\\hat{Y}=1 \\mid A=g)}{\\max_g P(\\hat{Y}=1 \\mid A=g)}$ | Tests the EEOC 80% rule. Ratios below 0.80 indicate potential adverse impact. |
        | **Demographic Parity Difference** | $\\max_g P(\\hat{Y}=1 \\mid A=g) - \\min_g P(\\hat{Y}=1 \\mid A=g)$ | Measures the absolute divergence in favorable outcome rates between subgroups. |
        | **Equal Opportunity Difference** | $\\max_g \\text{TPR}_g - \\min_g \\text{TPR}_g$ | Measures whether qualified members across different groups have equal chances of approval. |
        | **Equalized Odds Difference** | $\\max(\\Delta \\text{TPR}, \\Delta \\text{FPR})$ | Requires both True Positive Rate and False Positive Rate to be balanced across groups. |
        | **Counterfactual Fairness** | $P(\\hat{Y}_{A \\leftarrow a} = 1) = P(\\hat{Y}_{A \\leftarrow a'} = 1)$ | A decision for an individual is fair if it remains identical in a counterfactual world where their sensitive group is changed. |
        """
    )

st.markdown("---")
col_ret1, col_ret2 = st.columns([3, 1])
with col_ret2:
    if st.button("⬅️ Return to Upload & Setup", type="primary", use_container_width=True):
        st.switch_page("pages/2_Upload_and_Setup.py")

