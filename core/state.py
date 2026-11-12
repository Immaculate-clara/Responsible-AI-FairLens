"""Session state schema, page guards, and pipeline progress tracker for FairLens."""

from typing import Dict, Any, List, Optional
import streamlit as st

PIPELINE_STAGES = [
    ("Upload & Setup", "2_Upload_and_Setup.py"),
    ("Data Audit", "3_Data_Audit.py"),
    ("Model Performance", "4_Model_Performance.py"),
    ("Fairness Report", "5_Fairness_Report.py"),
    ("Why Is It Biased?", "6_Why_Is_It_Biased.py"),
    ("Fix It (Mitigation)", "7_Fix_It.py"),
    ("What-If Simulator", "8_What_If_Simulator.py"),
    ("Report & Export", "9_Report_and_Export.py"),
]


def init_session_state() -> None:
    """Initialize all session state keys with default empty values."""
    defaults: Dict[str, Any] = {
        # Raw Data & Configurations
        "raw_df": None,
        "dataset_name": None,
        "target_col": None,
        "positive_label": None,
        "sensitive_cols": [],
        "exclude_cols": [],
        "user_uploaded_model": None,

        # Processed Data & Split
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "sensitive_test": None,
        "preprocessor": None,
        "feature_names": [],

        # Trained Models & Predictions
        "trained_models": {},      # {model_name: trained_model_object}
        "model_predictions": {},   # {model_name: y_pred_array}
        "model_probabilities": {}, # {model_name: y_prob_array}
        "active_model_name": None,

        # Audits & Metrics
        "data_audit_results": None,
        "model_performance_results": {},
        "fairness_results": {},
        "intersectional_results": None,

        # Explainability & Mitigations
        "shap_results": None,
        "proxy_features": None,
        "mitigation_results": [],   # List of dicts: {name, model, metrics, fairness}
        "selected_mitigated_model": None,

        # What-If
        "what_if_record": None,
        "decision_threshold": 0.5,

        # UI & Flow
        "theme": "dark",
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_derived_state() -> None:
    """Reset all downstream ML models, fairness calculations, and reports when new data is uploaded."""
    keys_to_clear = [
        "X_train", "X_test", "y_train", "y_test", "sensitive_test",
        "preprocessor", "feature_names", "trained_models",
        "model_predictions", "model_probabilities", "active_model_name",
        "data_audit_results", "model_performance_results", "fairness_results",
        "intersectional_results", "shap_results", "proxy_features",
        "mitigation_results", "selected_mitigated_model", "what_if_record",
        "decision_threshold"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None if not isinstance(st.session_state[key], (list, dict)) else ([] if isinstance(st.session_state[key], list) else {})
    st.session_state["decision_threshold"] = 0.5


def is_data_loaded() -> bool:
    """Check if raw dataset is present in session state."""
    return (
        st.session_state.get("raw_df") is not None
        and len(st.session_state.get("raw_df", [])) > 0
    )


def is_setup_configured() -> bool:
    """Check if target and sensitive attributes have been configured."""
    return (
        is_data_loaded()
        and st.session_state.get("target_col") is not None
        and len(st.session_state.get("sensitive_cols", [])) > 0
    )


def is_model_trained() -> bool:
    """Check if at least one model has been trained or evaluated."""
    return bool(
        st.session_state.get("trained_models")
        and len(st.session_state["trained_models"]) > 0
    )


def is_fairness_evaluated() -> bool:
    """Check if fairness assessment has been completed."""
    return bool(
        st.session_state.get("fairness_results")
        and len(st.session_state["fairness_results"]) > 0
    )


def require_data_guard(stage_name: str = "this analysis") -> None:
    """
    Ensure dataset is loaded and setup is configured.
    If not, renders a clean empty state card and stops execution.
    """
    if not is_data_loaded() or not is_setup_configured():
        from ui.components import render_empty_state
        render_empty_state(
            title="Upload a dataset to begin",
            description=(
                f"Before accessing {stage_name}, please upload a dataset (CSV or Excel) "
                "and select your target and sensitive attributes in Setup."
            ),
            action_label="Go to Upload & Setup",
            target_page="pages/2_Upload_and_Setup.py",
        )
        st.stop()


def require_model_guard(stage_name: str = "this section") -> None:
    """
    Ensure models have been trained or uploaded before proceeding.
    If not, renders a clean empty state card and stops execution.
    """
    require_data_guard(stage_name)
    if not is_model_trained():
        from ui.components import render_empty_state
        render_empty_state(
            title="Train a model first",
            description=(
                f"To view {stage_name}, train your baseline models on the Model Performance page. "
                "FairLens requires trained model predictions to evaluate disparities."
            ),
            action_label="Go to Model Performance",
            target_page="pages/4_Model_Performance.py",
        )
        st.stop()


def render_sidebar_progress() -> None:
    """Render the interactive workflow progress tracker in Streamlit sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.3rem;">⚖️</span>
                    <div>
                        <div style="font-weight: 800; font-size: 1.1rem; color: #ffffff;">FairLens</div>
                        <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">Responsible AI Audit Suite</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Pipeline Progress</p>", unsafe_allow_html=True)

        stages = [
            ("1. Data Setup", is_setup_configured()),
            ("2. Data Audit", st.session_state.get("data_audit_results") is not None),
            ("3. Model Training", is_model_trained()),
            ("4. Fairness Evaluation", is_fairness_evaluated()),
            ("5. Root Cause (SHAP)", st.session_state.get("shap_results") is not None),
            ("6. Mitigation Applied", bool(st.session_state.get("mitigation_results"))),
            ("7. Report Generated", bool(st.session_state.get("fairness_results"))),
        ]

        html_stages = "<div class='sidebar-pipeline'>"
        for name, completed in stages:
            dot_class = "completed" if completed else ""
            status_text = "✓ Ready" if completed else "○ Pending"
            color = "#10b981" if completed else "#64748b"
            html_stages += f"""
            <div class="pipeline-step" style="display:flex; justify-content:space-between; align-items:center; padding: 4px 6px;">
                <div style="display:flex; align-items:center; gap: 8px;">
                    <span class="step-dot {dot_class}"></span>
                    <span style="color: {'#e2e8f0' if completed else '#94a3b8'}; font-size: 0.82rem;">{name}</span>
                </div>
                <span style="color: {color}; font-size: 0.72rem; font-weight: 600;">{status_text}</span>
            </div>
            """
        html_stages += "</div>"
        st.markdown(html_stages, unsafe_allow_html=True)

        # Dataset status summary badge in sidebar
        if is_data_loaded():
            df = st.session_state["raw_df"]
            st.markdown(
                f"""
                <div style="margin-top: 1.5rem; padding: 0.8rem; background: rgba(19, 27, 46, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;">
                    <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Active Dataset</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; word-break: break-all;">{st.session_state.get('dataset_name', 'User Dataset')}</div>
                    <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 4px;">{len(df):,} rows × {len(df.columns)} columns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="margin-top: 1.5rem; padding: 0.8rem; background: rgba(19, 27, 46, 0.4); border: 1px dashed rgba(255,255,255,0.12); border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">No active dataset loaded</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

