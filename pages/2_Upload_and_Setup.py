"""Page 2: Upload & Setup — Secure Dataset Ingestion, Type Inference, and Schema Configuration."""

import streamlit as st
import pandas as pd
import joblib
from core.state import (
    init_session_state, render_sidebar_progress, reset_derived_state,
    is_data_loaded, is_setup_configured
)
from core.data_loader import (
    load_uploaded_file, detect_column_types, validate_setup_configuration,
    DataValidationError
)
from core.audit import compute_data_audit
from ui.components import load_custom_css

st.set_page_config(page_title="Upload & Setup | FairLens", page_icon="📂", layout="wide")
load_custom_css()
init_session_state()
render_sidebar_progress()

st.title("📂 Upload & Configure Dataset")
st.markdown(
    "<p style='color: #94a3b8;'>Upload your tabular data (CSV or Excel) and configure your target and protected attributes. "
    "All computations are executed locally in memory.</p>",
    unsafe_allow_html=True,
)

tab_dataset, tab_model = st.tabs(["1. Dataset Ingestion", "2. (Optional) Custom Model Upload (.pkl)"])

with tab_dataset:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload a tabular dataset containing historical records, a binary target, and demographic features.",
    )

    if uploaded_file is not None:
        try:
            # Check if this is a newly uploaded file or previously stored
            if st.session_state.get("dataset_name") != uploaded_file.name:
                with st.spinner("Parsing and validating uploaded dataset..."):
                    df = load_uploaded_file(uploaded_file, uploaded_file.name)
                    reset_derived_state()
                    st.session_state["raw_df"] = df
                    st.session_state["dataset_name"] = uploaded_file.name
                st.success(f"Successfully loaded `{uploaded_file.name}` ({len(df):,} rows × {len(df.columns)} columns)!")
            else:
                df = st.session_state["raw_df"]

            # Quick Dataset Preview
            st.markdown("#### 🔍 Data Preview (First 5 Rows)")
            try:
                st.dataframe(df.head(5), use_container_width=True)
            except Exception:
                st.table(df.head(5))

            # Auto-detect column types
            detected = detect_column_types(df)

            st.markdown("---")
            st.markdown("### ⚙️ Schema Configuration")

            col1, col2 = st.columns(2)

            with col1:
                # Target column selector
                target_options = detected["candidate_targets"] + [
                    c for c in detected["all_cols"] if c not in detected["candidate_targets"]
                ]
                default_target_idx = 0 if target_options else 0
                selected_target = st.selectbox(
                    "🎯 Target Column (Outcome to Predict)",
                    options=target_options,
                    index=default_target_idx,
                    help="Select the binary outcome column (e.g. Loan_Approved, Hired, Readmitted).",
                )

                # Positive outcome label selector
                if selected_target:
                    unique_labels = list(df[selected_target].dropna().unique())
                    # Convert to string for consistent selection
                    label_options = [str(x) for x in unique_labels]
                    
                    # Try to infer positive label (1, 'Yes', 'Approved', 'Pass', 'True')
                    default_pos_idx = 0
                    pos_candidates = ["1", "yes", "approved", "pass", "true", "good", "granted", "hired"]
                    for idx, opt in enumerate(label_options):
                        if opt.lower() in pos_candidates:
                            default_pos_idx = idx
                            break

                    selected_pos_label = st.selectbox(
                        "✅ Favorable (Positive) Outcome Label",
                        options=label_options,
                        index=default_pos_idx,
                        help="The label representing the favorable or preferred decision (e.g., 1, 'Approved').",
                    )
                else:
                    selected_pos_label = None

            with col2:
                # Sensitive attributes selector (multi-select)
                sens_options = [
                    c for c in (detected["candidate_sensitive"] + detected["all_cols"])
                    if c != selected_target
                ]
                # Deduplicate while preserving order
                sens_options = list(dict.fromkeys(sens_options))
                default_sens = [sens_options[0]] if sens_options else []
                # Ensure default_sens is strictly inside sens_options
                default_sens = [s for s in default_sens if s in sens_options]

                selected_sensitive = st.multiselect(
                    "🛡️ Sensitive / Protected Demographic Attribute(s)",
                    options=sens_options,
                    default=default_sens,
                    help="Attributes representing legally protected groups (e.g. Gender, Age, Race, Religion).",
                )

                # Columns to exclude from modeling
                exclude_candidates = [
                    c for c in detected["all_cols"]
                    if c != selected_target and c not in selected_sensitive
                ]
                # Auto-exclude obvious IDs
                auto_excludes = [
                    c for c in exclude_candidates
                    if any(id_kw in c.lower() for id_kw in ["id", "index", "ssn", "uuid", "account_no", "applicant_id"])
                ]
                auto_excludes = [c for c in auto_excludes if c in exclude_candidates]

                selected_exclude = st.multiselect(
                    "🚫 Exclude from Model Training (e.g. IDs, Unusable Columns)",
                    options=exclude_candidates,
                    default=auto_excludes,
                    help="Columns that should not be used by the ML algorithms as training features.",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Validation & Commit Button
            if st.button("💾 Confirm Setup & Run Data Audit", type="primary", use_container_width=True):
                is_valid, err_msg = validate_setup_configuration(
                    df=df,
                    target_col=selected_target,
                    positive_label=selected_pos_label,
                    sensitive_cols=selected_sensitive,
                    exclude_cols=selected_exclude,
                )

                if not is_valid:
                    st.error(f"❌ Configuration Error: {err_msg}")
                else:
                    # Save configurations
                    st.session_state["target_col"] = selected_target
                    st.session_state["positive_label"] = selected_pos_label
                    st.session_state["sensitive_cols"] = selected_sensitive
                    st.session_state["exclude_cols"] = selected_exclude

                    # Compute baseline data audit immediately
                    with st.spinner("Performing pre-training demographic disparity audit..."):
                        audit_results = compute_data_audit(
                            df=df,
                            target_col=selected_target,
                            positive_label=selected_pos_label,
                            sensitive_cols=selected_sensitive,
                        )
                        st.session_state["data_audit_results"] = audit_results

                    st.success("Setup validated and data audit computed successfully!")
                    st.switch_page("pages/3_Data_Audit.py")

        except DataValidationError as dve:
            st.error(f"❌ {str(dve)}")
        except Exception as ex:
            st.error(f"❌ An error occurred while processing the file: {str(ex)}")

    else:
        # Clean empty state when no file uploaded yet
        st.info("👆 Please upload a CSV or Excel file to get started.", icon="💡")

with tab_model:
    st.markdown("#### Upload a Pre-Trained Model (.pkl)")
    st.markdown(
        "If you already have a trained scikit-learn model, you can evaluate it directly against your test dataset."
    )
    model_file = st.file_uploader("Upload serialized model (.pkl)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            user_model = joblib.load(model_file)
            st.session_state["user_uploaded_model"] = user_model
            st.success("Trained model uploaded successfully! Make sure to also upload matching test data in Tab 1.")
        except Exception as e:
            st.error(f"Failed to load model file: {str(e)}")
