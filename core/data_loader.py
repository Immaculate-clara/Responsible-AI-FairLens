"""Safe dataset loading, column type inference, and setup validation for FairLens."""

from typing import Dict, List, Tuple, Any, Optional, Union
import io
import pandas as pd
import numpy as np


class DataValidationError(Exception):
    """Custom exception for user-friendly validation error messages."""
    pass


def load_uploaded_file(file_obj: Union[io.BytesIO, str], filename: str) -> pd.DataFrame:
    """
    Safely load CSV or Excel file into a pandas DataFrame.
    
    Args:
        file_obj: File buffer or file path string.
        filename: Name of the uploaded file for extension identification.
        
    Returns:
        pd.DataFrame of loaded data.
        
    Raises:
        DataValidationError: If file is empty, corrupted, or unsupported type.
    """
    if not filename:
        raise DataValidationError("No file specified.")

    ext = filename.lower().split(".")[-1]
    if ext not in ["csv", "xlsx", "xls"]:
        raise DataValidationError(
            f"Unsupported file format: '.{ext}'. Please upload a valid CSV (.csv) or Excel (.xlsx, .xls) file."
        )

    try:
        if ext == "csv":
            try:
                df = pd.read_csv(file_obj, encoding="utf-8")
            except pd.errors.EmptyDataError:
                raise DataValidationError("The uploaded file contains no data (empty file).")
            except UnicodeDecodeError:
                if isinstance(file_obj, io.BytesIO):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="latin-1")
        else:
            try:
                df = pd.read_excel(file_obj)
            except pd.errors.EmptyDataError:
                raise DataValidationError("The uploaded file contains no data (empty file).")
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise DataValidationError("The uploaded file contains no data (empty file).")
    except Exception as e:
        raise DataValidationError(f"Could not read the uploaded file: {str(e)}")

    if df is None or df.empty:
        raise DataValidationError("The uploaded file contains no data (empty table).")

    if len(df.columns) < 2:
        raise DataValidationError(
            f"The file has only {len(df.columns)} column. FairLens requires at least features, a target, and a sensitive attribute."
        )

    # Clean whitespace in string columns and column names
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Categorize columns into numeric, categorical, potential targets, and potential sensitive attributes.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    candidate_targets: List[str] = []
    candidate_sensitive: List[str] = []

    sensitive_keywords = [
        "gender", "sex", "age", "race", "ethnicity", "religion",
        "region", "marital", "nationality", "disability", "caste", "language"
    ]
    target_keywords = [
        "target", "approved", "approval", "label", "outcome", "class", "churn",
        "default", "status", "decision", "risk", "hired", "admitted", "readmitted", "y"
    ]
    id_keywords = ["id", "index", "ssn", "uuid", "account_no", "identifier", "applicant_id"]

    priority_targets: List[str] = []
    other_targets: List[str] = []
    priority_sensitive: List[str] = []
    other_sensitive: List[str] = []

    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        col_lower = col.lower()
        is_id = any(ikw in col_lower for ikw in id_keywords)

        # Target candidates: typically binary or 2-3 distinct values
        if not is_id and 2 <= n_unique <= 5:
            if any(tkw in col_lower for tkw in target_keywords):
                priority_targets.append(col)
            else:
                other_targets.append(col)

        # Sensitive candidates: name matches known protected attributes or has 2-12 categories
        if not is_id:
            is_sensitive_kw = any(kw in col_lower for kw in sensitive_keywords)
            if is_sensitive_kw:
                priority_sensitive.append(col)
            elif 2 <= n_unique <= 12 and (df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col])):
                other_sensitive.append(col)

        # General type breakdown
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    candidate_targets = priority_targets + other_targets
    candidate_sensitive = priority_sensitive + other_sensitive

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "candidate_targets": candidate_targets,
        "candidate_sensitive": candidate_sensitive,
        "all_cols": list(df.columns),
        "total_rows": len(df),
        "total_cols": len(df.columns),
    }


def validate_setup_configuration(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    sensitive_cols: List[str],
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that user configuration is mathematically valid for model training and fairness auditing.
    
    Returns:
        (is_valid, error_message_or_none)
    """
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."

    if not sensitive_cols:
        return False, "At least one sensitive demographic attribute must be selected."

    for scol in sensitive_cols:
        if scol not in df.columns:
            return False, f"Sensitive column '{scol}' not found in dataset."
        if scol == target_col:
            return False, f"Sensitive attribute '{scol}' cannot be identical to the target column '{target_col}'."

    # Validate target column classes
    target_series = df[target_col].dropna()
    unique_targets = target_series.unique()
    if len(unique_targets) < 2:
        return False, f"Target column '{target_col}' contains only a single class: {list(unique_targets)}. Binary classification requires at least two distinct classes."
    
    if len(unique_targets) > 2:
        return False, f"FairLens currently focuses on binary classification. Target '{target_col}' has {len(unique_targets)} classes: {list(unique_targets)[:5]}."

    # Validate positive label exists
    str_uniques = [str(x) for x in unique_targets]
    if str(positive_label) not in str_uniques:
        return False, f"Specified positive label '{positive_label}' is not among unique values in '{target_col}': {str_uniques}."

    # Check that remaining feature set is not empty
    exclude_set = set(exclude_cols or [])
    exclude_set.add(target_col)
    remaining_features = [col for col in df.columns if col not in exclude_set]
    if not remaining_features:
        return False, "All columns have been excluded or assigned as target. At least one feature is required to train a model."

    return True, None
