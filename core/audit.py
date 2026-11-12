"""Data Audit engine for FairLens: representation checks, missingness, and pre-training raw disparities."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.utils import apply_theme_to_figure, COLOR_PRIMARY, COLOR_CYAN, COLOR_BIASED, COLOR_FAIR, COLOR_CAUTION


def compute_data_audit(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    sensitive_cols: List[str],
    small_group_threshold: int = 30,
) -> Dict[str, Any]:
    """
    Perform comprehensive pre-modeling audit on dataset.
    
    Computes:
    - Overall row count, column count, total missing values.
    - Class balance in the target variable.
    - Missing value distribution per column.
    - Per-sensitive attribute group representation (counts, percentages, small group flags).
    - Raw historical favorable outcome rate per demographic group BEFORE any model is trained.
    """
    # 1. Target column distribution
    target_series = df[target_col].astype(str)
    pos_str = str(positive_label)
    
    total_rows = len(df)
    class_counts = target_series.value_counts().to_dict()
    positive_count = int(class_counts.get(pos_str, 0))
    negative_count = total_rows - positive_count
    overall_positive_rate = positive_count / total_rows if total_rows > 0 else 0.0

    # 2. Missing value statistics
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_summary = [
        {"column": col, "missing_count": int(missing_counts[col]), "missing_pct": float(missing_pct[col])}
        for col in df.columns if missing_counts[col] > 0
    ]
    missing_summary.sort(key=lambda x: x["missing_count"], reverse=True)

    # 3. Demographic group breakdown and raw favorable outcome rate
    group_audits: Dict[str, Dict[str, Any]] = {}
    small_group_warnings: List[Dict[str, Any]] = []

    for scol in sensitive_cols:
        col_series = df[scol].fillna("Missing/Unknown").astype(str)
        group_counts = col_series.value_counts()
        
        # Binary favorable outcome: 1 if positive_label, else 0
        is_pos = (target_series == pos_str).astype(int)
        
        breakdown = []
        for g_name, count in group_counts.items():
            mask = (col_series == g_name)
            pos_in_group = int(is_pos[mask].sum())
            raw_rate = pos_in_group / count if count > 0 else 0.0
            is_small = count < small_group_threshold

            if is_small:
                small_group_warnings.append({
                    "attribute": scol,
                    "group": g_name,
                    "count": int(count),
                    "threshold": small_group_threshold,
                })

            breakdown.append({
                "group": g_name,
                "count": int(count),
                "percentage": float((count / total_rows) * 100),
                "positive_count": pos_in_group,
                "raw_outcome_rate": float(raw_rate),
                "is_small_group": bool(is_small),
            })

        # Calculate max baseline disparity
        rates = [item["raw_outcome_rate"] for item in breakdown if item["count"] > 0]
        max_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        raw_gap = max_rate - min_rate
        raw_disparate_impact = (min_rate / max_rate) if max_rate > 0 else 1.0

        group_audits[scol] = {
            "breakdown": breakdown,
            "max_rate": max_rate,
            "min_rate": min_rate,
            "raw_gap": raw_gap,
            "raw_disparate_impact": raw_disparate_impact,
            "unique_groups": len(breakdown),
        }

    return {
        "total_rows": total_rows,
        "total_cols": len(df.columns),
        "target_col": target_col,
        "positive_label": pos_str,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "overall_positive_rate": overall_positive_rate,
        "class_counts": class_counts,
        "missing_summary": missing_summary,
        "group_audits": group_audits,
        "small_group_warnings": small_group_warnings,
    }


def create_group_size_chart(group_breakdown: List[Dict[str, Any]], attribute_name: str) -> go.Figure:
    """Create interactive bar chart of demographic group representation."""
    groups = [x["group"] for x in group_breakdown]
    counts = [x["count"] for x in group_breakdown]
    pcts = [x["percentage"] for x in group_breakdown]
    colors = [COLOR_BIASED if x["is_small_group"] else COLOR_PRIMARY for x in group_breakdown]

    hover_texts = [
        f"Group: <b>{g}</b><br>Count: {c:,}<br>Share: {p:.1f}%" + ("<br>⚠️ Small Group (<30)" if s else "")
        for g, c, p, s in zip(groups, counts, pcts, [x["is_small_group"] for x in group_breakdown])
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=groups,
                y=counts,
                text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts, pcts)],
                textposition="auto",
                marker_color=colors,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=f"Sample Representation by <b>{attribute_name}</b>",
        xaxis_title=attribute_name,
        yaxis_title="Record Count",
        height=320,
    )
    return apply_theme_to_figure(fig)


def create_raw_outcome_chart(group_breakdown: List[Dict[str, Any]], attribute_name: str, positive_label: str) -> go.Figure:
    """Create interactive bar chart of raw historical outcome rates before modeling."""
    groups = [x["group"] for x in group_breakdown]
    rates = [x["raw_outcome_rate"] * 100 for x in group_breakdown]
    colors = [COLOR_CYAN for _ in groups]

    fig = go.Figure(
        data=[
            go.Bar(
                x=groups,
                y=rates,
                text=[f"{r:.1f}%" for r in rates],
                textposition="auto",
                marker_color=colors,
                hovertext=[
                    f"Group: <b>{g}</b><br>Favorable Rate: {r:.1f}%<br>Approved: {x['positive_count']:,} of {x['count']:,}"
                    for g, r, x in zip(groups, rates, group_breakdown)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=f"Raw Favorable Outcome Rate (Target = '{positive_label}')",
        xaxis_title=attribute_name,
        yaxis_title="Favorable Outcome Rate (%)",
        yaxis_range=[0, 105],
        height=320,
    )
    return apply_theme_to_figure(fig)


def create_missing_value_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create visualization of missing values across columns."""
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].sort_values(ascending=False)

    if cols_with_missing.empty:
        # Return a clean empty-indicator figure
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values detected across all columns in this dataset",
            showarrow=False,
            font=dict(size=14, color=COLOR_FAIR),
        )
        fig.update_layout(height=180)
        return apply_theme_to_figure(fig)

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(cols_with_missing.index),
                y=[int(v) for v in cols_with_missing.values],
                text=[f"{int(v)} ({(v / len(df))*100:.1f}%)" for v in cols_with_missing.values],
                textposition="auto",
                marker_color=COLOR_CAUTION,
            )
        ]
    )
    fig.update_layout(
        title="Missing Values by Feature",
        xaxis_title="Column",
        yaxis_title="Missing Row Count",
        height=300,
    )
    return apply_theme_to_figure(fig)

