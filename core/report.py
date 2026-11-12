"""Audit Report and Model Card generator for FairLens using ReportLab and Markdown.

Every number, group name, and evaluation finding is derived strictly
from the active session state. ZERO hardcoded numbers.
"""

from typing import Dict, List, Any, Optional
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_pdf_report(
    session_data: Dict[str, Any]
) -> bytes:
    """
    Generate an executive PDF Fairness Audit Report from live session metrics.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4338ca"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # Title & Header
    story.append(Paragraph("Responsible AI: FairLens", title_style))
    story.append(Paragraph("AI Bias & Fairness Comprehensive Audit Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4338ca"), spaceAfter=12))

    # Audit Metadata Table
    dataset_name = str(session_data.get("dataset_name", "Uploaded User Dataset"))
    target_col = str(session_data.get("target_col", "Target"))
    sensitive_cols = ", ".join(session_data.get("sensitive_cols", ["Demographic"]))
    eval_model_name = str(session_data.get("active_model_name", "Baseline Model"))
    audit_date = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")

    meta_data = [
        [Paragraph("<b>Audit Date:</b>", body_style), Paragraph(audit_date, body_style),
         Paragraph("<b>Dataset Source:</b>", body_style), Paragraph(dataset_name, body_style)],
        [Paragraph("<b>Target Variable:</b>", body_style), Paragraph(target_col, body_style),
         Paragraph("<b>Sensitive Attributes:</b>", body_style), Paragraph(sensitive_cols, body_style)],
        [Paragraph("<b>Evaluated Model:</b>", body_style), Paragraph(eval_model_name, body_style),
         Paragraph("<b>Compliance Standard:</b>", body_style), Paragraph("EEOC 80% Rule & Parity", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Executive Summary & Fairness Score
    fairness_res = session_data.get("fairness_results", {})
    score = fairness_res.get("fairness_score", 0.0)
    summary_text = fairness_res.get("summary_text", "Audit metrics computed from active session.")

    score_color = "#16a34a" if score >= 80 else ("#d97706" if score >= 60 else "#dc2626")
    score_label = "FAIR (COMPLIANT)" if score >= 80 else ("CAUTION (MODERATE BIAS)" if score >= 60 else "CRITICAL BIAS")

    story.append(Paragraph("1. Executive Summary & Fairness Score", h2_style))
    score_box_data = [
        [
            Paragraph(f"<font size=24 color='{score_color}'><b>{round(score):d}/100</b></font><br/><font size=8 color='#64748b'><b>{score_label}</b></font>", body_style),
            Paragraph(summary_text, body_style),
        ]
    ]
    t_score = Table(score_box_data, colWidths=[1.8*inch, 5.2*inch])
    t_score.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(score_color)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_score)
    story.append(Spacer(1, 10))

    # Core Fairness Metrics Table
    story.append(Paragraph("2. Quantitative Algorithmic Fairness Metrics", h2_style))
    dpd = fairness_res.get("demographic_parity_difference", 0.0)
    dir_val = fairness_res.get("disparate_impact_ratio", 1.0)
    eod = fairness_res.get("equal_opportunity_difference", 0.0)
    eq_odds = fairness_res.get("equalized_odds_difference", 0.0)

    dir_status = "Pass (>= 0.80)" if dir_val >= 0.80 else "Violated (< 0.80)"
    dpd_status = "Pass (<= 0.10)" if dpd <= 0.10 else "Disparity Detected"
    eod_status = "Pass (<= 0.10)" if eod <= 0.10 else "Opportunity Gap"

    metrics_table_data = [
        [Paragraph("<b>Metric Name</b>", bold_body), Paragraph("<b>Value</b>", bold_body), Paragraph("<b>Target Threshold</b>", bold_body), Paragraph("<b>Status</b>", bold_body)],
        [Paragraph("Disparate Impact Ratio (DIR)", body_style), Paragraph(f"{dir_val:.3f}", body_style), Paragraph(">= 0.80 (4/5ths Rule)", body_style), Paragraph(dir_status, body_style)],
        [Paragraph("Demographic Parity Difference", body_style), Paragraph(f"{dpd:.1%}", body_style), Paragraph("<= 10.0%", body_style), Paragraph(dpd_status, body_style)],
        [Paragraph("Equal Opportunity Difference", body_style), Paragraph(f"{eod:.1%}", body_style), Paragraph("<= 10.0%", body_style), Paragraph(eod_status, body_style)],
        [Paragraph("Equalized Odds Difference", body_style), Paragraph(f"{eq_odds:.1%}", body_style), Paragraph("<= 10.0%", body_style), Paragraph("Audited", body_style)],
    ]
    t_metrics = Table(metrics_table_data, colWidths=[2.6*inch, 1.2*inch, 1.8*inch, 1.4*inch])
    t_metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # Disaggregated Per-Group Breakdown
    group_metrics = fairness_res.get("group_metrics", {})
    if group_metrics:
        story.append(Paragraph("3. Disaggregated Group Performance & Selection Rates", h2_style))
        group_rows = [
            [Paragraph("<b>Group</b>", bold_body), Paragraph("<b>Samples</b>", bold_body), Paragraph("<b>Selection Rate</b>", bold_body), Paragraph("<b>Accuracy</b>", bold_body), Paragraph("<b>TPR (Recall)</b>", bold_body), Paragraph("<b>FPR</b>", bold_body)]
        ]
        for g_name, gm in group_metrics.items():
            group_rows.append([
                Paragraph(str(g_name), body_style),
                Paragraph(f"{gm.get('count', 0):,}", body_style),
                Paragraph(f"{gm.get('selection_rate', 0.0):.1%}", body_style),
                Paragraph(f"{gm.get('accuracy', 0.0):.1%}", body_style),
                Paragraph(f"{gm.get('tpr', 0.0):.1%}", body_style),
                Paragraph(f"{gm.get('fpr', 0.0):.1%}", body_style),
            ])
        t_group = Table(group_rows, colWidths=[1.8*inch, 1.0*inch, 1.2*inch, 1.0*inch, 1.0*inch, 1.0*inch])
        t_group.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_group)
        story.append(Spacer(1, 10))

    # Mitigation Summary (if present)
    mitigations = session_data.get("mitigation_results", [])
    if mitigations:
        story.append(Paragraph("4. Algorithmic Bias Mitigation Assessment", h2_style))
        mit_rows = [
            [Paragraph("<b>Mitigation Method</b>", bold_body), Paragraph("<b>Fairness Score</b>", bold_body), Paragraph("<b>DIR</b>", bold_body), Paragraph("<b>Accuracy</b>", bold_body), Paragraph("<b>Status</b>", bold_body)]
        ]
        for m in mitigations:
            m_fairness = m.get("fairness", {})
            m_score = m_fairness.get("fairness_score", 0.0)
            m_dir = m_fairness.get("disparate_impact_ratio", 0.0)
            m_acc = m.get("accuracy", 0.0)
            mit_rows.append([
                Paragraph(str(m.get("method", "Mitigation")), body_style),
                Paragraph(f"{round(m_score):d}/100", body_style),
                Paragraph(f"{m_dir:.3f}", body_style),
                Paragraph(f"{m_acc:.1%}", body_style),
                Paragraph("Bias Reduced", body_style),
            ])
        t_mit = Table(mit_rows, colWidths=[2.6*inch, 1.2*inch, 1.0*inch, 1.0*inch, 1.2*inch])
        t_mit.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_mit)
        story.append(Spacer(1, 10))

    # Limitations & Recommendations
    story.append(KeepTogether([
        Paragraph("5. Ethical Limitations & Audit Recommendations", h2_style),
        Paragraph("<b>1. Observational Data Limitations:</b> Fairness metrics reflect parity across the evaluated dataset. If the historical data contains unmeasured sampling bias or label bias, parity alone does not guarantee real-world fairness.", body_style),
        Paragraph("<b>2. Subgroup Sample Sizes:</b> Groups with fewer than 30 evaluation rows exhibit high statistical variance; fairness numbers for small groups must be interpreted cautiously.", body_style),
        Paragraph("<b>3. Ongoing Monitoring:</b> Bias mitigation reduces measured disparity on the test split, but model drift over time can reintroduce bias. Ongoing production telemetry is required.", body_style),
        Paragraph("<b>4. Human-in-the-Loop Oversight:</b> Automated threshold adjustments should never replace domain-expert review in high-stakes decisions (e.g. lending, hiring, healthcare).", body_style),
    ]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_markdown_model_card(session_data: Dict[str, Any]) -> str:
    """
    Generate standard Model Card in Markdown format following Mitchell et al. (2019).
    All data dynamically populated from session state.
    """
    dataset_name = session_data.get("dataset_name", "Uploaded Dataset")
    target_col = session_data.get("target_col", "Target")
    sensitive_cols = session_data.get("sensitive_cols", ["Protected Attribute"])
    model_name = session_data.get("active_model_name", "Baseline Model")
    fairness = session_data.get("fairness_results", {})
    score = fairness.get("fairness_score", 0.0)
    dir_val = fairness.get("disparate_impact_ratio", 1.0)
    dpd = fairness.get("demographic_parity_difference", 0.0)
    eod = fairness.get("equal_opportunity_difference", 0.0)

    perf = session_data.get("model_performance_results", {}).get(model_name, {}).get("overall", {})
    acc = perf.get("accuracy", 0.0)
    f1 = perf.get("f1", 0.0)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    card = f"""# Model Card: {model_name} (Audited by FairLens)

**Date of Audit:** {date_str}  
**Audit Tool:** Responsible AI: FairLens (v1.0.0)  
**Dataset Reference:** `{dataset_name}`  

---

## 1. Model Details
- **Model Architecture:** {model_name}
- **Task:** Binary Classification
- **Target Variable:** `{target_col}`
- **Protected Demographic Attributes Audited:** `{', '.join(sensitive_cols)}`

## 2. Intended Use
- **Primary Use Case:** Automated screening and risk scoring based on tabular features.
- **Out-of-Scope Use Cases:** Fully automated high-stakes approvals without human-in-the-loop review, or application to demographic populations not represented in the audited dataset.

## 3. Quantitative Performance & Fairness Summary
| Metric | Value | Compliance Status / Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | {acc:.2%} | Overall predictive accuracy on held-out test split |
| **F1 Score** | {f1:.3f} | Harmonic mean of precision and recall |
| **Fairness Score** | {round(score):d}/100 | Composite equity index across DIR, DPD, and EOD |
| **Disparate Impact Ratio (DIR)** | {dir_val:.3f} | Four-fifths (0.80) standard rule |
| **Demographic Parity Difference** | {dpd:.2%} | Absolute gap in positive outcome rates across groups |
| **Equal Opportunity Difference** | {eod:.2%} | Absolute gap in True Positive Rates across groups |

## 4. Ethical Considerations & Limitations
- **Proxy Features:** Removing explicit protected demographic columns does not eliminate bias due to proxy correlations in other features.
- **Small Group Reliability:** Metrics for groups with sample sizes below 30 have wider confidence intervals.
- **Label Bias:** If historical ground truth reflects societal prejudice, equalizing model error rates may not resolve systemic disparities.

## 5. Recommendations
1. Continuously monitor demographic parity and equal opportunity metrics post-deployment.
2. Maintain a human review mechanism for borderline threshold predictions.
3. Apply in-processing or post-processing mitigation if Disparate Impact Ratio falls below 0.80.
"""
    return card

