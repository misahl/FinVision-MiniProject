"""Audit Report Generation Module (PDF via ReportLab) for FinVision."""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import firebase_service


def generate_pdf_report(doc: Dict[str, Any]) -> bytes:
    """Generate professional PDF audit report for an invoice document."""
    buffer = io.BytesIO()
    doc_template = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        fontName="Helvetica-Oblique",
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("FinVision Audit & Verification Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
            f"Document ID: {doc.get('docId', 'Unknown')}",
            body_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=14))

    # 2. Executive Risk Assessment Summary
    risk = doc.get("risk", {})
    risk_level = risk.get("level", "LOW")
    risk_score = risk.get("score", 0.0)

    level_color = colors.HexColor("#10B981") if risk_level == "LOW" else (
        colors.HexColor("#F59E0B") if risk_level == "MEDIUM" else colors.HexColor("#EF4444")
    )

    summary_data = [
        [
            Paragraph("<b>Overall Risk Level</b>", body_style),
            Paragraph(f"<font color='{level_color.hexval()}'><b>{risk_level} ({risk_score:.0f}/100)</b></font>", body_style),
            Paragraph("<b>Review Status</b>", body_style),
            Paragraph(f"<b>{(doc.get('review') or {}).get('decision', 'Unreviewed').upper()}</b>", body_style),
        ],
        [
            Paragraph("<b>Recommended Action</b>", body_style),
            Paragraph(risk.get("recommendedAction", "Manual verification suggested."), body_style),
            Paragraph("<b>OCR Word Conf</b>", body_style),
            Paragraph(f"{doc.get('overallConfidence', 0.0):.1f}%", body_style),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[120, 160, 100, 150])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 3. Document Fields
    story.append(Paragraph("1. Extracted Document Fields", h2_style))
    fields = doc.get("fields", {})
    fields_data = [
        [Paragraph("<b>Field Name</b>", body_style), Paragraph("<b>Stated / Extracted Value</b>", body_style), Paragraph("<b>Confidence</b>", body_style)],
        [Paragraph("Invoice Number", body_style), Paragraph(str(fields.get("invoiceNumber") or "N/A"), body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('invoiceNumber', 0):.0f}%", body_style)],
        [Paragraph("Date of Issue", body_style), Paragraph(str(fields.get("date") or "N/A"), body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('date', 0):.0f}%", body_style)],
        [Paragraph("Vendor Name", body_style), Paragraph(str(fields.get("vendor") or "N/A"), body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('vendor', 0):.0f}%", body_style)],
        [Paragraph("Customer Name", body_style), Paragraph(str(fields.get("customer") or "N/A"), body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('customer', 0):.0f}%", body_style)],
        [Paragraph("Subtotal Amount", body_style), Paragraph(f"${float(fields.get('subtotal', 0.0)):,.2f}" if fields.get("subtotal") else "N/A", body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('subtotal', 0):.0f}%", body_style)],
        [Paragraph("Tax Amount", body_style), Paragraph(f"${float(fields.get('tax', 0.0)):,.2f}" if fields.get("tax") else "N/A", body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('tax', 0):.0f}%", body_style)],
        [Paragraph("Total Stated Amount", body_style), Paragraph(f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A", body_style), Paragraph(f"{doc.get('fieldConfidence', {}).get('total', 0):.0f}%", body_style)],
    ]
    fields_table = Table(fields_data, colWidths=[150, 260, 120])
    fields_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(fields_table)
    story.append(Spacer(1, 12))

    # 4. Deterministic Verification Rules
    story.append(Paragraph("2. Deterministic Verification Rules", h2_style))
    verif = doc.get("verification", {})
    rules_data = [
        [Paragraph("<b>Rule Identifier</b>", body_style), Paragraph("<b>Outcome</b>", body_style), Paragraph("<b>Audit Message</b>", body_style)],
    ]
    for r in verif.get("ruleResults", []):
        p_status = "<font color='green'><b>PASS</b></font>" if r.get("passed", True) else "<font color='red'><b>FAIL</b></font>"
        rules_data.append([
            Paragraph(r.get("rule_id", "RULE"), body_style),
            Paragraph(p_status, body_style),
            Paragraph(r.get("message", ""), body_style),
        ])
    rules_table = Table(rules_data, colWidths=[140, 70, 320])
    rules_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(rules_table)
    story.append(Spacer(1, 12))

    # 5. Machine Learning & Computer Vision
    story.append(Paragraph("3. Computer Vision & Statistical ML Diagnostics", h2_style))
    ml = doc.get("ml", {})
    prob_pct = f"{ml.get('probability', 0.0) * 100:.1f}%"
    diag_text = (
        f"<b>Statistical Classifier (Random Forest v1.0.0):</b> Prediction: {ml.get('verdict', 'Normal').upper()} "
        f"(Suspiciousness Probability: {prob_pct}, Experimental: True).<br/>"
        f"<b>Computer Vision Anomaly Score:</b> {verif.get('cvScore', 0.0):.1f} / 100.<br/>"
        f"<b>Visual Outlier Regions:</b> {len(verif.get('cvRegions', []))} localized bounding cluster(s)."
    )
    story.append(Paragraph(diag_text, body_style))
    story.append(Spacer(1, 16))

    # 6. Mandatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#94A3B8"), spaceAfter=10))
    disclaimer_text = (
        "<b>LEGAL & COMPLIANCE DISCLAIMER:</b> This verification report is generated by automated computer-vision heuristics, "
        "deterministic business rules, and statistical classification algorithms. It provides an advisory risk assessment "
        "and does NOT declare or prove fraudulent activity. All decisions must be validated by qualified human accounting personnel."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc_template.build(story)
    return buffer.getvalue()


def render_reports(user: dict):
    """Render PDF report generation and download interface."""
    st.markdown("## Official Verification Audit Reports (PDF)")
    st.markdown("Generate and download compliance-ready PDF verification reports with full rule traces and legal disclaimers.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=50)

    if not docs:
        st.info("No documents in repository to generate reports for.")
        return

    doc_options = {
        d["docId"]: f"{d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} ({d['docId'][:10]}) - {d.get('fields', {}).get('vendor') or 'Unknown'}"
        for d in docs
    }

    selected_doc_id = st.selectbox(
        "Select Document to Export",
        options=list(doc_options.keys()),
        format_func=lambda did: doc_options[did],
    )

    doc = firebase_service.get_document(selected_doc_id)
    if not doc:
        st.error("Document not found.")
        return

    st.markdown("### Document Summary")
    st.write(f"**Invoice #:** {doc.get('fields', {}).get('invoiceNumber') or 'N/A'}")
    st.write(f"**Vendor:** {doc.get('fields', {}).get('vendor') or 'N/A'}")
    st.write(f"**Stated Total:** ${float(doc.get('fields', {}).get('total', 0.0)):,.2f}" if doc.get('fields', {}).get('total') else "**Stated Total:** N/A")
    st.write(f"**Risk Level:** {doc.get('risk', {}).get('level', 'LOW')}")

    if st.button("📄 Generate Compliance PDF Report", type="primary"):
        with st.spinner("Compiling ReportLab flowables, styling, and audit tables..."):
            pdf_bytes = generate_pdf_report(doc)

            # Save report to Firebase Storage
            report_filename = f"report_{doc['docId']}.pdf"
            firebase_service.upload_invoice_bytes(
                user.get("uid", "anonymous"), doc["docId"], report_filename, pdf_bytes, "application/pdf"
            )

            st.success("PDF Report generated successfully!")
            st.download_button(
                label="⬇️ Download PDF Audit Report",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True,
            )

