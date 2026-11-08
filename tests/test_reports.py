"""Unit tests for FinVision PDF Report Generation."""

from decimal import Decimal
import pymupdf
import pytest

from frontend.reports import generate_pdf_report


def test_generate_pdf_report_validity():
    """Verify generated PDF byte payload is valid, contains fields and mandatory disclaimer."""
    doc = {
        "docId": "doc_test_report_12345",
        "createdAt": "2024-03-15T10:00:00Z",
        "overallConfidence": 94.5,
        "fields": {
            "invoiceNumber": "INV-778899",
            "date": "2024-03-15",
            "vendor": "Acme Industrial Supplies",
            "customer": "Global Corp",
            "subtotal": 1000.0,
            "tax": 100.0,
            "total": 1100.0,
        },
        "fieldConfidence": {
            "invoiceNumber": 98.0,
            "date": 96.0,
            "vendor": 94.0,
            "customer": 95.0,
            "subtotal": 92.0,
            "tax": 90.0,
            "total": 97.0,
        },
        "verification": {
            "ruleResults": [
                {"rule_id": "RULE_TOTAL_SUM", "passed": True, "message": "Total arithmetic verified"},
                {"rule_id": "RULE_DATE_SANITY", "passed": True, "message": "Date within normal range"},
            ],
            "cvScore": 8.5,
            "cvRegions": [],
        },
        "ml": {
            "modelVersion": "v1.0.0",
            "probability": 0.08,
            "verdict": "normal",
        },
        "risk": {
            "level": "LOW",
            "score": 8.5,
            "recommendedAction": "Eligible for straight-through approval.",
        },
        "review": {
            "decision": "approved",
            "reviewerUid": "analyst_42",
        },
    }

    pdf_bytes = generate_pdf_report(doc)

    # 1. Byte header check
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file does not have valid PDF magic bytes!"
    assert len(pdf_bytes) > 1000

    # 2. Text inspection via PyMuPDF
    pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(pdf_doc) >= 1
    text_content = ""
    for page in pdf_doc:
        text_content += page.get_text()

    assert "FinVision Audit & Verification Report" in text_content
    assert "INV-778899" in text_content
    assert "Acme Industrial Supplies" in text_content
    assert "$1,100.00" in text_content
    assert "LEGAL & COMPLIANCE DISCLAIMER" in text_content
    assert "automated" in text_content
    assert "fraudulent" not in text_content.lower() or "not declare or prove fraudulent" in text_content.lower()

