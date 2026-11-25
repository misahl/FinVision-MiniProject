"""Test multi-engine document extraction on all generated sample invoices."""

from pathlib import Path
import pytest
from src import ocr, verification, duplicate_detection


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_invoices"


def test_sample_01_valid_invoice_extraction():
    """Verify clean valid sample invoice extracts all fields and passes financial arithmetic."""
    pdf_path = SAMPLE_DIR / "sample_01_valid_invoice.pdf"
    assert pdf_path.exists()

    words, raw, conf = ocr.extract_document_words(pdf_path)
    assert len(words) > 20
    assert conf >= 90.0

    inv = ocr.extract_invoice_fields(words, raw, conf)
    f = inv.to_dict()["fields"]

    assert f["invoiceNumber"] == "INV-2026-001"
    assert "ACME GLOBAL" in (f["vendor"] or "")
    assert f["date"] == "2026-10-15"
    assert f["dueDate"] == "2026-11-15"
    assert float(f["subtotal"]) == 1000.00
    assert float(f["tax"]) == 100.00
    assert float(f["total"]) == 1100.00

    # Verification rules pass
    rule_res = verification.run_verification_rules(f, [])
    assert rule_res["allPassed"] is True


def test_sample_02_math_mismatch_extraction():
    """Verify sample invoice with arithmetic discrepancy is accurately extracted and flagged."""
    pdf_path = SAMPLE_DIR / "sample_02_math_mismatch.pdf"
    assert pdf_path.exists()

    words, raw, conf = ocr.extract_document_words(pdf_path)
    inv = ocr.extract_invoice_fields(words, raw, conf)
    f = inv.to_dict()["fields"]

    assert f["invoiceNumber"] == "INV-2026-002"
    assert "APEX LOGISTICS" in (f["vendor"] or "")
    assert float(f["subtotal"]) == 500.00
    assert float(f["tax"]) == 50.00
    assert float(f["total"]) == 850.00

    # Verification rules must fail total sum rule (500 + 50 = 550 != 850)
    rule_res = verification.run_verification_rules(f, [])
    assert rule_res["allPassed"] is False
    total_rule = next(r for r in rule_res["ruleResults"] if r["rule_id"] == "RULE_TOTAL_SUM")
    assert total_rule["passed"] is False


def test_sample_03_duplicate_collision():
    """Verify duplicate invoice generates semantic collision with sample 01 (identical number, vendor, total)."""
    pdf_01 = SAMPLE_DIR / "sample_01_valid_invoice.pdf"
    pdf_03 = SAMPLE_DIR / "sample_03_duplicate_invoice.pdf"

    inv_01 = ocr.extract_invoice_fields(*ocr.extract_document_words(pdf_01))
    inv_03 = ocr.extract_invoice_fields(*ocr.extract_document_words(pdf_03))

    f1 = inv_01.to_dict()["fields"]
    f3 = inv_03.to_dict()["fields"]

    assert f1["invoiceNumber"] == f3["invoiceNumber"]
    assert f1["vendor"] == f3["vendor"]
    assert f1["total"] == f3["total"]

    existing = [{
        "docId": "doc_sample_01",
        "fields": f1,
    }]

    target = {
        "docId": "doc_sample_03",
        "fields": f3,
    }

    dup_res = duplicate_detection.find_duplicates(target, existing)
    assert dup_res["hasDuplicate"] is True
    assert dup_res["duplicateMatches"][0]["matchType"] in ["SEMANTIC_INVOICE_VENDOR", "SEMANTIC_VENDOR_DATE_TOTAL"]


def test_sample_05_with_discount_tax_extraction():
    """Verify invoice with discount and tax extracts and validates: subtotal - discount + tax == total."""
    pdf_path = SAMPLE_DIR / "sample_05_with_discount_tax.pdf"
    assert pdf_path.exists()

    words, raw, conf = ocr.extract_document_words(pdf_path)
    inv = ocr.extract_invoice_fields(words, raw, conf)
    f = inv.to_dict()["fields"]

    assert f["invoiceNumber"] == "INV-2026-005"
    assert float(f["subtotal"]) == 1000.00
    assert float(f["discount"]) == 50.00
    assert float(f["tax"]) == 76.00
    assert float(f["total"]) == 1026.00

    # 1000 - 50 + 76 = 1026
    rule_res = verification.run_verification_rules(f, [])
    assert rule_res["allPassed"] is True


def test_sample_06_european_format_extraction():
    """Verify European decimal and currency invoice is parsed accurately."""
    pdf_path = SAMPLE_DIR / "sample_06_european_format.pdf"
    assert pdf_path.exists()

    words, raw, conf = ocr.extract_document_words(pdf_path)
    inv = ocr.extract_invoice_fields(words, raw, conf)
    f = inv.to_dict()["fields"]

    assert f["invoiceNumber"] == "INV-DE-8841"
    assert f["currency"] == "EUR"
    assert float(f["subtotal"]) == 1000.00
    assert float(f["tax"]) == 190.00
    assert float(f["total"]) == 1190.00
    rule_res = verification.run_verification_rules(f, [])
    assert rule_res["allPassed"] is True
