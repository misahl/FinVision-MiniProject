"""Unit tests for FinVision OCR & Field Extraction Pipeline."""

from decimal import Decimal
from pathlib import Path
import pytest

from src.ocr import (
    parse_decimal_amount,
    standardize_date,
    detect_currency,
    cluster_words_into_lines,
    extract_invoice_fields,
    process_invoice_document,
    evaluate_extraction_accuracy,
)


def test_parse_decimal_amount():
    """Verify numeric currency parsing for US and EU comma/dot conventions."""
    assert parse_decimal_amount("1,250.50") == Decimal("1250.50")
    assert parse_decimal_amount("1.250,50") == Decimal("1250.50")
    assert parse_decimal_amount("232,95") == Decimal("232.95")
    assert parse_decimal_amount("232.95") == Decimal("232.95")
    assert parse_decimal_amount("$ 46.55") == Decimal("46.55")
    assert parse_decimal_amount("€ 1.000,00") == Decimal("1000.00")
    assert parse_decimal_amount("invalid") is None
    assert parse_decimal_amount("") is None


def test_standardize_date():
    """Verify date parsing into standard YYYY-MM-DD."""
    assert standardize_date("02/23/2021") == "2021-02-23"
    assert standardize_date("2021-02-23") == "2021-02-23"
    assert standardize_date("23-02-2021") == "2021-02-23"
    assert standardize_date("March 15, 2024") == "2024-03-15"
    assert standardize_date("not-a-date") is None


def test_detect_currency():
    """Verify currency detection from symbols and codes."""
    assert detect_currency("Total: $100.00") == "USD"
    assert detect_currency("Total: €100.00") == "EUR"
    assert detect_currency("Total: £100.00") == "GBP"
    assert detect_currency("Amount: 100.00 CAD") == "CAD"


def test_cluster_words_into_lines():
    """Verify 2D vertical proximity line clustering."""
    words = [
        {"text": "Invoice", "top": 100, "left": 50, "height": 20},
        {"text": "no:", "top": 102, "left": 150, "height": 18},
        {"text": "12345", "top": 101, "left": 200, "height": 21},
        {"text": "Date:", "top": 140, "left": 50, "height": 20},
        {"text": "2024-01-01", "top": 141, "left": 150, "height": 19},
    ]
    lines = cluster_words_into_lines(words)
    assert len(lines) == 2
    assert [w["text"] for w in lines[0]] == ["Invoice", "no:", "12345"]
    assert [w["text"] for w in lines[1]] == ["Date:", "2024-01-01"]


def test_extract_invoice_fields_synthetic():
    """Verify field extraction heuristics on synthetic structured OCR word tokens."""
    words = [
        {"text": "Invoice", "top": 50, "left": 50, "height": 20, "conf": 98.0},
        {"text": "no:", "top": 50, "left": 140, "height": 20, "conf": 97.0},
        {"text": "INV-9988", "top": 50, "left": 180, "height": 20, "conf": 99.0},
        {"text": "Date", "top": 80, "left": 50, "height": 20, "conf": 95.0},
        {"text": "of", "top": 80, "left": 100, "height": 20, "conf": 95.0},
        {"text": "issue:", "top": 80, "left": 130, "height": 20, "conf": 95.0},
        {"text": "05/12/2023", "top": 80, "left": 200, "height": 20, "conf": 96.0},
        {"text": "Seller:", "top": 120, "left": 50, "height": 20, "conf": 94.0},
        {"text": "Client:", "top": 120, "left": 300, "height": 20, "conf": 94.0},
        {"text": "Acme", "top": 145, "left": 50, "height": 20, "conf": 92.0},
        {"text": "Supplies", "top": 145, "left": 110, "height": 20, "conf": 92.0},
        {"text": "Global", "top": 145, "left": 300, "height": 20, "conf": 93.0},
        {"text": "Corp", "top": 145, "left": 370, "height": 20, "conf": 93.0},
        {"text": "Total", "top": 200, "left": 50, "height": 20, "conf": 95.0},
        {"text": "$", "top": 200, "left": 110, "height": 20, "conf": 95.0},
        {"text": "100.00", "top": 200, "left": 130, "height": 20, "conf": 96.0},
        {"text": "$", "top": 200, "left": 210, "height": 20, "conf": 95.0},
        {"text": "10.00", "top": 200, "left": 230, "height": 20, "conf": 96.0},
        {"text": "$", "top": 200, "left": 310, "height": 20, "conf": 95.0},
        {"text": "110.00", "top": 200, "left": 330, "height": 20, "conf": 97.0},
    ]
    raw_text = "Invoice no: INV-9988 Date of issue: 05/12/2023 Seller: Client: Acme Supplies Global Corp Total $ 100.00 $ 10.00 $ 110.00"
    
    inv = extract_invoice_fields(words, raw_text, overall_conf=95.0)

    assert inv.invoice_number == "INV-9988"
    assert inv.invoice_date == "2023-05-12"
    assert inv.vendor == "Acme Supplies"
    assert inv.customer == "Global Corp"
    assert inv.subtotal == Decimal("100.00")
    assert inv.tax == Decimal("10.00")
    assert inv.total == Decimal("110.00")
    assert inv.currency == "USD"
    assert inv.field_confidence["invoiceNumber"] >= 95.0


def test_real_dataset_sample_extraction():
    """Verify real extraction on sample invoice from the local dataset."""
    sample_path = Path("Dataset/high-quality-invoice-images/batch_1/batch_1/batch1_1/batch1-0494.jpg")
    if not sample_path.exists():
        pytest.skip("Dataset sample not found")

    extracted = process_invoice_document(sample_path)

    # Values must match ground truth from batch1_1.csv
    assert extracted.invoice_number == "84652373"
    assert extracted.invoice_date == "2021-02-23"
    assert "nguyen" in extracted.vendor.lower()
    assert "clark" in extracted.customer.lower()
    assert extracted.total == Decimal("232.95")
    assert extracted.subtotal == Decimal("211.77")
    assert extracted.tax == Decimal("21.18")
    assert extracted.tax_rate == Decimal("0.10")
    assert len(extracted.line_items) >= 4
    assert extracted.overall_confidence > 85.0


def test_evaluate_extraction_accuracy_sample():
    """Run real evaluation against a small sample of the dataset ground truth."""
    csv_path = Path("Dataset/high-quality-invoice-images/batch_1/batch_1/batch1_1.csv")
    images_dir = Path("Dataset/high-quality-invoice-images/batch_1/batch_1/batch1_1")
    if not csv_path.exists() or not images_dir.exists():
        pytest.skip("Dataset CSV not found")

    res = evaluate_extraction_accuracy(csv_path, images_dir, sample_size=5)

    assert res["samples_evaluated"] == 5
    assert res["overall_accuracy_pct"] >= 75.0
    assert "invoice_number" in res
    assert "total" in res

