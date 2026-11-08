"""Unit tests for FinVision Financial Rule Verification Engine."""

from decimal import Decimal
import pytest

from src.verification import (
    verify_line_items_math,
    verify_subtotal_sum,
    verify_tax_calculation,
    verify_total_sum,
    verify_required_fields,
    verify_date_sanity,
    verify_numeric_outlier,
    run_verification_rules,
)


def test_rule_line_items_math_pass():
    """Verify correct line item arithmetic passes."""
    items = [
        {"quantity": "2.00", "unitPrice": "25.00", "lineTotal": "50.00"},
        {"quantity": "1.00", "unitPrice": "42.50", "lineTotal": "42.50"},
    ]
    res = verify_line_items_math(items)
    assert res["passed"] is True
    assert res["rule_id"] == "RULE_LINE_ITEM_MATH"


def test_rule_line_items_math_fail():
    """Verify corrupted line item arithmetic triggers CRITICAL failure."""
    items = [
        {"quantity": "3.00", "unitPrice": "20.00", "lineTotal": "80.00"},  # 3 * 20 != 80
    ]
    res = verify_line_items_math(items)
    assert res["passed"] is False
    assert res["severity"] == "CRITICAL"
    assert "Arithmetic mismatch" in res["message"]


def test_rule_subtotal_sum():
    """Verify sum of line item totals equals subtotal."""
    items = [
        {"lineTotal": "50.00"},
        {"lineTotal": "42.50"},
    ]
    # Pass case
    res_pass = verify_subtotal_sum("92.50", items)
    assert res_pass["passed"] is True

    # Fail case
    res_fail = verify_subtotal_sum("100.00", items)
    assert res_fail["passed"] is False
    assert res_fail["severity"] == "CRITICAL"


def test_rule_tax_calculation():
    """Verify tax = subtotal * taxRate."""
    # Pass: 200.00 * 0.10 = 20.00
    res_pass = verify_tax_calculation("200.00", "20.00", "0.10")
    assert res_pass["passed"] is True

    # Fail: 200.00 * 0.10 != 35.00
    res_fail = verify_tax_calculation("200.00", "35.00", "0.10")
    assert res_fail["passed"] is False
    assert res_fail["severity"] == "WARNING"


def test_rule_total_sum():
    """Verify total = subtotal + tax."""
    # Pass: 100.00 + 10.00 = 110.00
    res_pass = verify_total_sum("100.00", "10.00", "110.00")
    assert res_pass["passed"] is True

    # Fail: 100.00 + 10.00 != 125.00
    res_fail = verify_total_sum("100.00", "10.00", "125.00")
    assert res_fail["passed"] is False
    assert res_fail["severity"] == "CRITICAL"


def test_rule_required_fields():
    """Verify required field checks."""
    valid_fields = {
        "invoiceNumber": "INV-100",
        "date": "2024-01-01",
        "vendor": "Acme Corp",
        "total": 150.0,
    }
    res_pass, missing = verify_required_fields(valid_fields)
    assert res_pass["passed"] is True
    assert len(missing) == 0

    # Missing total and vendor
    invalid_fields = {"invoiceNumber": "INV-100", "date": "2024-01-01"}
    res_fail, missing = verify_required_fields(invalid_fields)
    assert res_fail["passed"] is False
    assert "vendor" in missing
    assert "total" in missing


def test_rule_date_sanity():
    """Verify date bounds checking."""
    # Valid past date
    res_valid = verify_date_sanity("2024-01-15")
    assert res_valid["passed"] is True

    # Future date (year 2099)
    res_future = verify_date_sanity("2099-01-01")
    assert res_future["passed"] is False
    assert "future" in res_future["message"]

    # Implausibly old date (> 10 years ago)
    res_old = verify_date_sanity("2000-01-01")
    assert res_old["passed"] is False
    assert "old" in res_old["message"]


def test_rule_numeric_outlier():
    """Verify historical outlier detection and insufficient data handling."""
    # Case 1: Insufficient data (< 5 docs)
    res_few = verify_numeric_outlier("5000.00", [100.0, 150.0, 120.0])
    assert res_few["passed"] is True
    assert "Not enough data" in res_few["message"]

    # Case 2: Normal value consistent with history
    history = [100.0, 110.0, 95.0, 105.0, 102.0, 98.0, 108.0]
    res_norm = verify_numeric_outlier("104.00", history)
    assert res_norm["passed"] is True

    # Case 3: Extreme statistical outlier ($25,000 vs median ~$102)
    res_outlier = verify_numeric_outlier("25000.00", history)
    assert res_outlier["passed"] is False
    assert res_outlier["severity"] == "WARNING"


def test_run_verification_rules_composite():
    """Verify full verification runner aggregates rule results and flags."""
    fields = {
        "invoiceNumber": "INV-001",
        "date": "2024-02-01",
        "vendor": "Acme",
        "subtotal": "100.00",
        "tax": "10.00",
        "taxRate": "0.10",
        "total": "110.00",
    }
    items = [
        {"quantity": "2.00", "unitPrice": "50.00", "lineTotal": "100.00"}
    ]

    summary = run_verification_rules(fields, items)
    assert summary["allPassed"] is True
    assert summary["criticalFailuresCount"] == 0
    assert len(summary["ruleResults"]) == 7

