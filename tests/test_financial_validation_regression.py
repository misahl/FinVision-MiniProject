"""Automated Regression Test Suite for Financial Calculation and Rule Invariants.

Guarantees:
1. Correct line-item arithmetic formula: line_total == quantity * unit_price
2. Detection of arithmetic discrepancies with exact difference reporting
3. Graceful handling of missing quantities (NOT CHECKED / WARNING, never false math error)
4. Valid Subtotal + Tax - Discount == Grand Total
5. Grand total discrepancy detection
6. Missing tax rate reporting NOT CHECKED with exact expected message
7. Decimal handling and rounding tolerance within $0.02
8. Complete end-to-end invoice verification of the user's test invoice
"""

from decimal import Decimal
import pytest

from src.verification import (
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_WARNING,
    STATUS_NOT_CHECKED,
    verify_line_items_math,
    verify_subtotal_sum,
    verify_tax_calculation,
    verify_total_sum,
    run_verification_rules,
)


def test_regression_01_matching_line_item_chairs():
    """Test 1: Office chairs: Qty 2, Unit Price 1500, Line Total 3000 -> PASSED."""
    items = [
        {"description": "Office chairs", "quantity": 2, "unitPrice": 1500, "lineTotal": 3000}
    ]
    res = verify_line_items_math(items)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True
    assert res["severity"] == "INFO"
    assert len(res["details"]) == 1
    d = res["details"][0]
    assert d["status"] == STATUS_PASSED
    assert d["quantity"] == 2.0
    assert d["unit_price"] == 1500.0
    assert d["expected_total"] == 3000.0
    assert d["line_total"] == 3000.0
    assert d["difference"] == 0.0


def test_regression_02_matching_line_item_toner():
    """Test 2: Printer toner: Qty 1, Unit Price 2000, Line Total 2000 -> PASSED."""
    items = [
        {"description": "Printer toner", "quantity": 1, "unitPrice": 2000, "lineTotal": 2000}
    ]
    res = verify_line_items_math(items)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True
    assert res["severity"] == "INFO"
    assert len(res["details"]) == 1
    d = res["details"][0]
    assert d["status"] == STATUS_PASSED
    assert d["quantity"] == 1.0
    assert d["unit_price"] == 2000.0
    assert d["expected_total"] == 2000.0
    assert d["line_total"] == 2000.0
    assert d["difference"] == 0.0


def test_regression_03_corrupted_line_total():
    """Test 3: Corrupted line total: Qty 2, Unit Price 1500, Line Total 3500 -> FAILED (diff: 500)."""
    items = [
        {"description": "Office chairs", "quantity": 2, "unitPrice": 1500, "lineTotal": 3500}
    ]
    res = verify_line_items_math(items)
    assert res["status"] == STATUS_FAILED
    assert res["passed"] is False
    assert res["severity"] == "CRITICAL"
    assert len(res["details"]) == 1
    d = res["details"][0]
    assert d["status"] == STATUS_FAILED
    assert d["difference"] == 500.0
    assert d["expected_total"] == 3000.0
    assert d["line_total"] == 3500.0
    assert "difference: 500.00" in res["message"] or "difference: 500" in res["message"]


def test_regression_04_missing_quantity():
    """Test 4: Missing quantity -> NOT CHECKED or WARNING, never false multiplication error."""
    items = [
        {"description": "Consulting retainer", "quantity": None, "unitPrice": 1500, "lineTotal": 1500}
    ]
    res = verify_line_items_math(items)
    assert res["status"] in (STATUS_NOT_CHECKED, STATUS_WARNING)
    # Must not multiply Unit Price by Line Total
    assert "2250000" not in res["message"]
    assert len(res["details"]) == 1
    assert res["details"][0]["status"] == STATUS_NOT_CHECKED


def test_regression_05_valid_subtotal_tax_discount():
    """Test 5: Subtotal 5000, Tax 900, Discount 100, Grand Total 5800 -> PASSED."""
    res = verify_total_sum(subtotal=5000, tax=900, total=5800, discount=100)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True
    assert res["severity"] == "INFO"
    assert res["difference"] == 0.0


def test_regression_06_arithmetic_error_in_grand_total():
    """Test 6: Subtotal 5000, Tax 900, Discount 100, Total 6000 -> FAILED (diff: 200)."""
    res = verify_total_sum(subtotal=5000, tax=900, total=6000, discount=100)
    assert res["status"] == STATUS_FAILED
    assert res["passed"] is False
    assert res["severity"] == "CRITICAL"
    assert res["difference"] == 200.0
    assert "diff: 200.00" in res["message"] or "diff: 200" in res["message"]


def test_regression_07_missing_tax_rate():
    """Test 7: Missing tax rate: Subtotal 5000, Tax 900, Tax Rate None -> NOT CHECKED."""
    res = verify_tax_calculation(subtotal=5000, tax=900, tax_rate=None)
    assert res["status"] == STATUS_NOT_CHECKED
    assert res["passed"] is True
    assert res["message"] == "Tax rate not provided. Rate-based tax verification could not be performed."


def test_regression_08_decimal_values():
    """Test 8: Fractional quantities and decimal amounts (2.5 hrs @ $10.50 = $26.25)."""
    items = [
        {"description": "Design consultation", "quantity": 2.5, "unitPrice": 10.50, "lineTotal": 26.25}
    ]
    res = verify_line_items_math(items)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True
    assert res["details"][0]["expected_total"] == 26.25


def test_regression_09_rounding_tolerance():
    """Test 9: Rounding tolerance allows minor sub-cent or 1-2 cent discrepancies."""
    # 3 items at 33.33 each = 99.99, stated line total 100.00 (diff: 0.01 <= 0.02 tolerance)
    items = [
        {"description": "Shared subscription 1/3", "quantity": 1, "unitPrice": 99.99, "lineTotal": 100.00}
    ]
    res = verify_line_items_math(items)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True


def test_regression_10_full_user_invoice_pipeline():
    """Test 10: Complete verification run for the user's actual invoice.
    
    Invoice contents:
    - Office chairs: 2 @ 1500 = 3000
    - Printer toner: 1 @ 2000 = 2000
    - Subtotal: 5000
    - Tax: 900 (Tax rate not specified on face)
    - Discount: 100
    - Grand Total: 5800
    """
    fields = {
        "invoiceNumber": "INV-2026-0001",
        "vendor": "Northstar Finance Ltd",
        "date": "2026-10-05",
        "subtotal": 5000.00,
        "tax": 900.00,
        "taxRate": None,
        "discount": 100.00,
        "total": 5800.00,
    }
    line_items = [
        {"description": "Office chairs", "quantity": 2.0, "unitPrice": 1500.0, "lineTotal": 3000.0},
        {"description": "Printer toner", "quantity": 1.0, "unitPrice": 2000.0, "lineTotal": 2000.0},
    ]

    verif = run_verification_rules(fields, line_items)

    # 1. Line item calculation must PASS
    rule_math = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_LINE_ITEM_MATH")
    assert rule_math["status"] == STATUS_PASSED
    assert rule_math["passed"] is True
    assert len(rule_math["details"]) == 2
    assert all(d["status"] == STATUS_PASSED for d in rule_math["details"])

    # 2. Subtotal sum must PASS (3000 + 2000 == 5000)
    rule_sub = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_SUBTOTAL_SUM")
    assert rule_sub["status"] == STATUS_PASSED
    assert rule_sub["passed"] is True

    # 3. Tax calculation must be NOT CHECKED (no tax rate)
    rule_tax = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_TAX_CALCULATION")
    assert rule_tax["status"] == STATUS_NOT_CHECKED
    assert rule_tax["message"] == "Tax rate not provided. Rate-based tax verification could not be performed."

    # 4. Total sum must PASS (5000 + 900 - 100 == 5800)
    rule_tot = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_TOTAL_SUM")
    assert rule_tot["status"] == STATUS_PASSED
    assert rule_tot["passed"] is True

    # 5. Non-negative total must PASS
    rule_pos = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_TOTAL_NON_NEGATIVE")
    assert rule_pos["status"] == STATUS_PASSED
    assert rule_pos["passed"] is True

    # 6. Overall verification should have ZERO critical failures
    assert verif["criticalFailuresCount"] == 0
    assert verif["allPassed"] is True


def test_regression_11_healing_raw_unhealed_format():
    """Test 11: Even if raw OCR previously passed concatenated quantities (e.g. 'Office chairs 2' with 1500/3000
    and 'Printer toner 1' with 2000/2000), self-healing must correctly normalize them and PASS.
    """
    raw_items = [
        {"description": "Office chairs 2", "quantity": 1500.0, "unitPrice": 3000.0, "lineTotal": 3000.0},
        {"description": "Printer toner 1", "quantity": 2000.0, "unitPrice": 2000.0, "lineTotal": 2000.0},
    ]
    res = verify_line_items_math(raw_items)
    assert res["status"] == STATUS_PASSED
    assert res["passed"] is True
    assert res["severity"] == "INFO"
    assert len(res["details"]) == 2

    # Check Office chairs
    d1 = res["details"][0]
    assert d1["status"] == STATUS_PASSED
    assert d1["quantity"] == 2.0
    assert d1["unit_price"] == 1500.0
    assert d1["line_total"] == 3000.0
    assert d1["expected_total"] == 3000.0

    # Check Printer toner
    d2 = res["details"][1]
    assert d2["status"] == STATUS_PASSED
    assert d2["quantity"] == 1.0
    assert d2["unit_price"] == 2000.0
    assert d2["line_total"] == 2000.0
    assert d2["expected_total"] == 2000.0

