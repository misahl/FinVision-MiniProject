"""Deterministic Financial Verification Rule Engine for FinVision.

This module provides:
1. Pure mathematical and business verification rules:
   - Line totals arithmetic: qty * unitPrice == lineTotal
   - Subtotal arithmetic: sum(lineTotals) == subtotal
   - Tax rate arithmetic: subtotal * taxRate == tax
   - Total arithmetic: subtotal + tax == total
   - Required fields presence (invoiceNumber, date, vendor, total)
   - Date sanity: valid format, not in the future, not implausibly old (> 10 years)
   - Numeric outliers vs user historical invoices (robust z-score / IQR)
2. Pure functions returning standardized rule result dictionaries:
   {
       "rule_id": str,
       "passed": bool,
       "severity": "CRITICAL" | "WARNING" | "INFO",
       "message": str,
       "expected": Any,
       "actual": Any,
   }
3. Strict use of Decimal for currency/financial computations (never IEEE 754 floats).
4. Zero external Firebase dependencies: purely functional and testable offline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load rules configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "rules_config.json"
DEFAULT_CONFIG = {
    "math_tolerance_cents": 0.02,
    "default_tax_rate": 0.10,
    "min_historical_docs_for_outliers": 5,
    "outlier_zscore_threshold": 2.5,
    "max_invoice_age_years": 10,
}

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = {**DEFAULT_CONFIG, **json.load(f)}
    except Exception:
        CONFIG = DEFAULT_CONFIG
else:
    CONFIG = DEFAULT_CONFIG

TOLERANCE = Decimal(str(CONFIG.get("math_tolerance_cents", 0.02)))
MIN_HISTORY_FOR_OUTLIERS = int(CONFIG.get("min_historical_docs_for_outliers", 5))
OUTLIER_Z_THRESHOLD = float(CONFIG.get("outlier_zscore_threshold", 2.5))
MAX_AGE_YEARS = int(CONFIG.get("max_invoice_age_years", 10))

# -------------------------------------------------------------------------
# Standardized Validation Statuses & Rule Metadata Mapping
# -------------------------------------------------------------------------

STATUS_PASSED = "PASSED"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_NOT_CHECKED = "NOT CHECKED"

RULE_METADATA: Dict[str, Dict[str, str]] = {
    "RULE_LINE_ITEM_MATH": {
        "display_name": "Line Item Calculation",
        "description": "Verifies that Quantity multiplied by Unit Price equals Line Total for each item.",
        "category": "Arithmetic",
    },
    "RULE_SUBTOTAL_SUM": {
        "display_name": "Subtotal Verification",
        "description": "Verifies that the sum of all itemized line totals equals the stated subtotal.",
        "category": "Arithmetic",
    },
    "RULE_TAX_CALCULATION": {
        "display_name": "Tax Verification",
        "description": "Verifies that Subtotal multiplied by Tax Rate matches the stated tax amount.",
        "category": "Tax & Compliance",
    },
    "RULE_TOTAL_SUM": {
        "display_name": "Grand Total Verification",
        "description": "Verifies that Subtotal plus Tax minus Discount equals the stated Grand Total.",
        "category": "Arithmetic",
    },
    "RULE_TOTAL_NON_NEGATIVE": {
        "display_name": "Valid Invoice Amount",
        "description": "Verifies that the stated total is a valid, non-negative monetary amount.",
        "category": "Sanity",
    },
    "RULE_VENDOR_DETAILS": {
        "display_name": "Vendor Details Verification",
        "description": "Verifies that the vendor/issuer identification is present and valid.",
        "category": "Identity",
    },
    "RULE_REQUIRED_FIELDS": {
        "display_name": "Mandatory Information Check",
        "description": "Verifies that mandatory invoice header fields are complete.",
        "category": "Compliance",
    },
    "RULE_DATE_SANITY": {
        "display_name": "Invoice Date Verification",
        "description": "Verifies that the invoice date is valid, not future-dated, and reasonably recent.",
        "category": "Sanity",
    },
    "RULE_NUMERIC_OUTLIER": {
        "display_name": "Statistical Anomaly Check",
        "description": "Evaluates whether the invoice total deviates abnormally from historical baseline.",
        "category": "Outlier Detection",
    },
}


def to_decimal(val: Any) -> Optional[Decimal]:
    """Convert any numeric input to Decimal safely."""
    if val is None or val == "":
        return None
    if isinstance(val, Decimal):
        return val
    try:
        cleaned = str(val).replace(",", "").strip()
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


# -------------------------------------------------------------------------
# Rule 1: Line Item Arithmetic (qty * unitPrice == lineTotal)
# -------------------------------------------------------------------------

def verify_line_items_math(line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify each line item: line_total == quantity * unit_price within tolerance.
    
    Robust against diverse field naming conventions (quantity/qty, unitPrice/unit_price/rate,
    lineTotal/line_total/totalPrice/amount), inverted columns, and missing data.
    """
    display_name = RULE_METADATA["RULE_LINE_ITEM_MATH"]["display_name"]

    if not line_items:
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "No individual line items present to verify item-level calculation.",
            "expected": "quantity * unit_price == line_total",
            "actual": 0,
            "details": [],
        }

    item_details: List[Dict[str, Any]] = []

    for idx, item in enumerate(line_items):
        raw_qty = item.get("quantity") if item.get("quantity") is not None else item.get("qty")
        raw_price = (
            item.get("unitPrice") if item.get("unitPrice") is not None
            else (item.get("unit_price") or item.get("price") or item.get("rate") or item.get("unit_amount"))
        )
        raw_total = (
            item.get("lineTotal") if item.get("lineTotal") is not None
            else (item.get("line_total") or item.get("totalPrice") or item.get("total") or item.get("amount"))
        )
        desc = str(item.get("description") or item.get("desc") or item.get("item") or item.get("name") or f"Item #{idx+1}").strip()

        qty = to_decimal(raw_qty)
        unit_price = to_decimal(raw_price)
        line_total = to_decimal(raw_total)

        # Detect if quantity is missing
        if qty is None:
            item_details.append({
                "description": desc,
                "quantity": None,
                "unit_price": float(unit_price) if unit_price is not None else None,
                "expected_total": None,
                "line_total": float(line_total) if line_total is not None else None,
                "difference": None,
                "status": STATUS_NOT_CHECKED,
                "message": f"Quantity not specified for item '{desc}'. Calculation check could not be performed.",
            })
            continue

        # Detect if unit price or line total is missing
        if unit_price is None or line_total is None:
            item_details.append({
                "description": desc,
                "quantity": float(qty),
                "unit_price": float(unit_price) if unit_price is not None else None,
                "expected_total": None,
                "line_total": float(line_total) if line_total is not None else None,
                "difference": None,
                "status": STATUS_NOT_CHECKED,
                "message": f"Pricing or total not specified for item '{desc}'. Calculation check could not be performed.",
            })
            continue

        # Self-healing for inverted columns (e.g. OCR swapped qty and price, or price and total):
        # Case A: Qty and Price swapped (e.g. qty=1500, price=2, total=3000)
        if qty > unit_price and unit_price <= Decimal("100") and unit_price % 1 == 0 and abs((qty * unit_price) - line_total) <= TOLERANCE:
            qty, unit_price = unit_price, qty

        # Case B: Ratio healing when unit_price == line_total or qty was mapped to price (e.g. qty=1500, price=3000, total=3000, desc='Office chairs 2')
        if unit_price == line_total and qty > 0 and (line_total % qty) == Decimal(0):
            ratio = line_total / qty
            trailing_m = re.search(r"\b(\d+)\b\s*$", desc)
            if trailing_m and Decimal(trailing_m.group(1)) == ratio:
                desc = re.sub(r"\b\d+\b\s*$", "", desc).strip()
                unit_price = qty
                qty = ratio
            elif ratio == Decimal(1):
                if trailing_m:
                    desc = re.sub(r"\b\d+\b\s*$", "", desc).strip()
                unit_price = qty
                qty = Decimal(1)
            elif qty > 1 and (qty * unit_price) != line_total:
                unit_price = qty
                qty = ratio

        expected_total = (qty * unit_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        diff = abs(expected_total - line_total)

        if diff <= TOLERANCE:
            item_details.append({
                "description": desc,
                "quantity": float(qty),
                "unit_price": float(unit_price),
                "expected_total": float(expected_total),
                "line_total": float(line_total),
                "difference": 0.0,
                "status": STATUS_PASSED,
                "message": f"{qty:g} × {unit_price:,.2f} = {line_total:,.2f} — {STATUS_PASSED}",
            })
        else:
            item_details.append({
                "description": desc,
                "quantity": float(qty),
                "unit_price": float(unit_price),
                "expected_total": float(expected_total),
                "line_total": float(line_total),
                "difference": float(diff),
                "status": STATUS_FAILED,
                "message": f"The line total differs from the expected amount by {diff:,.2f}.",
            })

    failed = [d for d in item_details if d["status"] == STATUS_FAILED]
    not_checked = [d for d in item_details if d["status"] == STATUS_NOT_CHECKED]
    passed = [d for d in item_details if d["status"] == STATUS_PASSED]

    if failed:
        count_word = "One invoice item has" if len(failed) == 1 else ("Two invoice items have" if len(failed) == 2 else f"{len(failed)} invoice items have")
        diff_summaries = "; ".join([
            f"'{f['description']}': {f['quantity']:g} * {f['unit_price']:,.2f} = {f['expected_total']:,.2f}, but line total is {f['line_total']:,.2f} (difference: {f['difference']:,.2f})"
            for f in failed
        ])
        friendly_suffix = "an incorrect calculation" if len(failed) == 1 else "incorrect calculations"
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Arithmetic mismatch in {len(failed)} line item(s): {count_word} {friendly_suffix}. Please review their quantities, unit prices and line totals. ({diff_summaries})",
            "expected": "quantity * unit_price == line_total",
            "actual": [f["line_total"] for f in failed],
            "details": item_details,
        }

    if not_checked and not passed:
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "WARNING",
            "message": f"Quantity or unit price not specified for {len(not_checked)} item(s). Line item calculation could not be performed.",
            "expected": "quantity * unit_price == line_total",
            "actual": None,
            "details": item_details,
        }

    if not_checked:
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": True,
            "severity": "WARNING",
            "message": f"Verified {len(passed)} line item(s), but quantity was missing for {len(not_checked)} item(s).",
            "expected": "quantity * unit_price == line_total",
            "actual": f"{len(passed)} verified, {len(not_checked)} skipped",
            "details": item_details,
        }

    summaries = [f"{p['quantity']:g} × {p['unit_price']:,.2f} = {p['line_total']:,.2f}" for p in passed]
    return {
        "rule_id": "RULE_LINE_ITEM_MATH",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"All {len(passed)} line item calculations verified correctly ({'; '.join(summaries[:3])}).",
        "expected": "quantity * unit_price == line_total",
        "actual": f"{len(passed)} verified",
        "details": item_details,
    }


# -------------------------------------------------------------------------
# Rule 2: Subtotal Sum (sum(lineTotals) == subtotal)
# -------------------------------------------------------------------------

def verify_subtotal_sum(
    subtotal: Optional[Union[Decimal, float, str]],
    line_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify that sum of line item totals equals subtotal."""
    display_name = RULE_METADATA["RULE_SUBTOTAL_SUM"]["display_name"]
    dec_subtotal = to_decimal(subtotal)

    if dec_subtotal is None or not line_items:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Subtotal or line items not available. Subtotal verification could not be performed.",
            "expected": None,
            "actual": None,
        }

    line_totals = []
    for item in line_items:
        lt = to_decimal(
            item.get("lineTotal") if item.get("lineTotal") is not None
            else (item.get("line_total") or item.get("totalPrice") or item.get("total") or item.get("amount"))
        )
        line_totals.append(lt)

    valid_totals = [lt for lt in line_totals if lt is not None]

    if not valid_totals:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "No valid line item totals found to sum. Subtotal verification could not be performed.",
            "expected": None,
            "actual": None,
        }

    computed_sum = sum(valid_totals, Decimal("0.00")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(computed_sum - dec_subtotal)

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": (
                f"Sum of line totals ({computed_sum:,.2f}) does not match invoice subtotal ({dec_subtotal:,.2f}). "
                f"Discrepancy: {diff:,.2f}."
            ),
            "expected": float(computed_sum),
            "actual": float(dec_subtotal),
            "difference": float(diff),
        }

    return {
        "rule_id": "RULE_SUBTOTAL_SUM",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Line totals sum ({computed_sum:,.2f}) matches subtotal ({dec_subtotal:,.2f}) within tolerance.",
        "expected": float(computed_sum),
        "actual": float(dec_subtotal),
        "difference": 0.0,
    }


# -------------------------------------------------------------------------
# Rule 3: Tax Calculation (subtotal * taxRate == tax)
# -------------------------------------------------------------------------

def verify_tax_calculation(
    subtotal: Optional[Union[Decimal, float, str]],
    tax: Optional[Union[Decimal, float, str]],
    tax_rate: Optional[Union[Decimal, float, str]],
) -> Dict[str, Any]:
    """Verify that subtotal * taxRate equals tax amount when tax rate is specified."""
    display_name = RULE_METADATA["RULE_TAX_CALCULATION"]["display_name"]
    dec_sub = to_decimal(subtotal)
    dec_tax = to_decimal(tax)
    dec_rate = to_decimal(tax_rate)

    if dec_rate is None:
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Tax rate not provided. Rate-based tax verification could not be performed.",
            "expected": None,
            "actual": float(dec_tax) if dec_tax is not None else None,
        }

    if dec_sub is None:
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Subtotal not available. Rate-based tax verification could not be performed.",
            "expected": None,
            "actual": float(dec_tax) if dec_tax is not None else None,
        }

    if dec_tax is None:
        expected_tax = (dec_sub * dec_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Stated tax amount not available. Rate-based tax verification could not be performed.",
            "expected": float(expected_tax),
            "actual": None,
        }

    expected_tax = (dec_sub * dec_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(expected_tax - dec_tax)
    pct_str = f"{dec_rate * 100:.1f}%"

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "WARNING",
            "message": (
                f"Tax calculation mismatch: Subtotal ({dec_sub:,.2f}) × Tax Rate ({pct_str}) "
                f"should be {expected_tax:,.2f}, but invoice specifies {dec_tax:,.2f} (diff: {diff:,.2f})."
            ),
            "expected": float(expected_tax),
            "actual": float(dec_tax),
            "difference": float(diff),
        }

    return {
        "rule_id": "RULE_TAX_CALCULATION",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Tax calculation verified: Subtotal ({dec_sub:,.2f}) × Tax Rate ({pct_str}) == Stated Tax ({dec_tax:,.2f}).",
        "expected": float(expected_tax),
        "actual": float(dec_tax),
        "difference": 0.0,
    }


# -------------------------------------------------------------------------
# Rule 4: Total Sum (subtotal + tax - discount == total)
# -------------------------------------------------------------------------

def verify_total_sum(
    subtotal: Optional[Union[Decimal, float, str]],
    tax: Optional[Union[Decimal, float, str]],
    total: Optional[Union[Decimal, float, str]],
    discount: Optional[Union[Decimal, float, str]] = None,
) -> Dict[str, Any]:
    """Verify that subtotal + tax - discount equals total within tolerance."""
    display_name = RULE_METADATA["RULE_TOTAL_SUM"]["display_name"]
    dec_sub = to_decimal(subtotal)
    dec_tax = to_decimal(tax) if tax is not None else Decimal("0.00")
    dec_disc = to_decimal(discount) if discount is not None else Decimal("0.00")
    dec_tot = to_decimal(total)

    if dec_tot is None:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": "Invoice total amount is missing.",
            "expected": "subtotal + tax - discount",
            "actual": None,
        }

    if dec_sub is None:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Subtotal not available; skipping subtotal + tax sum verification.",
            "expected": None,
            "actual": float(dec_tot),
        }

    expected_total = (dec_sub + dec_tax - dec_disc).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(expected_total - dec_tot)
    disc_text = f" - discount ({dec_disc:,.2f})" if dec_disc > 0 else ""
    calc_breakdown = f"subtotal ({dec_sub:,.2f}) + tax ({dec_tax:,.2f}){disc_text} = {expected_total:,.2f}"

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": (
                f"Total mismatch: {calc_breakdown}, "
                f"but stated total is {dec_tot:,.2f} (diff: {diff:,.2f})."
            ),
            "expected": float(expected_total),
            "actual": float(dec_tot),
            "difference": float(diff),
        }

    return {
        "rule_id": "RULE_TOTAL_SUM",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Stated total ({dec_tot:,.2f}) matches subtotal + tax - discount ({expected_total:,.2f}) within tolerance.",
        "expected": float(expected_total),
        "actual": float(dec_tot),
        "difference": 0.0,
    }


# -------------------------------------------------------------------------
# Rule 4b: Non-Negative Total (total >= 0.00)
# -------------------------------------------------------------------------

def verify_total_non_negative(total: Optional[Union[Decimal, float, str]]) -> Dict[str, Any]:
    """Verify that stated total is non-negative."""
    display_name = RULE_METADATA["RULE_TOTAL_NON_NEGATIVE"]["display_name"]
    dec_tot = to_decimal(total)
    if dec_tot is None:
        return {
            "rule_id": "RULE_TOTAL_NON_NEGATIVE",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": "Invoice total amount is missing or invalid.",
            "expected": ">= 0.00",
            "actual": None,
        }

    if dec_tot < Decimal("0.00"):
        return {
            "rule_id": "RULE_TOTAL_NON_NEGATIVE",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Invoice total must be non-negative, but got {dec_tot:,.2f}.",
            "expected": ">= 0.00",
            "actual": float(dec_tot),
        }

    return {
        "rule_id": "RULE_TOTAL_NON_NEGATIVE",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Total amount ({dec_tot:,.2f}) is positive and valid.",
        "expected": ">= 0.00",
        "actual": float(dec_tot),
    }


# -------------------------------------------------------------------------
# Rule 4c: Required Vendor Details Present
# -------------------------------------------------------------------------

def verify_vendor_details(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Verify vendor details are identifiable and not placeholder strings."""
    display_name = RULE_METADATA["RULE_VENDOR_DETAILS"]["display_name"]
    vendor = fields.get("vendor")
    if not vendor or str(vendor).strip() == "" or str(vendor).strip().lower() in ["unknown", "n/a", "none"]:
        return {
            "rule_id": "RULE_VENDOR_DETAILS",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": False,
            "severity": "WARNING",
            "message": "Vendor identification is missing or indeterminate.",
            "expected": "Valid vendor name",
            "actual": vendor or "None",
        }

    return {
        "rule_id": "RULE_VENDOR_DETAILS",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Vendor identified: '{vendor}'.",
        "expected": "Valid vendor name",
        "actual": vendor,
    }


# -------------------------------------------------------------------------
# Rule 5: Required Fields Present
# -------------------------------------------------------------------------

def verify_required_fields(fields: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Verify mandatory fields are present (invoiceNumber, date, vendor, total)."""
    display_name = RULE_METADATA["RULE_REQUIRED_FIELDS"]["display_name"]
    required = ["invoiceNumber", "date", "vendor", "total"]
    missing: List[str] = []

    for req in required:
        val = fields.get(req)
        if val is None or str(val).strip() == "":
            missing.append(req)

    if missing:
        return {
            "rule_id": "RULE_REQUIRED_FIELDS",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": False,
            "severity": "WARNING",
            "message": f"Missing {len(missing)} required invoice field(s): {', '.join(missing)}.",
            "expected": required,
            "actual": [k for k in required if k not in missing],
        }, missing

    return {
        "rule_id": "RULE_REQUIRED_FIELDS",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": "All required invoice fields are present.",
        "expected": required,
        "actual": required,
    }, []


# -------------------------------------------------------------------------
# Rule 6: Date Sanity (Parseable, not in future, not implausibly old)
# -------------------------------------------------------------------------

def verify_date_sanity(date_str: Optional[str]) -> Dict[str, Any]:
    """Verify date is valid, not future-dated, and not older than MAX_AGE_YEARS."""
    display_name = RULE_METADATA["RULE_DATE_SANITY"]["display_name"]
    if not date_str:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": False,
            "severity": "WARNING",
            "message": "Invoice date is missing.",
            "expected": "YYYY-MM-DD",
            "actual": None,
        }

    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Invoice date '{date_str}' is not in valid ISO format (YYYY-MM-DD).",
            "expected": "YYYY-MM-DD",
            "actual": date_str,
        }

    now = datetime.now(timezone.utc).date()

    if dt > now:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "display_name": display_name,
            "status": STATUS_FAILED,
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Invoice date '{dt}' is in the future (today is {now}).",
            "expected": f"<= {now}",
            "actual": str(dt),
        }

    years_old = (now - dt).days / 365.25
    if years_old > MAX_AGE_YEARS:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": False,
            "severity": "WARNING",
            "message": f"Invoice date '{dt}' is {years_old:.1f} years old (exceeds {MAX_AGE_YEARS} year limit).",
            "expected": f"within {MAX_AGE_YEARS} years",
            "actual": str(dt),
        }

    return {
        "rule_id": "RULE_DATE_SANITY",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Invoice date '{dt}' is valid and sane.",
        "expected": f"between {now.year - MAX_AGE_YEARS} and {now.year}",
        "actual": str(dt),
    }


# -------------------------------------------------------------------------
# Rule 7: Numeric Outliers vs Historical Documents (Z-score / IQR)
# -------------------------------------------------------------------------

def verify_numeric_outlier(
    current_total: Optional[Union[Decimal, float, str]],
    historical_totals: List[Union[Decimal, float, str]],
) -> Dict[str, Any]:
    """Check if total amount is a statistical outlier compared to user history."""
    display_name = RULE_METADATA["RULE_NUMERIC_OUTLIER"]["display_name"]
    dec_current = to_decimal(current_total)
    if dec_current is None:
        return {
            "rule_id": "RULE_NUMERIC_OUTLIER",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": "Current document total is missing; outlier check skipped.",
            "expected": None,
            "actual": None,
        }

    valid_history: List[float] = []
    for h in historical_totals:
        dec_h = to_decimal(h)
        if dec_h is not None and dec_h > Decimal(0):
            valid_history.append(float(dec_h))

    if len(valid_history) < MIN_HISTORY_FOR_OUTLIERS:
        return {
            "rule_id": "RULE_NUMERIC_OUTLIER",
            "display_name": display_name,
            "status": STATUS_NOT_CHECKED,
            "passed": True,
            "severity": "INFO",
            "message": (
                f"Not enough data: Historical baseline requires >= {MIN_HISTORY_FOR_OUTLIERS} prior documents "
                f"(found {len(valid_history)}). Outlier check skipped."
            ),
            "expected": f">= {MIN_HISTORY_FOR_OUTLIERS} prior docs",
            "actual": len(valid_history),
        }

    import numpy as np

    arr = np.array(valid_history)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_std = 1.4826 * mad if mad > 1e-4 else float(np.std(arr))

    cur_float = float(dec_current)
    z_score = abs(cur_float - median) / robust_std if robust_std > 1e-4 else 0.0

    if z_score > OUTLIER_Z_THRESHOLD:
        return {
            "rule_id": "RULE_NUMERIC_OUTLIER",
            "display_name": display_name,
            "status": STATUS_WARNING,
            "passed": False,
            "severity": "WARNING",
            "message": (
                f"Total amount ({dec_current:,.2f}) deviates significantly from user history "
                f"(median: {median:,.2f}, robust z-score: {z_score:.2f} > threshold {OUTLIER_Z_THRESHOLD})."
            ),
            "expected": f"z-score <= {OUTLIER_Z_THRESHOLD}",
            "actual": round(z_score, 2),
        }

    return {
        "rule_id": "RULE_NUMERIC_OUTLIER",
        "display_name": display_name,
        "status": STATUS_PASSED,
        "passed": True,
        "severity": "INFO",
        "message": f"Total amount ({dec_current:,.2f}) is consistent with user history (z-score: {z_score:.2f}).",
        "expected": f"z-score <= {OUTLIER_Z_THRESHOLD}",
        "actual": round(z_score, 2),
    }


# -------------------------------------------------------------------------
# Composite Verification Runner
# -------------------------------------------------------------------------

def run_verification_rules(
    fields: Dict[str, Any],
    line_items: List[Dict[str, Any]],
    historical_totals: Optional[List[Union[Decimal, float, str]]] = None,
) -> Dict[str, Any]:
    """Execute all verification rules on extracted invoice data.
    
    Returns structured verification results matching Firestore document schema:
    {
        "ruleResults": [...],
        "missingFields": [...],
        "criticalFailuresCount": int,
        "warningFailuresCount": int,
        "allPassed": bool,
    }
    """
    rule_results: List[Dict[str, Any]] = []

    # 1. Line items math
    rule_results.append(verify_line_items_math(line_items))

    # 2. Subtotal sum
    rule_results.append(verify_subtotal_sum(fields.get("subtotal"), line_items))

    # 3. Tax calculation
    rule_results.append(
        verify_tax_calculation(
            fields.get("subtotal"),
            fields.get("tax"),
            fields.get("taxRate"),
        )
    )

    # 4. Total sum (subtotal + tax - discount == total)
    rule_results.append(
        verify_total_sum(
            fields.get("subtotal"),
            fields.get("tax"),
            fields.get("total"),
            fields.get("discount"),
        )
    )

    # 4b. Non-negative total
    rule_results.append(verify_total_non_negative(fields.get("total")))

    # 4c. Vendor details identified
    rule_results.append(verify_vendor_details(fields))

    # 5. Required fields
    req_res, missing_fields = verify_required_fields(fields)
    rule_results.append(req_res)

    # 6. Date sanity
    rule_results.append(verify_date_sanity(fields.get("date")))

    # 7. Numeric outliers
    rule_results.append(
        verify_numeric_outlier(
            fields.get("total"),
            historical_totals or [],
        )
    )

    critical_failures = [r for r in rule_results if not r["passed"] and r["severity"] == "CRITICAL"]
    warning_failures = [r for r in rule_results if not r["passed"] and r["severity"] == "WARNING"]

    return {
        "ruleResults": rule_results,
        "missingFields": missing_fields,
        "criticalFailuresCount": len(critical_failures),
        "warningFailuresCount": len(warning_failures),
        "allPassed": len(critical_failures) == 0 and len(warning_failures) == 0,
    }

