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
    """Verify each line item: quantity * unitPrice == lineTotal within tolerance."""
    if not line_items:
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "passed": True,
            "severity": "INFO",
            "message": "No individual line items present to verify item-level arithmetic.",
            "expected": None,
            "actual": 0,
        }

    mismatches: List[str] = []
    checked_count = 0

    for idx, item in enumerate(line_items):
        qty = to_decimal(item.get("quantity"))
        unit_price = to_decimal(item.get("unitPrice"))
        line_total = to_decimal(item.get("lineTotal"))

        if qty is not None and unit_price is not None and line_total is not None:
            checked_count += 1
            expected_total = (qty * unit_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            diff = abs(expected_total - line_total)
            if diff > TOLERANCE:
                desc = item.get("description", f"Item #{idx+1}")[:25]
                mismatches.append(
                    f"'{desc}': {qty} * {unit_price} = {expected_total}, but line total is {line_total}"
                )

    if mismatches:
        return {
            "rule_id": "RULE_LINE_ITEM_MATH",
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Arithmetic mismatch in {len(mismatches)} line item(s): " + "; ".join(mismatches),
            "expected": "quantity * unitPrice == lineTotal",
            "actual": mismatches,
        }

    return {
        "rule_id": "RULE_LINE_ITEM_MATH",
        "passed": True,
        "severity": "INFO",
        "message": f"All {checked_count} line items verified mathematically.",
        "expected": "quantity * unitPrice == lineTotal",
        "actual": f"{checked_count} verified",
    }


# -------------------------------------------------------------------------
# Rule 2: Subtotal Sum (sum(lineTotals) == subtotal)
# -------------------------------------------------------------------------

def verify_subtotal_sum(
    subtotal: Optional[Union[Decimal, float, str]],
    line_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify that sum of line item totals equals subtotal."""
    dec_subtotal = to_decimal(subtotal)

    if dec_subtotal is None or not line_items:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "passed": True,
            "severity": "INFO",
            "message": "Subtotal or line items not present; subtotal sum check skipped.",
            "expected": None,
            "actual": None,
        }

    line_totals = [to_decimal(item.get("lineTotal")) for item in line_items]
    valid_totals = [lt for lt in line_totals if lt is not None]

    if not valid_totals:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "passed": True,
            "severity": "INFO",
            "message": "No valid line totals found to sum.",
            "expected": None,
            "actual": None,
        }

    computed_sum = sum(valid_totals, Decimal("0.00")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(computed_sum - dec_subtotal)

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_SUBTOTAL_SUM",
            "passed": False,
            "severity": "CRITICAL",
            "message": (
                f"Sum of line totals ({computed_sum}) does not match invoice subtotal ({dec_subtotal}). "
                f"Discrepancy: {diff}."
            ),
            "expected": float(computed_sum),
            "actual": float(dec_subtotal),
        }

    return {
        "rule_id": "RULE_SUBTOTAL_SUM",
        "passed": True,
        "severity": "INFO",
        "message": f"Line totals sum ({computed_sum}) matches subtotal ({dec_subtotal}) within tolerance.",
        "expected": float(computed_sum),
        "actual": float(dec_subtotal),
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
    dec_sub = to_decimal(subtotal)
    dec_tax = to_decimal(tax)
    dec_rate = to_decimal(tax_rate)

    if dec_sub is None or dec_tax is None or dec_rate is None:
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "passed": True,
            "severity": "INFO",
            "message": "Tax rate, tax amount, or subtotal not specified; tax rate calculation check skipped.",
            "expected": None,
            "actual": None,
        }

    expected_tax = (dec_sub * dec_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(expected_tax - dec_tax)

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_TAX_CALCULATION",
            "passed": False,
            "severity": "WARNING",
            "message": (
                f"Tax calculation mismatch: subtotal ({dec_sub}) * taxRate ({dec_rate}) "
                f"should be {expected_tax}, but invoice specifies {dec_tax} (diff: {diff})."
            ),
            "expected": float(expected_tax),
            "actual": float(dec_tax),
        }

    return {
        "rule_id": "RULE_TAX_CALCULATION",
        "passed": True,
        "severity": "INFO",
        "message": f"Tax calculation verified: {dec_sub} * {dec_rate} == {dec_tax}.",
        "expected": float(expected_tax),
        "actual": float(dec_tax),
    }


# -------------------------------------------------------------------------
# Rule 4: Total Sum (subtotal + tax == total)
# -------------------------------------------------------------------------

def verify_total_sum(
    subtotal: Optional[Union[Decimal, float, str]],
    tax: Optional[Union[Decimal, float, str]],
    total: Optional[Union[Decimal, float, str]],
) -> Dict[str, Any]:
    """Verify that subtotal + tax equals total."""
    dec_sub = to_decimal(subtotal)
    dec_tax = to_decimal(tax) if tax is not None else Decimal("0.00")
    dec_tot = to_decimal(total)

    if dec_tot is None:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "passed": False,
            "severity": "CRITICAL",
            "message": "Invoice total amount is missing.",
            "expected": "subtotal + tax",
            "actual": None,
        }

    if dec_sub is None:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "passed": True,
            "severity": "INFO",
            "message": "Subtotal not available; skipping subtotal + tax sum verification.",
            "expected": None,
            "actual": float(dec_tot),
        }

    expected_total = (dec_sub + dec_tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    diff = abs(expected_total - dec_tot)

    if diff > TOLERANCE:
        return {
            "rule_id": "RULE_TOTAL_SUM",
            "passed": False,
            "severity": "CRITICAL",
            "message": (
                f"Total mismatch: subtotal ({dec_sub}) + tax ({dec_tax}) = {expected_total}, "
                f"but stated total is {dec_tot} (diff: {diff})."
            ),
            "expected": float(expected_total),
            "actual": float(dec_tot),
        }

    return {
        "rule_id": "RULE_TOTAL_SUM",
        "passed": True,
        "severity": "INFO",
        "message": f"Stated total ({dec_tot}) matches subtotal + tax ({expected_total}) within tolerance.",
        "expected": float(expected_total),
        "actual": float(dec_tot),
    }


# -------------------------------------------------------------------------
# Rule 5: Required Fields Present
# -------------------------------------------------------------------------

def verify_required_fields(fields: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Verify mandatory fields are present (invoiceNumber, date, vendor, total)."""
    required = ["invoiceNumber", "date", "vendor", "total"]
    missing: List[str] = []

    for req in required:
        val = fields.get(req)
        if val is None or str(val).strip() == "":
            missing.append(req)

    if missing:
        return {
            "rule_id": "RULE_REQUIRED_FIELDS",
            "passed": False,
            "severity": "WARNING",
            "message": f"Missing {len(missing)} required invoice field(s): {', '.join(missing)}.",
            "expected": required,
            "actual": [k for k in required if k not in missing],
        }, missing

    return {
        "rule_id": "RULE_REQUIRED_FIELDS",
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
    if not date_str:
        return {
            "rule_id": "RULE_DATE_SANITY",
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
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Invoice date '{date_str}' is not in valid ISO format (YYYY-MM-DD).",
            "expected": "YYYY-MM-DD",
            "actual": date_str,
        }

    now = datetime.now(timezone.utc).date()

    # Check future date
    if dt > now:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "passed": False,
            "severity": "CRITICAL",
            "message": f"Invoice date '{dt}' is in the future (today is {now}).",
            "expected": f"<= {now}",
            "actual": str(dt),
        }

    # Check implausibly old date
    years_old = (now - dt).days / 365.25
    if years_old > MAX_AGE_YEARS:
        return {
            "rule_id": "RULE_DATE_SANITY",
            "passed": False,
            "severity": "WARNING",
            "message": f"Invoice date '{dt}' is {years_old:.1f} years old (exceeds {MAX_AGE_YEARS} year limit).",
            "expected": f"within {MAX_AGE_YEARS} years",
            "actual": str(dt),
        }

    return {
        "rule_id": "RULE_DATE_SANITY",
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
    """Check if total amount is a statistical outlier compared to user history.
    
    If fewer than MIN_HISTORY_FOR_OUTLIERS documents exist, marks rule 'not enough data'.
    """
    dec_current = to_decimal(current_total)
    if dec_current is None:
        return {
            "rule_id": "RULE_NUMERIC_OUTLIER",
            "passed": True,
            "severity": "INFO",
            "message": "Current document total is missing; outlier check skipped.",
            "expected": None,
            "actual": None,
        }

    # Clean historical amounts
    valid_history: List[float] = []
    for h in historical_totals:
        dec_h = to_decimal(h)
        if dec_h is not None and dec_h > Decimal(0):
            valid_history.append(float(dec_h))

    if len(valid_history) < MIN_HISTORY_FOR_OUTLIERS:
        return {
            "rule_id": "RULE_NUMERIC_OUTLIER",
            "passed": True,
            "severity": "INFO",
            "message": (
                f"Not enough data: user has {len(valid_history)} prior documents "
                f"(requires >= {MIN_HISTORY_FOR_OUTLIERS} for outlier detection)."
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
            "passed": False,
            "severity": "WARNING",
            "message": (
                f"Total amount ({dec_current}) deviates significantly from user history "
                f"(median: {median:.2f}, robust z-score: {z_score:.2f} > threshold {OUTLIER_Z_THRESHOLD})."
            ),
            "expected": f"z-score <= {OUTLIER_Z_THRESHOLD}",
            "actual": round(z_score, 2),
        }

    return {
        "rule_id": "RULE_NUMERIC_OUTLIER",
        "passed": True,
        "severity": "INFO",
        "message": f"Total amount ({dec_current}) is consistent with user history (z-score: {z_score:.2f}).",
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

    # 4. Total sum
    rule_results.append(
        verify_total_sum(
            fields.get("subtotal"),
            fields.get("tax"),
            fields.get("total"),
        )
    )

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

