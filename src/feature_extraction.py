"""Feature Extraction for FinVision ML Suspiciousness Classifier.

Features extracted:
1. total_mismatch: Discrepancy between stated total and expected (subtotal + tax).
2. subtotal_mismatch: Discrepancy between stated subtotal and sum of line totals.
3. tax_mismatch: Discrepancy between stated tax and (subtotal * tax_rate).
4. missing_fields: Count of missing required fields (0 to 4).
5. duplicate_flag: Binary indicator (1 if duplicate match found, 0 otherwise).
6. ocr_confidence: Overall OCR word recognition confidence (0.0 to 100.0).
7. image_anomaly_score: Aggregate CV anomaly indicator score (0.0 to 100.0).

Pure Python module: zero Firebase dependencies.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
import numpy as np


FEATURE_NAMES: List[str] = [
    "total_mismatch",
    "subtotal_mismatch",
    "tax_mismatch",
    "missing_fields",
    "duplicate_flag",
    "ocr_confidence",
    "image_anomaly_score",
]


def extract_features_from_document_state(
    fields: Dict[str, Any],
    verification_results: Dict[str, Any],
    ocr_confidence: float = 90.0,
    image_anomaly_score: float = 0.0,
    duplicate_flag: int = 0,
) -> Dict[str, float]:
    """Extract tabular ML feature vector from processed document components."""
    rule_results = {r["rule_id"]: r for r in verification_results.get("ruleResults", [])}

    # 1. Total mismatch
    tot_rule = rule_results.get("RULE_TOTAL_SUM")
    if tot_rule and not tot_rule.get("passed", True) and tot_rule.get("expected") is not None and tot_rule.get("actual") is not None:
        try:
            total_mismatch = float(abs(Decimal(str(tot_rule["expected"])) - Decimal(str(tot_rule["actual"]))))
        except Exception:
            total_mismatch = 1.0
    else:
        total_mismatch = 0.0

    # 2. Subtotal mismatch
    sub_rule = rule_results.get("RULE_SUBTOTAL_SUM")
    if sub_rule and not sub_rule.get("passed", True) and sub_rule.get("expected") is not None and sub_rule.get("actual") is not None:
        try:
            subtotal_mismatch = float(abs(Decimal(str(sub_rule["expected"])) - Decimal(str(sub_rule["actual"]))))
        except Exception:
            subtotal_mismatch = 1.0
    else:
        subtotal_mismatch = 0.0

    # 3. Tax mismatch
    tax_rule = rule_results.get("RULE_TAX_CALCULATION")
    if tax_rule and not tax_rule.get("passed", True) and tax_rule.get("expected") is not None and tax_rule.get("actual") is not None:
        try:
            tax_mismatch = float(abs(Decimal(str(tax_rule["expected"])) - Decimal(str(tax_rule["actual"]))))
        except Exception:
            tax_mismatch = 1.0
    else:
        tax_mismatch = 0.0

    # 4. Missing fields count
    missing_fields = float(len(verification_results.get("missingFields", [])))

    # 5. Duplicate flag
    dup_flag = float(1.0 if duplicate_flag else 0.0)

    # 6. OCR confidence
    ocr_conf = float(max(0.0, min(100.0, ocr_confidence)))

    # 7. Image anomaly score
    img_score = float(max(0.0, min(100.0, image_anomaly_score)))

    return {
        "total_mismatch": total_mismatch,
        "subtotal_mismatch": subtotal_mismatch,
        "tax_mismatch": tax_mismatch,
        "missing_fields": missing_fields,
        "duplicate_flag": dup_flag,
        "ocr_confidence": ocr_conf,
        "image_anomaly_score": img_score,
    }


def to_feature_array(features_dict: Dict[str, float]) -> pd.DataFrame:
    """Convert features dict into ordered pandas DataFrame ready for scikit-learn."""
    import pandas as pd
    vector = {name: [features_dict.get(name, 0.0)] for name in FEATURE_NAMES}
    return pd.DataFrame(vector)
