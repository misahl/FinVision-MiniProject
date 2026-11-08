"""Unit tests for FinVision Feature Extraction and ML Inference."""

from pathlib import Path
import pytest

from src.feature_extraction import (
    FEATURE_NAMES,
    extract_features_from_document_state,
    to_feature_array,
)
from src.predict import predict_document_risk


def test_feature_extraction_clean_document():
    """Verify feature extraction on clean document produces 0 mismatches."""
    fields = {"subtotal": 100.0, "tax": 10.0, "total": 110.0, "vendor": "Acme", "date": "2024-01-01", "invoiceNumber": "INV-1"}
    verification = {"ruleResults": [], "missingFields": []}

    feats = extract_features_from_document_state(fields, verification, ocr_confidence=95.0, image_anomaly_score=5.0)

    for name in FEATURE_NAMES:
        assert name in feats

    assert feats["total_mismatch"] == 0.0
    assert feats["subtotal_mismatch"] == 0.0
    assert feats["missing_fields"] == 0.0
    assert feats["duplicate_flag"] == 0.0
    assert feats["ocr_confidence"] == 95.0
    assert feats["image_anomaly_score"] == 5.0


def test_feature_extraction_mismatched_document():
    """Verify feature extraction accurately reflects verification rule failures."""
    fields = {"total": 150.0}
    verification = {
        "ruleResults": [
            {"rule_id": "RULE_TOTAL_SUM", "passed": False, "expected": 110.0, "actual": 150.0},
            {"rule_id": "RULE_TAX_CALCULATION", "passed": False, "expected": 10.0, "actual": 30.0},
        ],
        "missingFields": ["vendor", "date"],
    }

    feats = extract_features_from_document_state(fields, verification, duplicate_flag=1)

    assert feats["total_mismatch"] == 40.0
    assert feats["tax_mismatch"] == 20.0
    assert feats["missing_fields"] == 2.0
    assert feats["duplicate_flag"] == 1.0


def test_predict_document_risk_inference():
    """Verify ML inference returns structured risk probability and experimental tag."""
    clean_feats = {
        "total_mismatch": 0.0,
        "subtotal_mismatch": 0.0,
        "tax_mismatch": 0.0,
        "missing_fields": 0.0,
        "duplicate_flag": 0.0,
        "ocr_confidence": 95.0,
        "image_anomaly_score": 5.0,
    }
    pred_clean = predict_document_risk(clean_feats)
    assert pred_clean["prediction"] == 0
    assert pred_clean["verdict"] == "normal"
    assert pred_clean["experimental"] is True
    assert 0.0 <= pred_clean["probability"] <= 1.0

    suspicious_feats = {
        "total_mismatch": 150.0,
        "subtotal_mismatch": 50.0,
        "tax_mismatch": 30.0,
        "missing_fields": 2.0,
        "duplicate_flag": 1.0,
        "ocr_confidence": 60.0,
        "image_anomaly_score": 75.0,
    }
    pred_suspicious = predict_document_risk(suspicious_feats)
    assert pred_suspicious["prediction"] == 1
    assert pred_suspicious["verdict"] == "suspicious"
    assert pred_suspicious["probability"] > 0.5
    assert pred_suspicious["experimental"] is True

