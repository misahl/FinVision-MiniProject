"""Mandatory Test Dataset & End-to-End Verification Pipeline Tests for FinVision 2.0.

Explicitly tests all 5 synthetic document cases:
1. Valid Invoice (Straight-through pass)
2. Arithmetic Mismatch (Line total or subtotal calculation discrepancy)
3. Duplicate Invoice (Exact cryptographic or semantic collision)
4. Missing Invoice Fields (Missing mandatory fields)
5. Suspiciously Modified Sample (Elevated computer vision / image anomaly)

All test documents are clearly documented as synthetic.
"""

from decimal import Decimal
import pytest

from src import (
    verification,
    duplicate_detection,
    feature_extraction,
    predict,
    risk_assessment,
)


@pytest.fixture
def synthetic_valid_invoice():
    """Synthetic Document #1: Fully valid clean invoice with zero discrepancies."""
    return {
        "sampleId": "SYNTH_001_VALID",
        "description": "Synthetic valid invoice with matching subtotal, tax, and item totals.",
        "fields": {
            "invoiceNumber": "INV-2026-001",
            "vendor": "Acme Global Solutions Inc.",
            "customer": "FinVision Financial LLC",
            "date": "2026-10-15",
            "dueDate": "2026-11-15",
            "currency": "USD",
            "subtotal": 1000.00,
            "tax": 100.00,
            "taxRate": 0.10,
            "discount": 0.00,
            "total": 1100.00,
        },
        "lineItems": [
            {"description": "Enterprise Platform License", "quantity": 1.0, "unitPrice": 600.00, "lineTotal": 600.00},
            {"description": "Technical Advisory Hours", "quantity": 4.0, "unitPrice": 100.00, "lineTotal": 400.00},
        ],
        "fileMeta": {
            "sha256": "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
            "pHash": "8f8f8f8f8f8f8f8f",
        },
        "cvScore": 5.0,
    }


@pytest.fixture
def synthetic_math_mismatch_invoice():
    """Synthetic Document #2: Invoice with intentional arithmetic mismatch (total inflated)."""
    return {
        "sampleId": "SYNTH_002_MATH_MISMATCH",
        "description": "Synthetic invoice with arithmetic discrepancy: subtotal + tax != stated total.",
        "fields": {
            "invoiceNumber": "INV-2026-002",
            "vendor": "Apex Logistics Corp",
            "customer": "FinVision Financial LLC",
            "date": "2026-10-20",
            "currency": "USD",
            "subtotal": 500.00,
            "tax": 50.00,
            "taxRate": 0.10,
            "discount": 0.00,
            "total": 850.00,  # 500 + 50 = 550 != 850 (Inflation of $300)
        },
        "lineItems": [
            {"description": "Freight Handling", "quantity": 2.0, "unitPrice": 250.00, "lineTotal": 500.00},
        ],
        "fileMeta": {
            "sha256": "222233334444555566667777888899990000aaaabbbbccccddddeeeeffff1111",
            "pHash": "a1a1a1a1a1a1a1a1",
        },
        "cvScore": 8.0,
    }


@pytest.fixture
def synthetic_missing_fields_invoice():
    """Synthetic Document #3: Invoice with missing required fields (no vendor, no date)."""
    return {
        "sampleId": "SYNTH_003_MISSING_FIELDS",
        "description": "Synthetic invoice missing vendor name and invoice date.",
        "fields": {
            "invoiceNumber": "INV-2026-003",
            "vendor": None,
            "customer": "FinVision Financial LLC",
            "date": "",
            "subtotal": 200.00,
            "tax": 20.00,
            "total": 220.00,
        },
        "lineItems": [],
        "fileMeta": {
            "sha256": "33334444555566667777888899990000aaaabbbbccccddddeeeeffff11112222",
            "pHash": "b2b2b2b2b2b2b2b2",
        },
        "cvScore": 12.0,
    }


def test_synthetic_valid_invoice_pipeline(synthetic_valid_invoice):
    """Test Case 1: Valid invoice achieves straight-through approval with LOW risk."""
    inv = synthetic_valid_invoice
    verif = verification.run_verification_rules(inv["fields"], inv["lineItems"])
    assert verif["criticalFailuresCount"] == 0
    assert len(verif["missingFields"]) == 0

    dup = duplicate_detection.find_duplicates(
        {"sha256": inv["fileMeta"]["sha256"], "pHash": inv["fileMeta"]["pHash"], "fields": inv["fields"]},
        candidate_docs=[],
    )
    assert dup["hasDuplicate"] is False

    feats = feature_extraction.extract_features_from_document_state(
        fields=inv["fields"],
        verification_results=verif,
        ocr_confidence=96.0,
        image_anomaly_score=inv["cvScore"],
        duplicate_flag=0,
    )
    ml_res = predict.predict_document_risk(feats)

    risk = risk_assessment.compute_document_risk(
        verification_results=verif,
        duplicate_results=dup,
        cv_results={"cvScore": inv["cvScore"], "indicators": []},
        ml_results=ml_res,
    )

    assert risk["level"] == "LOW"
    assert risk["score"] < 25.0
    assert "straight-through" in risk["recommendedAction"].lower()


def test_synthetic_arithmetic_mismatch_pipeline(synthetic_math_mismatch_invoice):
    """Test Case 2: Arithmetic discrepancy triggers CRITICAL failure and HIGH risk."""
    inv = synthetic_math_mismatch_invoice
    verif = verification.run_verification_rules(inv["fields"], inv["lineItems"])

    # Must fail total sum check
    total_rule = next(r for r in verif["ruleResults"] if r["rule_id"] == "RULE_TOTAL_SUM")
    assert total_rule["passed"] is False
    assert total_rule["severity"] == "CRITICAL"
    assert verif["criticalFailuresCount"] >= 1

    risk = risk_assessment.compute_document_risk(
        verification_results=verif,
        duplicate_results={"hasDuplicate": False, "duplicateMatches": []},
        cv_results={"cvScore": inv["cvScore"], "indicators": []},
        ml_results={"probability": 0.75, "verdict": "suspicious"},
    )

    assert risk["level"] == "HIGH"
    assert risk["score"] >= 60.0
    assert any("Total mismatch" in r or "Critical rule variance" in r for r in risk["reasons"])


def test_synthetic_duplicate_collision_pipeline(synthetic_valid_invoice):
    """Test Case 3: Duplicate invoice collision triggers duplicate flag and elevated risk."""
    inv = synthetic_valid_invoice
    existing_docs = [
        {
            "docId": "doc_prior_999",
            "createdAt": "2026-10-01T12:00:00Z",
            "sha256": inv["fileMeta"]["sha256"],
            "pHash": inv["fileMeta"]["pHash"],
            "fields": inv["fields"],
        }
    ]

    dup = duplicate_detection.find_duplicates(
        {"sha256": inv["fileMeta"]["sha256"], "pHash": inv["fileMeta"]["pHash"], "fields": inv["fields"]},
        candidate_docs=existing_docs,
    )
    assert dup["hasDuplicate"] is True
    assert dup["matchCount"] >= 1
    assert dup["duplicateMatches"][0]["matchType"] == "EXACT_SHA256"
    assert dup["duplicateMatches"][0]["similarityScore"] == 100.0

    verif = verification.run_verification_rules(inv["fields"], inv["lineItems"])
    risk = risk_assessment.compute_document_risk(
        verification_results=verif,
        duplicate_results=dup,
        cv_results={"cvScore": 5.0, "indicators": []},
        ml_results={"probability": 0.10, "verdict": "normal"},
    )
    assert risk["score"] >= 40.0
    assert any("Duplicate" in r for r in risk["reasons"])


def test_synthetic_missing_fields_pipeline(synthetic_missing_fields_invoice):
    """Test Case 4: Missing fields triggers required field warning and identified missing items."""
    inv = synthetic_missing_fields_invoice
    verif = verification.run_verification_rules(inv["fields"], inv["lineItems"])

    assert len(verif["missingFields"]) >= 2
    assert "vendor" in verif["missingFields"]
    assert "date" in verif["missingFields"]

    risk = risk_assessment.compute_document_risk(
        verification_results=verif,
        duplicate_results={"hasDuplicate": False, "duplicateMatches": []},
        cv_results={"cvScore": 10.0, "indicators": []},
        ml_results={"probability": 0.45, "verdict": "normal"},
    )
    assert any("missing" in r.lower() for r in risk["reasons"])


def test_synthetic_cv_anomaly_pipeline(synthetic_valid_invoice):
    """Test Case 5: Document with elevated computer-vision tampering indicators."""
    inv = synthetic_valid_invoice
    verif = verification.run_verification_rules(inv["fields"], inv["lineItems"])

    high_cv_results = {
        "cvScore": 78.5,
        "indicators": [
            {"indicator": "Error Level Analysis (ELA)", "score": 85.0, "details": "High compression residual variance in text regions."},
            {"indicator": "Local Noise Inconsistency", "score": 72.0, "details": "Inconsistent noise grain detected around amount block."},
        ],
    }

    risk = risk_assessment.compute_document_risk(
        verification_results=verif,
        duplicate_results={"hasDuplicate": False, "duplicateMatches": []},
        cv_results=high_cv_results,
        ml_results={"probability": 0.60, "verdict": "suspicious"},
    )

    assert risk["score"] > 25.0
    assert any("Elevated computer-vision" in r or "Visual indicator alert" in r for r in risk["reasons"])

