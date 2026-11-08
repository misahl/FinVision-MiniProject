"""Unit tests for FinVision Transparent Risk Assessment Engine."""

import pytest
from src.risk_assessment import compute_document_risk


def test_clean_document_low_risk():
    """Verify clean document with passing rules receives LOW risk."""
    verification = {
        "ruleResults": [
            {"rule_id": "RULE_TOTAL_SUM", "passed": True, "severity": "INFO"},
            {"rule_id": "RULE_DATE_SANITY", "passed": True, "severity": "INFO"},
        ],
        "missingFields": [],
    }
    risk = compute_document_risk(verification)

    assert risk["level"] == "LOW"
    assert risk["score"] < 25.0
    assert "Routine verification" in risk["recommendedAction"]


def test_rules_precedence_critical_failure():
    """Verify rule precedence: critical failure forces HIGH risk even if ML probability is 0."""
    verification = {
        "ruleResults": [
            {
                "rule_id": "RULE_TOTAL_SUM",
                "passed": False,
                "severity": "CRITICAL",
                "message": "Total mismatch: stated 500, expected 100",
            }
        ],
        "missingFields": [],
    }
    # ML claims document is 0% suspicious
    ml_results = {"prediction": 0, "probability": 0.0, "verdict": "normal"}

    risk = compute_document_risk(verification, ml_results=ml_results)

    assert risk["level"] == "HIGH"
    assert risk["score"] >= 60.0
    assert any("Critical rule variance" in r for r in risk["reasons"])
    assert "Immediate manual review required" in risk["recommendedAction"]


def test_duplicate_collision_raises_risk():
    """Verify duplicate match adds appropriate risk weight."""
    verification = {"ruleResults": [], "missingFields": []}
    duplicate = {
        "hasDuplicate": True,
        "duplicateMatches": [
            {"docId": "doc99", "matchType": "EXACT_SHA256", "reason": "Identical SHA-256 hash"}
        ],
    }

    risk = compute_document_risk(verification, duplicate_results=duplicate)

    assert risk["score"] >= 40.0
    assert risk["level"] in ["MEDIUM", "HIGH"]
    assert any("Duplicate collision" in r for r in risk["reasons"])


def test_vocabulary_never_declares_fraud():
    """Strict test: risk assessment wording must NEVER use the word fraud or fraudulent."""
    verification = {
        "ruleResults": [
            {"rule_id": "RULE_TOTAL_SUM", "passed": False, "severity": "CRITICAL", "message": "Discrepancy detected"},
        ],
        "missingFields": ["vendor", "total"],
    }
    ml_results = {"prediction": 1, "probability": 0.99, "verdict": "suspicious"}
    duplicate = {"hasDuplicate": True, "duplicateMatches": [{"matchType": "EXACT_SHA256"}]}

    risk = compute_document_risk(verification, duplicate_results=duplicate, ml_results=ml_results)

    combined_text = " ".join([risk["level"], risk["recommendedAction"]] + risk["reasons"]).lower()

    assert "fraud" not in combined_text, "FORBIDDEN WORD: 'fraud' found in risk output!"
    assert "fraudulent" not in combined_text, "FORBIDDEN WORD: 'fraudulent' found in risk output!"
    assert "review" in combined_text, "Expected 'review' requirement in risk recommendations!"

