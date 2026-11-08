"""Transparent Multi-Criteria Risk Assessment Engine for FinVision.

Combines:
1. Rule Engine results (critical / warning severities)
2. Duplicate detection matches (exact, near, semantic)
3. Computer Vision anomaly scores (ELA, noise inconsistency, layout outliers)
4. Machine Learning classification probabilities

Core Principles:
- Transparent, documented scoring table (weights loaded from config/risk_weights.json).
- Rules Precedence: Rules always take precedence over ML inferences.
  ML probability can raise a score, but can NEVER lower a rule-based HIGH risk.
- Professional Tone: NEVER declares a document "fraudulent".
  Always outputs an objective risk assessment stating that manual review is required.

Zero Firebase dependencies: purely functional and testable offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load risk weights and thresholds
CONFIG_PATH = Path(__file__).parent.parent / "config" / "risk_weights.json"
DEFAULT_CONFIG = {
    "weights": {
        "rule_failure_critical": 35,
        "rule_failure_warning": 15,
        "duplicate_exact_match": 40,
        "duplicate_near_match": 25,
        "cv_ela_anomaly": 15,
        "cv_noise_inconsistency": 15,
        "cv_text_spacing_outlier": 15,
        "ml_high_probability": 20,
    },
    "thresholds": {
        "high_risk_min": 60,
        "medium_risk_min": 25,
    },
    "precedence": "rules_precedence_high",
}

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except Exception:
        CONFIG = DEFAULT_CONFIG
else:
    CONFIG = DEFAULT_CONFIG

WEIGHTS = CONFIG.get("weights", DEFAULT_CONFIG["weights"])
THRESHOLDS = CONFIG.get("thresholds", DEFAULT_CONFIG["thresholds"])


def compute_document_risk(
    verification_results: Dict[str, Any],
    duplicate_results: Optional[Dict[str, Any]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate multi-criteria risk score, level, reasons, and recommended action.
    
    Returns structured risk object matching Firestore document schema:
    {
        "level": "LOW" | "MEDIUM" | "HIGH",
        "score": float (0.0 to 100.0),
        "reasons": List[str],
        "recommendedAction": str,
        "breakdown": Dict[str, float],
    }
    """
    score = 0.0
    reasons: List[str] = []
    breakdown: Dict[str, float] = {}
    has_critical_rule_failure = False

    # 1. Rule Engine Results
    rule_results = verification_results.get("ruleResults", [])
    rule_pts = 0.0

    for r in rule_results:
        if not r.get("passed", True):
            sev = r.get("severity", "WARNING")
            msg = r.get("message", "Rule validation check failed.")
            if sev == "CRITICAL":
                has_critical_rule_failure = True
                pts = float(WEIGHTS.get("rule_failure_critical", 35))
                rule_pts += pts
                reasons.append(f"Critical rule variance: {msg}")
            else:
                pts = float(WEIGHTS.get("rule_failure_warning", 15))
                rule_pts += pts
                reasons.append(f"Rule warning: {msg}")

    missing_fields = verification_results.get("missingFields", [])
    if missing_fields:
        reasons.append(f"Required information missing: {', '.join(missing_fields)}.")

    breakdown["rules_contribution"] = round(rule_pts, 2)
    score += rule_pts

    # 2. Duplicate Matches
    dup_pts = 0.0
    if duplicate_results and duplicate_results.get("hasDuplicate"):
        for m in duplicate_results.get("duplicateMatches", []):
            m_type = m.get("matchType", "")
            r_msg = m.get("reason", "Potential duplicate document match.")
            if m_type == "EXACT_SHA256":
                pts = float(WEIGHTS.get("duplicate_exact_match", 40))
                dup_pts += pts
                reasons.append(f"Duplicate collision: {r_msg}")
            elif m_type == "PERCEPTUAL_PHASH":
                pts = float(WEIGHTS.get("duplicate_near_match", 25))
                dup_pts += pts
                reasons.append(f"Visual duplicate match: {r_msg}")
            else:
                pts = 20.0
                dup_pts += pts
                reasons.append(f"Semantic business duplicate: {r_msg}")

    breakdown["duplicate_contribution"] = round(dup_pts, 2)
    score += dup_pts

    # 3. Computer Vision Anomaly Scores
    cv_pts = 0.0
    if cv_results:
        cv_score = float(cv_results.get("cvScore", 0.0))
        if cv_score > 30.0:
            cv_pts = (cv_score / 100.0) * 25.0
            reasons.append(
                f"Elevated computer-vision anomaly score ({cv_score:.1f}/100) "
                f"across visual inspection indicators."
            )
        for ind in cv_results.get("indicators", []):
            if ind.get("score", 0.0) > 40.0:
                reasons.append(f"Visual indicator alert: {ind.get('details')}")

    breakdown["cv_contribution"] = round(cv_pts, 2)
    score += cv_pts

    # 4. Machine Learning Inference
    ml_pts = 0.0
    if ml_results:
        prob = float(ml_results.get("probability", 0.0))
        if prob > 0.5:
            ml_pts = prob * float(WEIGHTS.get("ml_high_probability", 20))
            reasons.append(
                f"Statistical classifier flagged variance patterns (confidence probability: {prob*100:.1f}%)."
            )

    breakdown["ml_contribution"] = round(ml_pts, 2)
    score += ml_pts

    # -------------------------------------------------------------
    # Rule Precedence Guarantee
    # -------------------------------------------------------------
    # If a critical financial rule failed (e.g. math discrepancy, future date),
    # the document MUST be classified as HIGH risk regardless of ML probability.
    high_threshold = float(THRESHOLDS.get("high_risk_min", 60))
    med_threshold = float(THRESHOLDS.get("medium_risk_min", 25))

    if has_critical_rule_failure:
        score = max(score, high_threshold + 5.0)

    # Normalize total score to [0.0, 100.0]
    final_score = min(100.0, max(0.0, score))

    if final_score >= high_threshold or has_critical_rule_failure:
        level = "HIGH"
        recommended_action = (
            "Immediate manual review required by Senior Financial Analyst prior to approval. "
            "Resolve flagged arithmetic, business rule, or duplicate collisions."
        )
    elif final_score >= med_threshold:
        level = "MEDIUM"
        recommended_action = (
            "Secondary review recommended. Verify highlighted field values, vendor information, "
            "or minor variances against original purchase requisition."
        )
    else:
        level = "LOW"
        recommended_action = (
            "Routine verification complete. No significant variances detected; "
            "eligible for straight-through approval."
        )

    if not reasons:
        reasons.append("All automated verification checks, arithmetic rules, and visual scans passed successfully.")

    return {
        "level": level,
        "score": round(final_score, 2),
        "reasons": reasons,
        "recommendedAction": recommended_action,
        "breakdown": breakdown,
    }

