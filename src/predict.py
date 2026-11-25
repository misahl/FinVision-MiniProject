"""Inference Module for FinVision ML Classifier.

Loads trained model and generates predictions with confidence probabilities.
All outputs display experimental flag until further validated.

Zero Firebase dependencies: purely functional and testable offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np

from src.feature_extraction import FEATURE_NAMES, to_feature_array

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model_v1.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata_v1.json"

_CACHED_MODEL = None
_CACHED_VERSION = "v1.0.0"


def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Load model from local cache path with resilient fallback."""
    global _CACHED_MODEL, _CACHED_VERSION
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not target_path.exists():
        try:
            from src.train_model import train_and_evaluate
            train_and_evaluate()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Model training fallback failed: %s. Using heuristic ML fallback.", e)

    if _CACHED_MODEL is None and target_path.exists():
        try:
            _CACHED_MODEL = joblib.load(target_path)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to load model from %s: %s", target_path, e)
            _CACHED_MODEL = None

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                _CACHED_VERSION = meta.get("version", "v1.0.0")
        except Exception:
            pass

    return _CACHED_MODEL


def predict_document_risk(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """Generate ML inference prediction and probability for a document feature vector.
    
    Returns:
    {
        "modelVersion": str,
        "prediction": int (0: normal, 1: suspicious),
        "probability": float (0.0 to 1.0 probability of suspiciousness),
        "experimental": bool (always True),
        "verdict": "normal" | "suspicious",
    }
    """
    model = load_model()
    
    if model is not None:
        try:
            X = to_feature_array(features_dict)
            pred = int(model.predict(X)[0])
            prob_arr = model.predict_proba(X)[0]
            prob_suspicious = float(prob_arr[1]) if len(prob_arr) > 1 else float(pred)
        except Exception:
            model = None

    if model is None:
        # Resilient heuristic baseline when model weights are not loaded
        rules_failed = float(features_dict.get("rules_failed", 0.0))
        dup_flag = float(features_dict.get("duplicate_flag", 0.0))
        anomaly_score = float(features_dict.get("image_anomaly_score", 0.0)) / 100.0
        ocr_conf = float(features_dict.get("ocr_confidence", 85.0))
        ocr_penalty = max(0.0, (70.0 - ocr_conf) / 100.0)
        
        prob_suspicious = min(1.0, max(0.05, (rules_failed * 0.35) + (dup_flag * 0.40) + (anomaly_score * 0.20) + ocr_penalty))
        pred = 1 if prob_suspicious >= 0.5 else 0

    return {
        "modelVersion": _CACHED_VERSION,
        "prediction": pred,
        "probability": round(prob_suspicious, 4),
        "experimental": True,
        "verdict": "suspicious" if pred == 1 else "normal",
        "featuresUsed": {name: features_dict.get(name, 0.0) for name in FEATURE_NAMES},
    }

