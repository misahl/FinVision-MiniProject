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
    """Load model from local cache path."""
    global _CACHED_MODEL, _CACHED_VERSION
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not target_path.exists():
        # Fallback if model has not been trained yet: run training pipeline
        from src.train_model import train_and_evaluate
        train_and_evaluate()

    if _CACHED_MODEL is None:
        _CACHED_MODEL = joblib.load(target_path)
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
    X = to_feature_array(features_dict)

    pred = int(model.predict(X)[0])
    prob_arr = model.predict_proba(X)[0]
    prob_suspicious = float(prob_arr[1]) if len(prob_arr) > 1 else float(pred)

    return {
        "modelVersion": _CACHED_VERSION,
        "prediction": pred,
        "probability": round(prob_suspicious, 4),
        "experimental": True,
        "verdict": "suspicious" if pred == 1 else "normal",
        "featuresUsed": {name: features_dict.get(name, 0.0) for name in FEATURE_NAMES},
    }

