"""Training Pipeline for FinVision ML Suspiciousness Classifier.

Honest Machine Learning Protocol:
1. Loads ground-truth invoice metadata from dataset CSV.
2. Treats verified original invoices as Normal (Label 0).
3. Generates reproducible, documented synthetic perturbations as Suspicious (Label 1).
4. Prevents data leakage by splitting with GroupShuffleSplit on source_invoice_id.
5. Computes Accuracy, Precision, Recall, F1, and Confusion Matrix on held-out test set.
6. Evaluates against a deterministic rule-based baseline on the identical test split.
7. Serializes model to models/model_v1.joblib and saves modelRegistry metadata.

Zero Firebase dependencies: purely functional and testable offline.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src.feature_extraction import FEATURE_NAMES, extract_features_from_document_state
from src.verification import run_verification_rules

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATASET_CSV = ROOT_DIR / "Dataset" / "high-quality-invoice-images" / "batch_1" / "batch_1" / "batch1_1.csv"
MANIFEST_CSV = ROOT_DIR / "dataset" / "manifest.csv"
MODEL_PATH = ROOT_DIR / "models" / "model_v1.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata_v1.json"


def generate_synthetic_dataset(
    csv_path: Path,
    limit: int = 300,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate balanced normal and synthetically perturbed dataset from ground-truth invoices."""
    np.random.seed(random_seed)
    df = pd.read_csv(csv_path)
    sample_df = df.head(limit)

    records: List[Dict[str, Any]] = []

    for idx, row in sample_df.iterrows():
        source_id = f"inv_{idx:04d}"
        try:
            data = json.loads(row["Json Data"])
        except Exception:
            continue

        inv_info = data.get("invoice", {})
        sub_info = data.get("subtotal", {})
        items = data.get("items", [])

        # Base fields
        invoice_num = inv_info.get("invoice_number", f"INV-{idx}")
        date_str = inv_info.get("invoice_date", "2023-01-15")
        vendor_name = inv_info.get("seller_name", "Vendor Corp")
        
        # Numbers
        try:
            tot_val = float(str(sub_info.get("total", "100.00")).replace(",", "."))
            tax_val = float(str(sub_info.get("tax", "10.00")).replace(",", ".")) if sub_info.get("tax") else 0.0
            sub_val = tot_val - tax_val
        except Exception:
            tot_val, tax_val, sub_val = 110.0, 10.0, 100.0

        rate_val = round(tax_val / max(1.0, sub_val), 2) if sub_val > 0 else 0.10

        # Normal line items
        line_items_clean = []
        for it in items:
            try:
                q = float(str(it.get("quantity", "1.00")).replace(",", "."))
                tot = float(str(it.get("total_price", "20.00")).replace(",", "."))
                p = tot / max(1.0, q)
                line_items_clean.append({
                    "description": it.get("description", "Item"),
                    "quantity": q,
                    "unitPrice": p,
                    "lineTotal": tot,
                })
            except Exception:
                continue

        # -------------------------------------------------------------
        # 1. Normal Instance (Label 0)
        # -------------------------------------------------------------
        normal_fields = {
            "invoiceNumber": invoice_num,
            "date": date_str,
            "vendor": vendor_name,
            "subtotal": sub_val,
            "tax": tax_val,
            "taxRate": rate_val,
            "total": tot_val,
        }
        verif_norm = run_verification_rules(normal_fields, line_items_clean)
        feat_norm = extract_features_from_document_state(
            fields=normal_fields,
            verification_results=verif_norm,
            ocr_confidence=float(np.random.uniform(92.0, 98.0)),
            image_anomaly_score=float(np.random.uniform(0.0, 10.0)),
            duplicate_flag=0,
        )
        records.append({
            "sample_id": f"{source_id}_normal",
            "source_invoice_id": source_id,
            "label": 0,
            "perturbation_type": "NORMAL",
            **feat_norm,
        })

        # -------------------------------------------------------------
        # 2. Perturbed Instance (Label 1)
        # Choose randomly among documented perturbation types
        # -------------------------------------------------------------
        p_type = np.random.choice([
            "PERT_TOTAL_MISMATCH",
            "PERT_TAX_DISCREPANCY",
            "PERT_SUBTOTAL_MISMATCH",
            "PERT_MISSING_FIELDS",
            "PERT_DUPLICATE_COLLISION",
            "PERT_IMAGE_ANOMALY",
        ])

        pert_fields = dict(normal_fields)
        pert_items = [dict(it) for it in line_items_clean]
        dup_flag = 0
        img_score = float(np.random.uniform(0.0, 10.0))
        ocr_conf = float(np.random.uniform(88.0, 96.0))

        if p_type == "PERT_TOTAL_MISMATCH":
            # Inflate total by +20% to +50%
            inflation = np.random.uniform(1.2, 1.5)
            pert_fields["total"] = round(tot_val * inflation, 2)
        elif p_type == "PERT_TAX_DISCREPANCY":
            # Zero out or double tax
            pert_fields["tax"] = round(tax_val * 2.5 + 15.0, 2)
        elif p_type == "PERT_SUBTOTAL_MISMATCH":
            # Modify subtotal
            pert_fields["subtotal"] = round(sub_val + 75.0, 2)
        elif p_type == "PERT_MISSING_FIELDS":
            # Strip required field
            drop_target = np.random.choice(["total", "vendor", "date"])
            pert_fields[drop_target] = None
        elif p_type == "PERT_DUPLICATE_COLLISION":
            dup_flag = 1
        elif p_type == "PERT_IMAGE_ANOMALY":
            img_score = float(np.random.uniform(55.0, 85.0))
            ocr_conf = float(np.random.uniform(60.0, 75.0))

        verif_pert = run_verification_rules(pert_fields, pert_items)
        feat_pert = extract_features_from_document_state(
            fields=pert_fields,
            verification_results=verif_pert,
            ocr_confidence=ocr_conf,
            image_anomaly_score=img_score,
            duplicate_flag=dup_flag,
        )
        records.append({
            "sample_id": f"{source_id}_pert_{p_type}",
            "source_invoice_id": source_id,
            "label": 1,
            "perturbation_type": p_type,
            **feat_pert,
        })

    dataset = pd.DataFrame(records)
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(MANIFEST_CSV, index=False)
    return dataset


def evaluate_baseline_rules(X_df: pd.DataFrame) -> np.ndarray:
    """Evaluate deterministic rule-based baseline: flags 1 if math mismatch or missing fields."""
    preds = []
    for _, row in X_df.iterrows():
        math_fail = (
            row.get("total_mismatch", 0) > 0.05
            or row.get("subtotal_mismatch", 0) > 0.05
            or row.get("tax_mismatch", 0) > 0.05
        )
        missing_fail = row.get("missing_fields", 0) > 0
        dup_fail = row.get("duplicate_flag", 0) == 1
        preds.append(1 if (math_fail or missing_fail or dup_fail) else 0)
    return np.array(preds)


def train_and_evaluate(
    csv_path: Path = DATASET_CSV,
    limit: int = 300,
) -> Dict[str, Any]:
    """Train Random Forest classifier with GroupShuffleSplit and compare with baseline."""
    df = generate_synthetic_dataset(csv_path, limit=limit)

    X = df[FEATURE_NAMES]
    y = df["label"].values
    groups = df["source_invoice_id"].values

    # Strict GroupShuffleSplit to avoid data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Verify no source invoice ID is present in both train and test
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert len(train_groups.intersection(test_groups)) == 0, "DATA LEAKAGE: groups intersect between train and test!"

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    # ML Model Predictions on held-out test set
    y_pred_ml = clf.predict(X_test)
    ml_acc = float(accuracy_score(y_test, y_pred_ml))
    ml_prec = float(precision_score(y_test, y_pred_ml, zero_division=0))
    ml_rec = float(recall_score(y_test, y_pred_ml, zero_division=0))
    ml_f1 = float(f1_score(y_test, y_pred_ml, zero_division=0))
    ml_cm = confusion_matrix(y_test, y_pred_ml).tolist()

    # Rule-Based Baseline Predictions on the identical held-out test set
    y_pred_baseline = evaluate_baseline_rules(X_test)
    base_acc = float(accuracy_score(y_test, y_pred_baseline))
    base_prec = float(precision_score(y_test, y_pred_baseline, zero_division=0))
    base_rec = float(recall_score(y_test, y_pred_baseline, zero_division=0))
    base_f1 = float(f1_score(y_test, y_pred_baseline, zero_division=0))
    base_cm = confusion_matrix(y_test, y_pred_baseline).tolist()

    # Feature Importances
    feat_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
    }

    # Save model artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    metadata: Dict[str, Any] = {
        "version": "v1.0.0",
        "trainedAt": "2026-11-05T06:00:00Z",
        "datasetDescription": f"{len(df)} balanced normal and synthetically perturbed invoice records",
        "syntheticMethod": "See dataset/SYNTHETIC_METHOD.md for perturbation specifications",
        "trainTestSplit": "GroupShuffleSplit (grouped by source_invoice_id, 75% train / 25% test, 0 leakage)",
        "trainSamples": len(X_train),
        "testSamples": len(X_test),
        "features": FEATURE_NAMES,
        "featureImportances": feat_importances,
        "metrics": {
            "mlModel": {
                "accuracy": round(ml_acc, 4),
                "precision": round(ml_prec, 4),
                "recall": round(ml_rec, 4),
                "f1Score": round(ml_f1, 4),
                "confusionMatrix": ml_cm,
            },
            "ruleBasedBaseline": {
                "accuracy": round(base_acc, 4),
                "precision": round(base_prec, 4),
                "recall": round(base_rec, 4),
                "f1Score": round(base_f1, 4),
                "confusionMatrix": base_cm,
            },
        },
        "disclaimer": (
            "Model trained strictly on reproducible synthetic perturbations. "
            "Synthetic test performance does not guarantee real-world fraud detection efficacy."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    meta = train_and_evaluate()
    print("MODEL TRAINING & EVALUATION COMPLETED")
    print(json.dumps(meta["metrics"], indent=2))

