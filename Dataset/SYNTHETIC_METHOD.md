# Synthetic Data Generation Methodology for FinVision

## 1. Ethical & Scientific Disclaimer
**Hard Constraint**: The raw invoice dataset contains **zero verified fraud labels**. The original invoices are normal business documents with ground-truth field transcriptions. We do **not** assign arbitrary or fabricated fraud labels to real invoices.

To train and evaluate an experimental machine-learning classifier, we treat clean, mathematically sound original invoices as the **Normal class (Label 0)** and generate reproducible, documented **Suspicious examples (Label 1)** using controlled synthetic perturbations.

> [!WARNING]
> High performance on synthetic test sets **does not predict** real-world fraud detection performance. Real fraud exhibits adversarial tactics, subtle collusion, and non-random evasion that synthetic distributions cannot fully capture. All outputs are advisory risk indicators requiring human verification.

---

## 2. Perturbation Taxonomy

Each synthetic suspicious instance is derived from a parent source invoice and assigned a deterministic perturbation category:

| Category Code | Perturbation Name | Description | Affected Features |
|---|---|---|---|
| `PERT_TOTAL_MISMATCH` | Total Inflation | Inflates the stated total by +15% to +60%, intentionally breaking the invariant $\text{subtotal} + \text{tax} == \text{total}$. | `total_mismatch` |
| `PERT_TAX_DISCREPANCY` | Tax Alteration | Modifies the stated tax amount (or zeroes it out), violating $\text{subtotal} \times \text{taxRate} == \text{tax}$. | `tax_mismatch` |
| `PERT_SUBTOTAL_MISMATCH` | Subtotal Distortion | Modifies the invoice subtotal while leaving line item totals unchanged ($\sum \text{lineTotal} \neq \text{subtotal}$). | `subtotal_mismatch` |
| `PERT_MISSING_FIELDS` | Omission / Occlusion | Strips one or more required fields (e.g. missing vendor, blank date, or missing total). | `missing_fields` |
| `PERT_DUPLICATE_COLLISION` | Duplicate Collision | Flags an identical or near-identical invoice matching an existing transaction hash or vendor/amount pair. | `duplicate_flag` |
| `PERT_IMAGE_ANOMALY` | Visual Splicing | Simulates localized visual tampering (elevated ELA residuals and noise variance). | `image_anomaly_score` |

---

## 3. Data Splitting & Leakage Prevention (`GroupShuffleSplit`)

To prevent optimistic data leakage:
- Every synthetic sample retains its parent `source_invoice_id`.
- The dataset is split using scikit-learn's **`GroupShuffleSplit`** grouped by `source_invoice_id` (e.g. 80% train, 20% test).
- This strictly ensures that no original invoice and any of its perturbed variants ever straddle the train and test partitions simultaneously.

---

## 4. Baseline Comparison & Evaluation Metrics

The trained model (Gradient Boosting / Random Forest) is evaluated on the held-out test set against a **deterministic rule-based baseline** (which flags an invoice if any mathematical rule fails or mandatory fields are missing).

Reported metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix ($TP, FP, TN, FN$)

