"""Batch & Single Document Upload & Extraction Page for FinVision."""

from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from PIL import Image

from frontend.components import render_risk_badge
from src import (
    image_processing,
    ocr,
    verification,
    duplicate_detection,
    feature_extraction,
    predict,
    risk_assessment,
    firebase_service,
)


def process_single_invoice_payload(
    filename: str,
    file_bytes: bytes,
    user: Dict[str, Any],
    existing_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute end-to-end FinVision pipeline on a raw invoice file payload."""
    mime_type = image_processing.detect_mime_type(file_bytes)
    if mime_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise ValueError(f"Unsupported file format: {mime_type}")

    # 1. Cryptographic Hashes
    sha256_hash = duplicate_detection.compute_sha256(file_bytes)
    phash_data = duplicate_detection.compute_perceptual_hashes(file_bytes)

    # 2. Preprocessing Pipeline
    deskewed_bgr, binary_img, proc_log, img_meta = image_processing.preprocess_image(file_bytes)

    # 3. Tesseract OCR & Layout-Aware Field Extraction
    words, raw_text, overall_conf = ocr.run_ocr(binary_img)
    extracted = ocr.extract_invoice_fields(words, raw_text, overall_conf)

    extracted_dict = extracted.to_dict()
    fields = extracted_dict["fields"]
    line_items = extracted_dict["lineItems"]
    confidences = extracted_dict["fieldConfidence"]

    # 4. Deterministic Rule Engine Verification
    verif_results = verification.run_verification_rules(fields, line_items)

    # 5. Computer Vision Anomaly Indicators
    is_jpeg = mime_type == "image/jpeg"
    cv_results = image_processing.run_cv_anomaly_pipeline(deskewed_bgr, ocr_boxes=words, is_jpeg=is_jpeg)

    # 6. Multi-Modal Duplicate Detection
    target_candidate = {
        "sha256": sha256_hash,
        "pHash": phash_data["pHash"],
        "fields": fields,
    }
    dup_results = duplicate_detection.find_duplicates(target_candidate, existing_docs)
    verif_results["duplicateMatches"] = dup_results.get("duplicateMatches", [])

    # 7. Machine Learning Suspiciousness Classification
    feats = feature_extraction.extract_features_from_document_state(
        fields=fields,
        verification_results=verif_results,
        ocr_confidence=overall_conf,
        image_anomaly_score=cv_results.get("cvScore", 0.0),
        duplicate_flag=1 if dup_results.get("hasDuplicate") else 0,
    )
    ml_pred = predict.predict_document_risk(feats)

    # 8. Multi-Criteria Risk Assessment
    risk = risk_assessment.compute_document_risk(
        verification_results=verif_results,
        duplicate_results=dup_results,
        cv_results=cv_results,
        ml_results=ml_pred,
    )

    # 9. Storage & Firestore Persistence
    owner_uid = user.get("uid", "anonymous")
    temp_doc_id = f"doc_{sha256_hash[:16]}"
    storage_path = firebase_service.upload_invoice_bytes(
        owner_uid, temp_doc_id, filename, file_bytes, mime_type
    )

    file_meta = {
        "filename": filename,
        "storagePath": storage_path,
        "mimeType": mime_type,
        "sizeBytes": len(file_bytes),
        "sha256": sha256_hash,
        "pHash": phash_data["pHash"],
        "pageCount": img_meta.get("page_count", 1),
    }
    new_doc_id = firebase_service.create_document_record(owner_uid, file_meta)

    firebase_service.save_document_analysis(
        doc_id=new_doc_id,
        fields=fields,
        field_confidence=confidences,
        line_items=line_items,
        verification_results=verif_results,
        cv_results=cv_results,
        ml_results=ml_pred,
        risk_assessment=risk,
        storage_path=storage_path,
    )

    return {
        "docId": new_doc_id,
        "filename": filename,
        "fields": fields,
        "risk": risk,
        "verif": verif_results,
        "dup": dup_results,
        "overallConfidence": overall_conf,
        "storagePath": storage_path,
        "sha256": sha256_hash,
        "pHash": phash_data["pHash"],
    }


def render_upload(user: dict):
    """Render batch & single invoice upload and pipeline workspace."""
    st.markdown("## Upload & Process Invoice Documents")
    st.caption("Upload multiple documents simultaneously or process batches. Supported: **JPEG, PNG, PDF** (up to 10 MB per file).")

    # Mode Selector Tabs
    tab_files, tab_dataset = st.tabs(["📁 Upload Files (Batch or Single)", "⚡ Quick Demo Batch (From Dataset)"])

    with tab_files:
        uploaded_files = st.file_uploader(
            "Drop invoice documents here (select one or many files)",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            help="Select one or dozens of invoices to process through FinVision's verification pipeline.",
        )

        if uploaded_files:
            total_count = len(uploaded_files)
            total_bytes = sum(f.size for f in uploaded_files)
            total_mb = total_bytes / (1024 * 1024)

            st.markdown(
                f"""
                <div class="fin-card" style="padding: 14px 20px; margin-top: 10px; margin-bottom: 16px;">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #64748B;">BATCH SUMMARY</div>
                    <div style="font-weight: 700; font-size: 1.25rem; color: #0F172A; margin: 2px 0;">
                        {total_count} document(s) selected ({total_mb:.2f} MB total)
                    </div>
                    <div style="font-size: 0.85rem; color: #475569;">Ready for automated ingestion, OCR extraction, and multi-criteria audit verification.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Single file visual preview
            if total_count == 1:
                single_file = uploaded_files[0]
                single_bytes = single_file.read()
                single_file.seek(0)
                try:
                    bgr_img, meta = image_processing.load_image_from_source(single_bytes)
                    st.image(bgr_img, caption=f"{single_file.name} ({meta['width']}x{meta['height']}px)", use_container_width=True)
                except Exception:
                    pass

            # Action Button
            btn_label = f"🚀 Run Pipeline on {total_count} Document(s)" if total_count > 1 else "🚀 Process Document"
            if st.button(btn_label, type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                processed_results = []
                existing_docs = firebase_service.list_documents(limit=100)

                for idx, file_obj in enumerate(uploaded_files):
                    status_text.markdown(f"**Processing {idx+1}/{total_count}:** `{file_obj.name}`...")
                    file_bytes = file_obj.read()

                    try:
                        res = process_single_invoice_payload(file_obj.name, file_bytes, user, existing_docs)
                        processed_results.append(res)
                        # Add to existing_docs so duplicate detector catches collisions within the same batch!
                        existing_docs.append({
                            "docId": res["docId"],
                            "sha256": res["sha256"],
                            "pHash": res["pHash"],
                            "fields": res["fields"],
                        })
                    except Exception as err:
                        st.error(f"Error processing {file_obj.name}: {err}")

                    progress_bar.progress((idx + 1) / total_count)

                status_text.success(f"✓ Successfully processed and verified {len(processed_results)} invoice(s)!")

                # Batch Results Summary Cards
                if processed_results:
                    c1, c2, c3, c4 = st.columns(4)
                    low_cnt = sum(1 for r in processed_results if r["risk"]["level"] == "LOW")
                    med_cnt = sum(1 for r in processed_results if r["risk"]["level"] == "MEDIUM")
                    high_cnt = sum(1 for r in processed_results if r["risk"]["level"] == "HIGH")
                    dup_cnt = sum(1 for r in processed_results if r["dup"].get("hasDuplicate"))

                    with c1:
                        st.metric("Total Processed", len(processed_results))
                    with c2:
                        st.metric("Standard (Low Risk)", low_cnt)
                    with c3:
                        st.metric("Medium Variance", med_cnt)
                    with c4:
                        st.metric("High Risk / Flagged", high_cnt)

                    st.markdown("### Processed Document Summary")
                    summary_rows = []
                    for r in processed_results:
                        f = r["fields"]
                        summary_rows.append({
                            "File Name": r["filename"],
                            "Doc ID": r["docId"][:12],
                            "Invoice #": f.get("invoiceNumber") or "N/A",
                            "Vendor": f.get("vendor") or "N/A",
                            "Total": f"${float(f.get('total', 0.0)):,.2f}" if f.get("total") else "N/A",
                            "Risk Rating": r["risk"]["level"],
                            "Score": f"{r['risk']['score']:.0f}/100",
                            "Duplicate?": "Yes" if r["dup"].get("hasDuplicate") else "No",
                        })
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    st.session_state["selected_doc_id"] = processed_results[0]["docId"]
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("🔍 Open First Document in Verify Workspace", type="primary", use_container_width=True):
                            st.session_state["nav_page"] = "Verify Documents"
                            st.rerun()
                    with col_b2:
                        if st.button("📊 View Portfolio in Dashboard", use_container_width=True):
                            st.session_state["nav_page"] = "Dashboard"
                            st.rerun()

    with tab_dataset:
        st.markdown("### ⚡ Quick Demo Batch Ingestion")
        st.caption("Instantly load a batch of 5 to 20 real invoices directly from the local high-quality invoice dataset without manually selecting files.")

        dataset_dir = Path("Dataset/high-quality-invoice-images/batch_1/batch_1/batch1_1")
        if not dataset_dir.exists():
            st.warning("Dataset directory not found.")
            return

        sample_files = sorted(list(dataset_dir.glob("*.jpg")))
        st.write(f"**Available sample invoices in batch:** `{len(sample_files)} files`")

        batch_size = st.slider("Select batch size to ingest", min_value=3, max_value=20, value=8)

        if st.button(f"⚡ Ingest & Audit Batch of {batch_size} Invoices", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            selected_samples = sample_files[:batch_size]
            processed_results = []
            existing_docs = firebase_service.list_documents(limit=100)

            for idx, p in enumerate(selected_samples):
                status_text.markdown(f"**Processing {idx+1}/{batch_size}:** `{p.name}`...")
                file_bytes = p.read_bytes()

                try:
                    res = process_single_invoice_payload(p.name, file_bytes, user, existing_docs)
                    processed_results.append(res)
                    existing_docs.append({
                        "docId": res["docId"],
                        "sha256": res["sha256"],
                        "pHash": res["pHash"],
                        "fields": res["fields"],
                    })
                except Exception as err:
                    st.error(f"Error on {p.name}: {err}")

                progress_bar.progress((idx + 1) / batch_size)

            status_text.success(f"✓ Ingested and verified {len(processed_results)} dataset invoices!")

            if processed_results:
                st.session_state["selected_doc_id"] = processed_results[0]["docId"]
                st.markdown("### Batch Results Overview")
                rows = []
                for r in processed_results:
                    f = r["fields"]
                    rows.append({
                        "Doc ID": r["docId"][:12],
                        "File": r["filename"],
                        "Invoice #": f.get("invoiceNumber") or "N/A",
                        "Vendor": f.get("vendor") or "N/A",
                        "Total": f"${float(f.get('total', 0.0)):,.2f}" if f.get("total") else "N/A",
                        "Risk Rating": r["risk"]["level"],
                        "Score": f"{r['risk']['score']:.0f}/100",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if st.button("🔍 Go to Verify Workspace to Review", type="primary"):
                    st.session_state["nav_page"] = "Verify Documents"
                    st.rerun()
