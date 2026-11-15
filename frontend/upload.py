"""Batch & Single Document Upload & Extraction Page for FinVision."""

from decimal import Decimal
import io
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from PIL import Image

from frontend.components import render_risk_badge, render_rule_card
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


def run_pipeline_on_bytes(
    filename: str,
    file_bytes: bytes,
    user: Dict[str, Any],
    existing_docs: List[Dict[str, Any]],
    edited_fields: Dict[str, Any] = None,
    edited_line_items: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end FinVision pipeline on raw invoice file bytes with optional analyst field overrides."""
    mime_type = image_processing.detect_mime_type(file_bytes)
    if mime_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise ValueError(f"Unsupported file format: {mime_type}. Please upload JPEG, PNG, or PDF.")

    # 1. Cryptographic Hashes
    sha256_hash = duplicate_detection.compute_sha256(file_bytes)
    phash_data = duplicate_detection.compute_perceptual_hashes(file_bytes)

    # 2. Preprocessing
    deskewed_bgr, binary_img, proc_log, img_meta = image_processing.preprocess_image(file_bytes)

    # 3. Tesseract OCR & Layout-Aware Field Extraction
    words, raw_text, overall_conf = ocr.run_ocr(binary_img)
    extracted = ocr.extract_invoice_fields(words, raw_text, overall_conf)
    extracted_dict = extracted.to_dict()

    fields = edited_fields if edited_fields is not None else extracted_dict["fields"]
    line_items = edited_line_items if edited_line_items is not None else extracted_dict["lineItems"]
    confidences = extracted_dict["fieldConfidence"]

    # 4. Deterministic Financial Validation
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
        "lineItems": line_items,
        "risk": risk,
        "verif": verif_results,
        "dup": dup_results,
        "cv": cv_results,
        "ml": ml_pred,
        "overallConfidence": overall_conf,
        "fieldConfidence": confidences,
        "storagePath": storage_path,
        "sha256": sha256_hash,
        "pHash": phash_data["pHash"],
        "raw_text": raw_text,
    }


def render_upload(user: Dict[str, Any]):
    """Render intelligent invoice upload, field preview, analyst correction, and verification."""
    st.markdown("## 📤 Intelligent Document Upload & Verification")
    st.caption("Upload financial invoices for multi-stage OCR extraction, financial arithmetic validation, duplicate collision detection, and AI risk scoring.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    existing_docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    uploaded_files = st.file_uploader(
        "Upload Invoice Document (PDF, PNG, JPEG)",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Select one or more financial documents up to 10MB each.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected for processing.**")

        # If single file, show interactive analyst verification workspace with editable preview
        if len(uploaded_files) == 1:
            file_obj = uploaded_files[0]
            file_bytes = file_obj.getvalue()
            filename = file_obj.name

            col_preview, col_audit = st.columns([1, 1.2])

            with col_preview:
                st.markdown("### Document Preview")
                if file_obj.type == "application/pdf":
                    st.info(f"📄 PDF Document: **{filename}** ({len(file_bytes) / 1024:.1f} KB)")
                else:
                    try:
                        pil_img = Image.open(io.BytesIO(file_bytes))
                        st.image(pil_img, caption=filename, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render image preview: {e}")

            with col_audit:
                st.markdown("### Extracted Fields & Analyst Review")
                st.caption("Review extracted financial data. You may correct any OCR misreads before final persistence.")

                # Run extraction pass
                with st.spinner("Extracting invoice fields via Tesseract OCR..."):
                    mime_type = image_processing.detect_mime_type(file_bytes)
                    _, binary_img, _, _ = image_processing.preprocess_image(file_bytes)
                    words, raw_text, overall_conf = ocr.run_ocr(binary_img)
                    extracted = ocr.extract_invoice_fields(words, raw_text, overall_conf)
                    ext_data = extracted.to_dict()
                    f = ext_data["fields"]

                with st.form("analyst_correction_form"):
                    c_f1, c_f2 = st.columns(2)
                    with c_f1:
                        f_inv = st.text_input("Invoice Number", value=f.get("invoiceNumber") or "INV-")
                        f_vend = st.text_input("Vendor Name", value=f.get("vendor") or "")
                        f_date = st.text_input("Invoice Date (YYYY-MM-DD)", value=f.get("date") or "")
                        f_due = st.text_input("Due Date (YYYY-MM-DD)", value=f.get("dueDate") or "")
                    with c_f2:
                        f_sub = st.number_input("Subtotal ($)", value=float(f.get("subtotal") or 0.0), step=10.0, format="%.2f")
                        f_tax = st.number_input("Tax / VAT ($)", value=float(f.get("tax") or 0.0), step=5.0, format="%.2f")
                        f_disc = st.number_input("Discount ($)", value=float(f.get("discount") or 0.0), step=5.0, format="%.2f")
                        f_tot = st.number_input("Total Amount ($)", value=float(f.get("total") or 0.0), step=10.0, format="%.2f")

                    submit_verify = st.form_submit_button("🚀 Run Full Multi-Stage Verification & Save", type="primary", use_container_width=True)

                if submit_verify:
                    with st.spinner("Executing financial validation, duplicate scan, and ML risk evaluation..."):
                        corrected_fields = {
                            "invoiceNumber": f_inv,
                            "vendor": f_vend,
                            "date": f_date,
                            "dueDate": f_due,
                            "subtotal": f_sub,
                            "tax": f_tax,
                            "discount": f_disc,
                            "total": f_tot,
                            "currency": f.get("currency", "USD"),
                            "customer": f.get("customer", ""),
                        }
                        result = run_pipeline_on_bytes(
                            filename=filename,
                            file_bytes=file_bytes,
                            user=user,
                            existing_docs=existing_docs,
                            edited_fields=corrected_fields,
                            edited_line_items=ext_data.get("lineItems", []),
                        )
                        st.session_state["selected_doc_id"] = result["docId"]
                        st.success(f"✓ Verification complete! Document recorded as `{result['docId']}`. Risk: {result['risk']['level']} ({result['risk']['score']}/100)")
                        
                        # Display immediate rule summary
                        st.markdown("#### Verification Outcomes")
                        for r in result["verif"].get("ruleResults", []):
                            render_rule_card(r)

                        st.markdown(f"**Recommended Action:** {result['risk']['recommendedAction']}")

        else:
            # Batch upload processing
            if st.button(f"⚡ Ingest & Process All {len(uploaded_files)} Documents in Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, file_obj in enumerate(uploaded_files):
                    status_text.text(f"Processing ({i+1}/{len(uploaded_files)}): {file_obj.name}...")
                    try:
                        res = run_pipeline_on_bytes(
                            filename=file_obj.name,
                            file_bytes=file_obj.getvalue(),
                            user=user,
                            existing_docs=existing_docs + results,
                        )
                        results.append(res)
                    except Exception as e:
                        st.error(f"Error processing {file_obj.name}: {e}")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("✓ Batch processing complete!")
                st.success(f"Successfully processed and verified {len(results)} invoice(s)!")

                summary_table = []
                for r in results:
                    summary_table.append({
                        "Doc ID": r["docId"][:12],
                        "Filename": r["filename"],
                        "Invoice #": r["fields"].get("invoiceNumber", "N/A"),
                        "Vendor": r["fields"].get("vendor", "N/A"),
                        "Total": f"${float(r['fields'].get('total', 0.0)):,.2f}" if r['fields'].get('total') else "N/A",
                        "Risk Level": r["risk"]["level"],
                        "Risk Score": f"{r['risk']['score']:.0f}/100",
                    })
                st.dataframe(pd.DataFrame(summary_table), use_container_width=True)
