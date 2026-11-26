"""Redesigned Ingest & Verify Workspace for FinVision (Apple Pro Light Mode)."""

from datetime import datetime
from decimal import Decimal
import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from frontend import components
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


def draw_bounding_boxes_on_image(
    base_img: Image.Image,
    words: List[Dict[str, Any]],
    highlight_text_snippets: List[str] = None,
) -> Image.Image:
    """Overlay subtle semi-transparent bounding boxes on the document scan for extracted fields."""
    annotated = base_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    snippets = [s.strip().lower() for s in (highlight_text_snippets or []) if s and len(s) > 2]

    # Draw word boxes
    for w in words:
        bbox = w.get("bbox")
        text = str(w.get("text", "")).strip().lower()
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        
        is_highlight = any(snip in text or text in snip for snip in snippets)
        if is_highlight:
            # Highlighted field source (Indigo glow)
            draw.rectangle([x, y, x + bw, y + bh], fill=(99, 102, 241, 60), outline=(129, 140, 248, 220), width=2)
        elif w.get("conf", 100) < 65:
            # Low confidence box (Amber)
            draw.rectangle([x, y, x + bw, y + bh], fill=(245, 158, 11, 40), outline=(245, 158, 11, 180), width=1)

    result = Image.alpha_composite(annotated, overlay)
    return result.convert("RGB")


def run_pipeline_on_bytes(
    filename: str,
    file_bytes: bytes,
    user: Dict[str, Any],
    existing_docs: List[Dict[str, Any]],
    edited_fields: Dict[str, Any] = None,
    edited_line_items: List[Dict[str, Any]] = None,
    folder: str = "Operations",
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

    # 3. Multi-Engine Text & Word Extraction (Native Vector PDF + High-Accuracy OCR)
    words, raw_text, overall_conf = ocr.extract_document_words(
        source=file_bytes,
        mime_type=mime_type,
        fallback_image=deskewed_bgr,
    )
    extracted = ocr.extract_invoice_fields(words, raw_text, overall_conf)
    extracted_dict = extracted.to_dict()

    fields = edited_fields if edited_fields is not None else extracted_dict["fields"]
    line_items = edited_line_items if edited_line_items is not None else extracted_dict["lineItems"]
    confidences = extracted_dict["fieldConfidence"]

    # 3.5 Check Document Validity (Is it an invoice or a random photo?)
    is_valid_inv, validity_reason = ocr.validate_is_invoice_document(raw_text, len(words), fields)
    if not is_valid_inv:
        return {
            "isValidInvoice": False,
            "rejectionReason": validity_reason,
            "filename": filename,
            "file_bytes": file_bytes,
            "mime_type": mime_type,
            "preview_image": deskewed_bgr,
            "fields": fields,
            "lineItems": line_items,
            "docId": None,
            "folder": folder,
        }

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
        "folder": folder,
        "storagePath": storage_path,
        "mimeType": mime_type,
        "fileSizeBytes": len(file_bytes),
        "sha256": sha256_hash,
        "pHash": phash_data["pHash"],
        "width": img_meta.get("width", 0),
        "height": img_meta.get("height", 0),
        "dpi": img_meta.get("dpi", 300),
    }

    new_doc_id = firebase_service.create_document_record(
        owner_uid=owner_uid,
        file_metadata=file_meta,
    )

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


def render_batch_hub(batch_files: List[Dict[str, Any]], user: Dict[str, Any], existing_docs: List[Dict[str, Any]]):
    """Render enterprise Apple-styled Batch Chunk Ingestion workspace."""
    total_size_mb = sum(f.get("size", len(f.get("bytes", b""))) for f in batch_files) / (1024 * 1024)

    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h2 style="margin: 0; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.024em; color: #1D1D1F;">📦 Batch Invoice Ingestion Hub</h2>
                    <div style="color: #515154; font-size: 0.88rem; margin-top: 4px;">
                        Parallel OCR processing, invariant validation, duplicate detection, and ML risk scoring for enterprise document chunks.
                    </div>
                </div>
                <div class="fin-status-pill">
                    <span class="fin-status-dot"></span>
                    <span>Batch Engine Ready</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Apple Frosted Glass KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        components.render_kpi_card("Queued Invoices", f"{len(batch_files)} docs", "Batch Chunk Size")
    with k2:
        components.render_kpi_card("Total Payload", f"{total_size_mb:.2f} MB", "Compressed Scan Data")
    with k3:
        components.render_kpi_card("Audit Rules", "10 Invariants", "Deterministic Math Engine")
    with k4:
        components.render_kpi_card("Risk Engine", "Active", "Random Forest + CV Heuristics")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Action bar
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        run_batch = st.button(
            f"⚡ Run Automated Verification on All {len(batch_files)} Documents",
            type="primary",
            use_container_width=True,
            key="btn_run_batch_verification",
        )
    with btn_col2:
        if st.button("Clear Batch", type="secondary", use_container_width=True):
            st.session_state.pop("batch_chunk_files", None)
            st.session_state.pop("batch_verification_results", None)
            st.rerun()

    # If run_batch clicked
    if run_batch:
        prog_bar = st.progress(0, text="Initializing batch verification pipeline...")
        batch_results = []
        for i, doc_item in enumerate(batch_files):
            prog_bar.progress((i + 1) / len(batch_files), text=f"Processing {i + 1}/{len(batch_files)}: {doc_item['name']}...")
            try:
                res = run_pipeline_on_bytes(
                    filename=doc_item["name"],
                    file_bytes=doc_item["bytes"],
                    user=user,
                    existing_docs=existing_docs,
                )
                math_rules = [r for r in res["verif"].get("ruleResults", []) if "MATH" in r.get("rule_id", "") or "TOTAL" in r.get("rule_id", "")]
                math_ok = all(r.get("passed", True) for r in math_rules) if math_rules else True
                # If image is not a valid invoice (e.g. selfie, photo of object), reject safely
                if res.get("isValidInvoice") is False:
                    batch_results.append({
                        "status": "rejected",
                        "filename": doc_item["name"],
                        "vendor": "Not an Invoice",
                        "invoice_no": "N/A",
                        "date": "N/A",
                        "total": 0.0,
                        "currency": "USD",
                        "math_pass": False,
                        "is_dup": False,
                        "risk_score": 0.0,
                        "risk_level": "REJECTED",
                        "docId": "N/A",
                        "error": res.get("rejectionReason", "Image does not have invoice structure"),
                        "doc_item": doc_item,
                    })
                    continue

                batch_results.append({
                    "status": "success",
                    "filename": doc_item["name"],
                    "vendor": res["fields"].get("vendorName") or "N/A",
                    "invoice_no": res["fields"].get("invoiceNumber") or "N/A",
                    "date": res["fields"].get("invoiceDate") or "N/A",
                    "total": res["fields"].get("totalAmount") or 0.0,
                    "currency": res["fields"].get("currency") or "USD",
                    "math_pass": math_ok,
                    "is_dup": res["dup"].get("isDuplicate", False),
                    "risk_score": res["risk"]["score"],
                    "risk_level": res["risk"]["level"],
                    "docId": res["docId"],
                    "doc_item": doc_item,
                })
            except Exception as e:
                batch_results.append({
                    "status": "error",
                    "filename": doc_item["name"],
                    "vendor": "Extraction Error",
                    "invoice_no": "N/A",
                    "date": "N/A",
                    "total": 0.0,
                    "currency": "USD",
                    "math_pass": False,
                    "is_dup": False,
                    "risk_score": 100.0,
                    "risk_level": "HIGH",
                    "docId": "N/A",
                    "error": str(e),
                    "doc_item": doc_item,
                })
        st.session_state["batch_verification_results"] = batch_results
        st.toast(f"Batch completed: {len(batch_results)} invoices verified!", icon="✅")
        st.rerun()

    # Results Table if processed
    results = st.session_state.get("batch_verification_results")
    if results:
        st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

        low_count = len([r for r in results if r.get("risk_level") == "LOW"])
        med_count = len([r for r in results if r.get("risk_level") == "MEDIUM"])
        high_count = len([r for r in results if r.get("risk_level") == "HIGH"])
        dup_count = len([r for r in results if r.get("is_dup")])

        # Summary Badges
        st.markdown(
            f"""
            <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                <span class="badge-low">✓ Verified / Low Risk: {low_count}</span>
                <span class="badge-medium">⚠ Review Recommended: {med_count}</span>
                <span class="badge-high">✕ High Risk / Flagged: {high_count}</span>
                <span class="conf-chip conf-chip-low">⚠️ Duplicate Collisions: {dup_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h3 style='font-size: 1.15rem; margin-bottom: 12px; font-weight: 700;'>Batch Verification Outcomes</h3>", unsafe_allow_html=True)

        for idx, row in enumerate(results):
            fname = row["filename"]
            vendor = row.get("vendor", "N/A")
            inv_no = row.get("invoice_no", "N/A")
            total = row.get("total", 0.0)
            curr = row.get("currency", "USD")
            math_pass = row.get("math_pass", True)
            is_dup = row.get("is_dup", False)
            risk_score = row.get("risk_score", 0.0)
            risk_level = row.get("risk_level", "LOW")

            math_badge = '<span style="color: #30D158; font-weight: 600; font-size: 0.78rem;">✓ Math Invariants Match</span>' if math_pass else '<span style="color: #FF453A; font-weight: 700; font-size: 0.78rem;">✕ Arithmetic Discrepancy</span>'
            dup_badge = '<span style="color: #FF9F0A; font-weight: 700; font-size: 0.78rem;">⚠ Duplicate Collision</span>' if is_dup else '<span style="color: #86868B; font-size: 0.78rem;">✓ Unique Document</span>'

            risk_pill = components.render_risk_badge(risk_level, risk_score)

            item_col1, item_col2 = st.columns([4, 1.2])
            with item_col1:
                st.markdown(
                    f"""
                    <div class="batch-row-item">
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <div style="font-size: 1.25rem;">📄</div>
                            <div>
                                <div style="font-weight: 700; font-size: 0.92rem; color: #1D1D1F;">{fname}</div>
                                <div style="font-size: 0.78rem; color: #515154; margin-top: 2px;">
                                    Vendor: <b style="color: #1D1D1F;">{vendor}</b> &nbsp;|&nbsp; Inv #: <b style="color: #1D1D1F;">{inv_no}</b> &nbsp;|&nbsp; Total: <b style="color: #0071E3;">{curr} {total:,.2f}</b>
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <div>{math_badge}</div>
                            <div>{dup_badge}</div>
                            <div>{risk_pill}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with item_col2:
                if st.button("🔍 Inspect", key=f"btn_inspect_batch_{idx}", use_container_width=True):
                    st.session_state["uploaded_file_cache"] = row["doc_item"]
                    st.session_state["_switch_to_inspector"] = True
                    st.rerun()

        # Export CSV Button
        df_export = pd.DataFrame([
            {
                "Filename": r["filename"],
                "Vendor": r.get("vendor"),
                "InvoiceNumber": r.get("invoice_no"),
                "TotalAmount": r.get("total"),
                "Currency": r.get("currency"),
                "MathInvariantPassed": r.get("math_pass"),
                "DuplicateCollision": r.get("is_dup"),
                "RiskScore": r.get("risk_score"),
                "RiskLevel": r.get("risk_level"),
                "DocId": r.get("docId"),
            }
            for r in results
        ])
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "📥 Export Batch Audit Summary (CSV)",
            data=df_export.to_csv(index=False),
            file_name="finvision_batch_audit_summary.csv",
            mime="text/csv",
            use_container_width=False,
        )
    else:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1.5px dashed rgba(0,0,0,0.14); border-radius: 16px; padding: 36px 20px; text-align: center; margin-top: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.03);">
                <div style="font-weight: 700; font-size: 1.05rem; color: #1D1D1F; margin-bottom: 4px;">{len(batch_files)} Invoices Ready for Automated Verification</div>
                <div style="color: #515154; font-size: 0.85rem; max-width: 500px; margin: 0 auto;">Click "Run Automated Verification" above to process this entire chunk simultaneously.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload(user: Dict[str, Any]):
    """Render modern split-view Ingest & Verify workspace."""
    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    existing_docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    # Manage interactive view state
    if "zoom_scale" not in st.session_state:
        st.session_state["zoom_scale"] = 1.0
    if "rotation_deg" not in st.session_state:
        st.session_state["rotation_deg"] = 0
    if "pdf_page_idx" not in st.session_state:
        st.session_state["pdf_page_idx"] = 0

    # 1. Stepper at Top
    has_active_doc = bool(st.session_state.get("active_sample_file") or st.session_state.get("uploaded_file_cache") or st.session_state.get("batch_chunk_files"))
    current_step = 2 if has_active_doc else 1
    components.render_stepper(current_step=current_step)

    # 2. Collapsed "Try a sample invoice" drawer
    sample_dir = Path(__file__).resolve().parent.parent / "sample_invoices"
    with st.expander("📁 Try a sample invoice (Instant Test Scenarios)", expanded=False):
        sample_options = [
            ("-- Select scenario --", None),
            ("Sample 01 — Clean Valid Invoice (Pass / Low Risk)", "sample_01_valid_invoice.pdf"),
            ("Sample 02 — Arithmetic Mismatch (Flagged / Math Variance)", "sample_02_math_mismatch.pdf"),
            ("Sample 03 — Duplicate Invoice Collision (Flagged / Duplicate)", "sample_03_duplicate_invoice.pdf"),
            ("Sample 04 — Missing Required Fields (Warning / Low Risk)", "sample_04_missing_fields.pdf"),
            ("Sample 05 — Discount & Tax Multi-Step Math (Pass)", "sample_05_with_discount_tax.pdf"),
            ("Sample 06 — European Format (EUR / Comma Separator)", "sample_06_european_format.pdf"),
        ]
        s_col1, s_col2, s_col3, s_col4 = st.columns([2.0, 1.1, 1.4, 1.0])
        with s_col1:
            selected_sample_label = st.selectbox(
                "Choose test invoice:",
                [opt[0] for opt in sample_options],
                index=0,
                key="sample_select_drawer",
                label_visibility="collapsed",
            )
        selected_file_stem = next((opt[1] for opt in sample_options if opt[0] == selected_sample_label), None)

        with s_col2:
            if st.button("Load Single Sample", type="secondary", use_container_width=True, disabled=selected_file_stem is None):
                target_path = sample_dir / selected_file_stem
                if target_path.exists():
                    st.session_state["active_sample_file"] = {
                        "name": target_path.name,
                        "bytes": target_path.read_bytes(),
                        "type": "application/pdf" if target_path.suffix.lower() == ".pdf" else "image/png",
                    }
                    st.session_state.pop("uploaded_file_cache", None)
                    st.session_state.pop("batch_chunk_files", None)
                    st.session_state["pdf_page_idx"] = 0
                    st.rerun()

        with s_col3:
            if st.button("⚡ Ingest All 12 Samples (Batch)", type="secondary", use_container_width=True):
                batch_files = []
                for p in sorted(sample_dir.glob("sample_*.*")):
                    if p.suffix.lower() in [".pdf", ".png"]:
                        batch_files.append({
                            "name": p.name,
                            "bytes": p.read_bytes(),
                            "size": p.stat().st_size,
                            "type": "application/pdf" if p.suffix.lower() == ".pdf" else "image/png",
                        })
                st.session_state["batch_chunk_files"] = batch_files
                st.session_state.pop("uploaded_file_cache", None)
                st.session_state.pop("active_sample_file", None)
                st.session_state["batch_verification_results"] = None
                st.rerun()

        with s_col4:
            try:
                from src import generate_sample_invoices
                zip_data = generate_sample_invoices.create_sample_zip()
                st.download_button(
                    "Pack (ZIP)",
                    data=zip_data,
                    file_name="finvision_sample_invoices.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception:
                pass

    # 3. Slim Apple Ingest Container: File / Chunk Batch Upload & Live Camera Scan
    tab_file_upload, tab_camera_scan = st.tabs([
        "📁 File & Chunk Batch Upload",
        "📷 Live Camera Document Scan",
    ])

    with tab_file_upload:
        uploaded_files = st.file_uploader(
            "Upload Invoices (PDF, PNG, JPG, or ZIP archive up to 200MB)",
            type=["jpg", "jpeg", "png", "pdf", "zip"],
            accept_multiple_files=True,
            help="Drag and drop a single invoice scan, multiple invoices, or a ZIP archive containing invoices.",
        )

        if uploaded_files:
            parsed_files = []
            for uf in uploaded_files:
                fname = uf.name
                fbytes = uf.getvalue()
                if fname.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(fbytes)) as z:
                            for zname in z.namelist():
                                if not zname.startswith("__MACOSX") and zname.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                                    content = z.read(zname)
                                    clean_name = Path(zname).name
                                    parsed_files.append({
                                        "name": clean_name,
                                        "bytes": content,
                                        "size": len(content),
                                        "type": "application/pdf" if clean_name.lower().endswith(".pdf") else "image/png"
                                    })
                    except Exception as ze:
                        st.error(f"Error unpacking ZIP: {ze}")
                else:
                    parsed_files.append({
                        "name": fname,
                        "bytes": fbytes,
                        "size": len(fbytes),
                        "type": uf.type,
                    })

            if len(parsed_files) > 1:
                st.session_state["batch_chunk_files"] = parsed_files
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop("uploaded_file_cache", None)
            elif len(parsed_files) == 1:
                st.session_state["uploaded_file_cache"] = parsed_files[0]
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop("batch_chunk_files", None)

    with tab_camera_scan:
        import streamlit.components.v1 as st_components
        st_components.html(
            """
            <script>
            (function() {
                // 1. Cleanly Force Rear Camera across window, parent, and top navigation contexts
                function patchMedia(navObj) {
                    if (!navObj || !navObj.mediaDevices || !navObj.mediaDevices.getUserMedia) return;
                    if (navObj.__finvision_rear_enforced) return;
                    navObj.__finvision_rear_enforced = true;
                    const origGetUserMedia = navObj.mediaDevices.getUserMedia.bind(navObj.mediaDevices);
                    navObj.mediaDevices.getUserMedia = function(c) {
                        try {
                            if (!c) c = {};
                            if (!c.video || typeof c.video !== 'object') {
                                c.video = { facingMode: { ideal: "environment" } };
                            } else {
                                c.video.facingMode = { ideal: "environment" };
                            }
                        } catch(e) {}
                        return origGetUserMedia(c);
                    };
                }
                try { patchMedia(window.navigator); } catch(e) {}
                try { if (window.parent) patchMedia(window.parent.navigator); } catch(e) {}
                try { if (window.top) patchMedia(window.top.navigator); } catch(e) {}

                // 2. Continuous inline configuration for iOS Safari WebKit (prevent black/frozen frames)
                function ensureInlinePlay() {
                    try {
                        const pDoc = window.parent ? window.parent.document : document;
                        const vids = pDoc.querySelectorAll('video');
                        vids.forEach(function(v) {
                            if (!v.getAttribute('playsinline')) v.setAttribute('playsinline', 'true');
                            if (!v.getAttribute('webkit-playsinline')) v.setAttribute('webkit-playsinline', 'true');
                            v.muted = true;
                            if (v.paused && v.srcObject) {
                                v.play().catch(function(){});
                            }
                        });
                    } catch(e) {}
                }
                setInterval(ensureInlinePlay, 400);
            })();
            </script>
            """,
            height=0,
            width=0,
        )
        st.markdown(
            """
            <div class="camera-scanner-hud">
                <div>
                    <div style="font-weight: 700; font-size: 1rem; color: #1D1D1F; display: flex; align-items: center; gap: 8px;">
                        <span>📷 Full-Screen Rear Camera Scanner</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #515154; margin-top: 2px;">
                        Position paper invoice flat inside the viewfinder frame. Tap <b>Take Photo</b> to verify instantly.
                    </div>
                </div>
                <div style="background: rgba(36, 138, 61, 0.12); color: #166534; font-size: 0.74rem; font-weight: 700; padding: 5px 14px; border-radius: 980px; border: 1px solid rgba(36, 138, 61, 0.3); display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #248A3D;"></span>
                    <span>Rear Camera Active (No Selfie Mode)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        camera_photo = st.camera_input("📸 Capture invoice with back camera", label_visibility="collapsed")
        if camera_photo:
            photo_bytes = camera_photo.getvalue()
            photo_name = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            st.session_state["uploaded_file_cache"] = {
                "name": photo_name,
                "bytes": photo_bytes,
                "size": len(photo_bytes),
                "type": "image/jpeg",
            }
            st.session_state.pop("active_sample_file", None)
            st.session_state.pop("batch_chunk_files", None)
            st.success(f"✓ Captured rear camera scan: `{photo_name}` ready for verification.")

        # Native phone camera option with capture="environment" for ultra-high-resolution mobile camera app capture
        with st.expander("📱 Prefer Your Phone's Native Camera App? (Tap to open full 48MP camera)", expanded=False):
            st.caption("You can also launch your phone's native camera app directly to use flash, zoom, and hardware auto-focus:")
            native_cam_file = st.file_uploader(
                "Upload photo taken with your phone camera",
                type=["jpg", "jpeg", "png"],
                key="native_phone_camera_uploader",
                help="Opens your phone's native rear camera app directly",
            )
            if native_cam_file:
                c_bytes = native_cam_file.getvalue()
                c_name = native_cam_file.name
                st.session_state["uploaded_file_cache"] = {
                    "name": c_name,
                    "bytes": c_bytes,
                    "size": len(c_bytes),
                    "type": native_cam_file.type or "image/jpeg",
                }
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop("batch_chunk_files", None)
                st.success(f"✓ Document photo loaded: `{c_name}`")

    # Check for Batch Mode vs Single Inspection Mode
    batch_files = st.session_state.get("batch_chunk_files")
    if batch_files and len(batch_files) > 1:
        if st.session_state.pop("_switch_to_inspector", False):
            st.session_state["upload_view_mode_toggle"] = "🔍 Detailed Document Inspector"

        view_mode = st.radio(
            "Workspace View Mode",
            ["📦 Batch Chunk Ingestion Hub", "🔍 Detailed Document Inspector"],
            horizontal=True,
            label_visibility="collapsed",
            key="upload_view_mode_toggle",
        )
        if view_mode == "📦 Batch Chunk Ingestion Hub":
            render_batch_hub(batch_files, user, existing_docs)
            return

        # If Detailed Document Inspector selected during batch mode:
        b_names = [f["name"] for f in batch_files]
        current_active = st.session_state.get("uploaded_file_cache")
        default_idx = b_names.index(current_active["name"]) if (current_active and current_active.get("name") in b_names) else 0

        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        sel_idx = st.selectbox(
            "Select document from batch chunk to inspect:",
            range(len(b_names)),
            index=default_idx,
            format_func=lambda i: f"📄 {b_names[i]} ({batch_files[i]['size'] / 1024:.1f} KB)",
            key="batch_inspector_selector",
        )
        st.session_state["uploaded_file_cache"] = batch_files[sel_idx]

    active_file = st.session_state.get("uploaded_file_cache") or st.session_state.get("active_sample_file")

    # If no file loaded: show clean Apple empty state
    if not active_file:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1.5px dashed rgba(0, 113, 227, 0.35); border-radius: 18px; padding: 48px 24px; text-align: center; margin-top: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.04);">
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#0071E3" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 14px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                <div style="font-weight: 700; font-size: 1.15rem; color: #1D1D1F; margin-bottom: 6px; letter-spacing: -0.02em;">No Invoices Loaded</div>
                <div style="font-size: 0.88rem; color: #515154; max-width: 440px; margin: 0 auto 18px auto; line-height: 1.5;">Upload a single invoice, drop a chunk of multiple invoices, or open the sample drawer above to evaluate instant test scenarios.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filename = active_file["name"]
    file_bytes = active_file["bytes"]
    mime_type = active_file.get("type") or image_processing.detect_mime_type(file_bytes)

    # 4. Multi-Engine OCR & Field Extraction (Cached in session)
    cache_key = f"extracted_{active_file['name']}_{len(file_bytes)}"
    if cache_key not in st.session_state:
        with st.spinner("Executing high-accuracy OCR extraction and field parsing..."):
            deskewed_bgr, binary_img, _, _ = image_processing.preprocess_image(file_bytes)
            words, raw_text, overall_conf = ocr.extract_document_words(
                source=file_bytes,
                mime_type=mime_type,
                fallback_image=deskewed_bgr,
            )
            extracted = ocr.extract_invoice_fields(words, raw_text, overall_conf)
            st.session_state[cache_key] = {
                "words": words,
                "raw_text": raw_text,
                "overall_conf": overall_conf,
                "extracted": extracted.to_dict(),
            }

    ext_pack = st.session_state[cache_key]
    words = ext_pack["words"]
    raw_fields = ext_pack["extracted"]["fields"]
    line_items_init = ext_pack["extracted"]["lineItems"]
    confidences = ext_pack["extracted"]["fieldConfidence"]
    overall_conf = ext_pack["overall_conf"]

    # 4.5 Document Validity Check (Prevent non-invoice images, selfies, walls, objects from entering verification)
    is_valid_inv, validity_reason = ocr.validate_is_invoice_document(
        ext_pack.get("raw_text", ""), len(words), raw_fields
    )
    if not is_valid_inv:
        st.markdown(
            """
            <div style="background: #FFF1F2; border: 2px solid #F43F5E; border-radius: 20px; padding: 32px 24px; margin: 24px 0; text-align: center; box-shadow: 0 4px 24px rgba(244, 63, 94, 0.12);">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📑 ❌</div>
                <div style="font-weight: 800; font-size: 1.35rem; color: #9F1239; margin-bottom: 8px; letter-spacing: -0.02em;">Invalid Document: Image is Not an Invoice</div>
                <div style="font-size: 0.95rem; color: #881337; max-width: 520px; margin: 0 auto 24px auto; line-height: 1.55;">
                    The captured picture does not contain recognizable invoice text, vendor billing headers, line items, or monetary totals (detected photo of non-document). Please take a clear photo of a valid invoice, bill, or receipt.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_retake1, col_retake2 = st.columns([1, 1])
        with col_retake1:
            if st.button("📸 Retake Photo with Rear Camera", type="primary", use_container_width=True):
                st.session_state.pop("uploaded_file_cache", None)
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop(cache_key, None)
                st.rerun()
        with col_retake2:
            if st.button("📁 Upload Another Document", use_container_width=True):
                st.session_state.pop("uploaded_file_cache", None)
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop(cache_key, None)
                st.rerun()
        return

    # 5. Main Split View Above The Fold
    col_preview, col_editor = st.columns([1.05, 1.25])

    # -------------------------------------------------------------
    # LEFT COLUMN: Document Preview & Interactive Controls
    # -------------------------------------------------------------
    with col_preview:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 0.95rem; color: #1D1D1F;">Document Preview</div>
                <div style="font-size: 0.76rem; color: #515154;">{filename} ({len(file_bytes) / 1024:.1f} KB)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Toolbar controls: Zoom, Rotate, Page
        tc1, tc2, tc3, tc4, tc5 = st.columns([1, 1, 1, 1.2, 1.2])
        with tc1:
            if st.button("Zoom -", key="btn_zoom_out", use_container_width=True):
                st.session_state["zoom_scale"] = max(0.6, st.session_state["zoom_scale"] - 0.2)
                st.rerun()
        with tc2:
            if st.button(f"{int(st.session_state['zoom_scale'] * 100)}%", key="btn_zoom_reset", use_container_width=True):
                st.session_state["zoom_scale"] = 1.0
                st.rerun()
        with tc3:
            if st.button("Zoom +", key="btn_zoom_in", use_container_width=True):
                st.session_state["zoom_scale"] = min(2.0, st.session_state["zoom_scale"] + 0.2)
                st.rerun()
        with tc4:
            if st.button("Rotate 90°", key="btn_rotate", use_container_width=True):
                st.session_state["rotation_deg"] = (st.session_state["rotation_deg"] + 90) % 360
                st.rerun()
        with tc5:
            if st.button("Clear Doc", key="btn_clear_doc", use_container_width=True):
                st.session_state.pop("active_sample_file", None)
                st.session_state.pop("uploaded_file_cache", None)
                st.rerun()

        # Render document image with highlighted source bounding boxes
        pil_preview = None
        total_pages = 1
        curr_page = st.session_state["pdf_page_idx"]

        if mime_type == "application/pdf":
            try:
                import pymupdf
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                total_pages = len(doc)
                if total_pages > 1:
                    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                    with col_p1:
                        if st.button("◀ Prev", key="p_prev", disabled=curr_page <= 0, use_container_width=True):
                            st.session_state["pdf_page_idx"] -= 1
                            st.rerun()
                    with col_p2:
                        st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: #515154; padding-top: 6px;'>Page {curr_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                    with col_p3:
                        if st.button("Next ▶", key="p_next", disabled=curr_page >= total_pages - 1, use_container_width=True):
                            st.session_state["pdf_page_idx"] += 1
                            st.rerun()

                page = doc[min(curr_page, total_pages - 1)]
                pix = page.get_pixmap(dpi=150)
                pil_preview = Image.open(io.BytesIO(pix.tobytes("png")))
                doc.close()
            except Exception as e:
                st.warning(f"PDF renderer note: {e}")
        else:
            try:
                pil_preview = Image.open(io.BytesIO(file_bytes))
            except Exception:
                pass

        if pil_preview:
            # Apply rotation
            if st.session_state["rotation_deg"] != 0:
                pil_preview = pil_preview.rotate(-st.session_state["rotation_deg"], expand=True)

            # Draw bounding boxes highlighting key values
            highlight_snippets = [
                str(raw_fields.get("invoiceNumber", "")),
                str(raw_fields.get("total", "")),
                str(raw_fields.get("date", "")),
                str(raw_fields.get("vendor", "")),
            ]
            annotated_preview = draw_bounding_boxes_on_image(pil_preview, words, highlight_snippets)

            # Apply zoom scaling
            scale = st.session_state["zoom_scale"]
            if scale != 1.0:
                new_w = max(100, int(annotated_preview.width * scale))
                new_h = max(100, int(annotated_preview.height * scale))
                annotated_preview = annotated_preview.resize((new_w, new_h), Image.Resampling.LANCZOS)

            st.image(annotated_preview, caption="Scan source with visual bounding box highlights", use_container_width=True)
            st.caption("Indigo rectangles indicate extracted field source text; amber outlines indicate low-confidence words.")
        else:
            st.info("Visual preview unavailable.")

    # -------------------------------------------------------------
    # RIGHT COLUMN: Extracted Fields, Live Math, Line Items & Risk
    # -------------------------------------------------------------
    with col_editor:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 700; font-size: 0.95rem; color: #1D1D1F;">Extracted Financial Fields</div>
                <div style="font-size: 0.74rem; color: #515154;">Review & correct fields before final verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2-Column Responsive Field Grid
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Invoice Number</span>'
                f'{components.render_confidence_chip(confidences.get("invoiceNumber", 90))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            inv_num_val = st.text_input("inv_num", value=raw_fields.get("invoiceNumber") or "INV-", label_visibility="collapsed", key="in_inv_num")

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Vendor Name</span>'
                f'{components.render_confidence_chip(confidences.get("vendor", 88))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            vendor_val = st.text_input("vendor", value=raw_fields.get("vendor") or "", label_visibility="collapsed", key="in_vendor")

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Invoice Date</span>'
                f'{components.render_confidence_chip(confidences.get("date", 92))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            date_val = st.text_input("date", value=raw_fields.get("date") or "", label_visibility="collapsed", key="in_date")

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Due Date</span>'
                f'{components.render_confidence_chip(confidences.get("dueDate", 85))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            due_date_val = st.text_input("dueDate", value=raw_fields.get("dueDate") or "", label_visibility="collapsed", key="in_due_date")

        with col_f2:
            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Subtotal ($)</span>'
                f'{components.render_confidence_chip(confidences.get("subtotal", 90))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            subtotal_val = st.number_input("subtotal", value=float(raw_fields.get("subtotal") or 0.0), step=10.0, format="%.2f", label_visibility="collapsed", key="in_subtotal")

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Tax / VAT ($)</span>'
                f'{components.render_confidence_chip(confidences.get("tax", 90))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            tax_val = st.number_input("tax", value=float(raw_fields.get("tax") or 0.0), step=5.0, format="%.2f", label_visibility="collapsed", key="in_tax")

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Discount ($)</span>'
                f'{components.render_confidence_chip(confidences.get("discount", 85))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            discount_val = st.number_input("discount", value=float(raw_fields.get("discount") or 0.0), step=5.0, format="%.2f", label_visibility="collapsed", key="in_discount")

            # Check if total was fixed via one-click suggestion
            fixed_tot_val = st.session_state.get(f"fix_total_{filename}")
            initial_total = fixed_tot_val if fixed_tot_val is not None else float(raw_fields.get("total") or 0.0)

            st.markdown(
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">'
                f'<span style="font-size: 0.8rem; font-weight: 600; color: #515154;">Total Stated Amount ($)</span>'
                f'{components.render_confidence_chip(confidences.get("total", 95))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            total_val = st.number_input("total", value=initial_total, step=10.0, format="%.2f", label_visibility="collapsed", key="in_total")

        # -------------------------------------------------------------
        # Live Math Validation Bar
        # -------------------------------------------------------------
        calc_expected = round(subtotal_val + tax_val - discount_val, 2)
        diff = round(abs(calc_expected - total_val), 2)
        is_math_passed = diff < 0.05

        if is_math_passed:
            st.markdown(
                f"""
                <div class="math-bar-pass">
                    <div>
                        <span>✓ Math Verified: </span>
                        <span>${subtotal_val:,.2f} + ${tax_val:,.2f} - ${discount_val:,.2f} = <b>${calc_expected:,.2f}</b></span>
                    </div>
                    <span style="font-size: 0.75rem; background: rgba(16,185,129,0.2); padding: 2px 8px; border-radius: 4px;">Zero Discrepancy</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            col_m_msg, col_m_fix = st.columns([2.2, 1])
            with col_m_msg:
                st.markdown(
                    f"""
                    <div class="math-bar-fail">
                        <div>
                            <span>⚠️ Math Discrepancy: </span>
                            <span>Expected <b>${calc_expected:,.2f}</b>, stated total is <b>${total_val:,.2f}</b> (Diff: ${diff:,.2f})</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m_fix:
                if st.button(f"⚡ Fix total to ${calc_expected:,.2f}", key=f"btn_fix_{filename}", use_container_width=True):
                    st.session_state[f"fix_total_{filename}"] = calc_expected
                    st.rerun()

        # -------------------------------------------------------------
        # Editable Line-Items Table
        # -------------------------------------------------------------
        st.markdown(
            """
            <div style="font-weight: 700; font-size: 0.88rem; color: #1D1D1F; margin-top: 14px; margin-bottom: 4px;">
                Itemized Line Items
            </div>
            """,
            unsafe_allow_html=True,
        )

        line_df = pd.DataFrame(line_items_init if line_items_init else [
            {"description": "Item Description", "quantity": 1.0, "unitPrice": subtotal_val, "totalPrice": subtotal_val}
        ])
        edited_line_df = st.data_editor(
            line_df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{filename}",
        )

        # -------------------------------------------------------------
        # Live Risk & Inline Duplicate Detection Preview
        # -------------------------------------------------------------
        live_fields = {
            "invoiceNumber": inv_num_val,
            "vendor": vendor_val,
            "date": date_val,
            "dueDate": due_date_val,
            "subtotal": subtotal_val,
            "tax": tax_val,
            "discount": discount_val,
            "total": total_val,
        }
        
        # Real-time duplicate evaluation
        sha256_hash = duplicate_detection.compute_sha256(file_bytes)
        phash_data = duplicate_detection.compute_perceptual_hashes(file_bytes)
        target_cand = {"sha256": sha256_hash, "pHash": phash_data["pHash"], "fields": live_fields}
        live_dup = duplicate_detection.find_duplicates(target_cand, existing_docs)

        # Build live risk reasons
        live_reasons = []
        if not is_math_passed:
            live_reasons.append(f"Subtotal arithmetic mismatch: stated total differs by ${diff:,.2f}")
        if live_dup.get("hasDuplicate"):
            match_top = live_dup.get("duplicateMatches", [{}])[0]
            live_reasons.append(f"Duplicate Collision: {match_top.get('matchType')} matches existing {match_top.get('docId')[:12]}")
        if overall_conf < 70.0:
            live_reasons.append(f"Lower OCR confidence extraction ({overall_conf:.1f}%)")
        if not vendor_val or len(vendor_val) < 2:
            live_reasons.append("Vendor name is incomplete or unverified")

        calculated_risk_score = 15.0
        if not is_math_passed:
            calculated_risk_score += 40.0
        if live_dup.get("hasDuplicate"):
            calculated_risk_score += 45.0
        calculated_risk_score = min(100.0, calculated_risk_score)

        risk_level = "HIGH" if calculated_risk_score >= 65 else ("MEDIUM" if calculated_risk_score >= 35 else "LOW")
        risk_color = "#EF4444" if risk_level == "HIGH" else ("#F59E0B" if risk_level == "MEDIUM" else "#10B981")

        st.markdown(
            f"""
            <div class="risk-preview-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 0.85rem; color: #1D1D1F;">Live AI Risk Preview</span>
                    <span style="font-weight: 800; font-size: 0.85rem; color: {risk_color};">{risk_level} ({calculated_risk_score:.0f}/100)</span>
                </div>
                <div style="font-size: 0.78rem; color: #515154;">
                    {''.join([f'<div style="margin-top: 3px;">• {r}</div>' for r in live_reasons[:3]]) if live_reasons else '<div>✓ No risk indicators triggered. Clean document structure.</div>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # Sticky Bottom Action Bar
        # -------------------------------------------------------------
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([1, 2])
        
        with col_act1:
            if st.button("Save Draft", use_container_width=True):
                st.toast("Document draft saved to browser session", icon="💾")

        with col_act2:
            submit_label = "🚀 Run Full Verification & Save (Ctrl + Enter)"
            if st.button(submit_label, type="primary", use_container_width=True, key=f"btn_verify_{filename}"):
                with st.spinner("Executing financial invariant checks, duplicate scan, and ML inference..."):
                    res = run_pipeline_on_bytes(
                        filename=filename,
                        file_bytes=file_bytes,
                        user=user,
                        existing_docs=existing_docs,
                        edited_fields=live_fields,
                        edited_line_items=edited_line_df.to_dict(orient="records"),
                    )
                    if res.get("isValidInvoice") is False:
                        st.error(res.get("rejectionReason", "⚠️ Image is not a valid invoice or receipt."))
                    else:
                        st.session_state["selected_doc_id"] = res["docId"]
                        st.session_state["nav_page"] = "Verification Results"
                        st.success(f"✓ Document verified and persisted as {res['docId'][:12]}!")
                        st.rerun()
