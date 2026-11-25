"""Verification Results & Review Workspace for FinVision."""

from decimal import Decimal
import io
import streamlit as st
from PIL import Image

from frontend.components import render_risk_badge, render_rule_card
from src import firebase_service, verification, risk_assessment


def render_verify(user: dict):
    """Render interactive document verification, editable field audit, and human decision workspace."""
    st.markdown("## 🛡️ Document Verification & Audit Workspace")
    st.caption("Inspect document scans, verify deterministic mathematical checks, review ML probability insights, and submit formal reviewer decisions.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    if not docs:
        st.info("No documents available for verification. Please upload an invoice first.")
        return

    # Select document from audit queue
    doc_options = {
        d["docId"]: f"📄 {d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} | {d.get('fields', {}).get('vendor') or 'Unknown'} (${float(d.get('fields', {}).get('total', 0.0) or 0.0):,.2f}) — {d.get('risk', {}).get('level', 'NEW')}"
        for d in docs
    }

    selected_id = st.session_state.get("selected_doc_id")
    if selected_id not in doc_options:
        selected_id = list(doc_options.keys())[0]

    chosen_id = st.selectbox(
        "Select Document from Audit Queue",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x],
        index=list(doc_options.keys()).index(selected_id),
    )
    st.session_state["selected_doc_id"] = chosen_id

    doc = firebase_service.get_document(chosen_id)
    if not doc:
        st.error("Document not found.")
        return

    fields = doc.get("fields", {})
    confidences = doc.get("fieldConfidence", {})
    risk = doc.get("risk", {})
    verif = doc.get("verification", {})
    ml = doc.get("ml", {})
    cv = doc.get("cv", {})
    file_info = doc.get("file", {})
    review = doc.get("review") or {}

    # 1. Top Executive Status Bar
    st.markdown("<div style='margin-top: 8px; margin-bottom: 16px;'>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 14px 18px; margin-bottom: 0;">
                <div class="fin-card-title">Risk Assessment</div>
                <div>{render_risk_badge(risk.get('level', 'LOW'), risk.get('score', 0))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 14px 18px; margin-bottom: 0;">
                <div class="fin-card-title">Processing Status</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC;">{doc.get('status', 'verified').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b3:
        dec = review.get("decision") or "unreviewed"
        dec_color = "#10B981" if dec == "approved" else ("#EF4444" if dec in ["rejected", "flagged"] else "#F59E0B")
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 14px 18px; margin-bottom: 0;">
                <div class="fin-card-title">Review Decision</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: {dec_color};">{dec.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b4:
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 14px 18px; margin-bottom: 0;">
                <div class="fin-card-title">OCR Confidence</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC;">{doc.get('overallConfidence', 0):.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Main Two-Column Layout (Scan Preview vs Verification Details)
    col_scan, col_details = st.columns([1, 1.25])

    with col_scan:
        st.markdown("### 📄 Original Document")
        storage_path = file_info.get("storagePath")
        if storage_path:
            img_bytes = firebase_service.download_invoice_bytes(storage_path)
            if img_bytes:
                try:
                    if img_bytes.startswith(b"%PDF"):
                        import pymupdf
                        doc_pdf = pymupdf.open(stream=img_bytes, filetype="pdf")
                        page = doc_pdf[0]
                        pix = page.get_pixmap(dpi=150)
                        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                    else:
                        pil_img = Image.open(io.BytesIO(img_bytes))
                    st.image(pil_img, caption=file_info.get("filename", "Invoice Scan"), use_container_width=True)
                except Exception as e:
                    st.info(f"File: {file_info.get('filename')} ({len(img_bytes)} bytes)")
            else:
                st.caption("File preview unavailable from storage.")
        else:
            st.caption("No storage path associated with record.")

        # Document metadata
        st.markdown(
            f"""
            <div style="background-color: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 0.82rem; color: #94A3B8;">
                <div><b>Filename:</b> <span style="color: #F8FAFC;">{file_info.get('filename', 'N/A')}</span></div>
                <div><b>SHA-256:</b> <span style="font-family: monospace; color: #60A5FA;">{file_info.get('sha256', 'N/A')[:20]}...</span></div>
                <div><b>pHash:</b> <span style="font-family: monospace; color: #A78BFA;">{file_info.get('pHash', 'N/A')}</span></div>
                <div><b>Created:</b> {doc.get('createdAt', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown("### 🔍 Verification Findings")
        tab_rules, tab_fields, tab_dup, tab_ai, tab_review = st.tabs([
            "Rule Engine",
            "Extracted Data",
            "Duplicates",
            "ML & CV Analysis",
            "Reviewer Sign-Off",
        ])

        # ---------------------------------------------------------
        # Tab 1: Deterministic Rules
        # ---------------------------------------------------------
        with tab_rules:
            st.markdown("#### Deterministic Financial Invariants")
            rule_results = verif.get("ruleResults", [])
            if rule_results:
                for r in rule_results:
                    render_rule_card(r)
            else:
                st.info("No validation outcomes recorded.")

            st.markdown("#### Risk Assessment Drivers")
            for reason in risk.get("reasons", []):
                st.markdown(f"- {reason}")

        # ---------------------------------------------------------
        # Tab 2: Extracted Data & Confidence
        # ---------------------------------------------------------
        with tab_fields:
            st.markdown("#### Extracted Header Fields")
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.text_input("Invoice Number", value=fields.get("invoiceNumber", "N/A"), disabled=True, key="v_inv")
                st.text_input("Vendor", value=fields.get("vendor", "N/A"), disabled=True, key="v_vend")
                st.text_input("Invoice Date", value=fields.get("date", "N/A"), disabled=True, key="v_date")
                st.text_input("Due Date", value=fields.get("dueDate", "N/A"), disabled=True, key="v_due")
            with h_col2:
                st.text_input("Subtotal", value=f"${float(fields.get('subtotal', 0.0) or 0.0):,.2f}", disabled=True, key="v_sub")
                st.text_input("Tax / VAT", value=f"${float(fields.get('tax', 0.0) or 0.0):,.2f}", disabled=True, key="v_tax")
                st.text_input("Discount", value=f"${float(fields.get('discount', 0.0) or 0.0):,.2f}", disabled=True, key="v_disc")
                st.text_input("Total", value=f"${float(fields.get('total', 0.0) or 0.0):,.2f}", disabled=True, key="v_tot")

            # Line items table
            line_items = doc.get("lineItems", [])
            if line_items:
                st.markdown("#### Itemized Line Items")
                st.dataframe(line_items, use_container_width=True)

        # ---------------------------------------------------------
        # Tab 3: Duplicate Collisions
        # ---------------------------------------------------------
        with tab_dup:
            dup_matches = verif.get("duplicateMatches", [])
            if dup_matches:
                st.warning(f"⚠️ {len(dup_matches)} Duplicate Collision(s) Detected:")
                for m in dup_matches:
                    st.markdown(f"- **{m.get('matchType')}** (Similarity: `{m.get('similarityScore', 100)}%`): {m.get('reason')}")
            else:
                st.success("✓ No duplicate matches detected for this invoice.")

        # ---------------------------------------------------------
        # Tab 4: Machine Learning & Computer Vision
        # ---------------------------------------------------------
        with tab_ai:
            st.markdown("#### Machine Learning Classifier (Experimental)")
            ml_pred = ml.get("prediction", 0)
            ml_prob = ml.get("probability", 0.0)
            verdict = ml.get("verdict", "normal")

            st.markdown(
                f"""
                <div style="background-color: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; margin-bottom: 12px; color: #E2E8F0;">
                    <div><b>Model Verdict:</b> <span style="font-weight: 700; color: {'#EF4444' if verdict == 'suspicious' else '#10B981'};">{verdict.upper()}</span></div>
                    <div><b>Suspiciousness Probability:</b> <span style="color: #60A5FA; font-weight: 600;">{ml_prob * 100:.1f}%</span></div>
                    <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;"><b>Model Registry:</b> {ml.get('modelVersion', 'v1.0.0')} (Random Forest Baseline)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Computer Vision Forensic Indicators")
            cv_score = cv.get("cvScore", 0.0)
            st.markdown(f"- **Visual Anomaly Score:** `{cv_score:.1f}/100`")
            for ind in cv.get("indicators", []):
                st.markdown(f"- **{ind.get('indicator')}:** {ind.get('details')} (score: `{ind.get('score', 0):.0f}`)")

        # ---------------------------------------------------------
        # Tab 5: Reviewer Sign-Off & Audit History
        # ---------------------------------------------------------
        with tab_review:
            st.markdown("#### Analyst Audit Sign-Off")
            st.caption("Submit formal review decision for this financial document record.")

            current_decision = review.get("decision")
            if current_decision:
                st.info(f"Current Decision: **{current_decision.upper()}** by `{review.get('reviewerUid', 'Analyst')}` on {review.get('reviewedAt', 'N/A')}")
                if review.get("comment"):
                    st.markdown(f"**Reviewer Notes:** {review.get('comment')}")

            with st.form("reviewer_decision_form"):
                decision_choice = st.selectbox(
                    "Review Decision",
                    options=["approved", "rejected", "flagged", "needs_review"],
                    format_func=lambda x: {
                        "approved": "Approve Document (Straight-Through)",
                        "rejected": "Reject Document (Discrepancy Confirmed)",
                        "flagged": "Flag for Investigation (Suspicious)",
                        "needs_review": "Request Additional Requisition Details",
                    }[x],
                )
                reviewer_notes = st.text_area("Audit Notes & Justification", value=review.get("comment", ""), placeholder="Enter formal review notes...")
                submit_decision = st.form_submit_button("Submit Formal Decision", type="primary", use_container_width=True)

                if submit_decision:
                    try:
                        firebase_service.update_document_review_decision(
                            doc_id=chosen_id,
                            reviewer_uid=user.get("uid", "anonymous"),
                            decision=decision_choice,
                            comment=reviewer_notes,
                        )
                        st.success(f"✓ Decision '{decision_choice.upper()}' recorded in audit trail!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record decision: {e}")
