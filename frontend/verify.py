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
    b1, b2, b3, b4, b5 = st.columns(5)
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
                <div style="font-weight: 700; font-size: 1.15rem; color: #1D1D1F;">{doc.get('status', 'verified').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b3:
        dec = review.get("decision") or "unreviewed"
        dec_color = "#16A34A" if dec == "approved" else ("#DC2626" if dec in ["rejected", "flagged"] else "#D97706")
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
                <div style="font-weight: 700; font-size: 1.15rem; color: #1D1D1F;">{doc.get('overallConfidence', 0):.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b5:
        folder_name = doc.get("folder", "Operations")
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 14px 18px; margin-bottom: 0;">
                <div class="fin-card-title">Assigned Folder</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #0071E3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📂 {folder_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Document Quick Action Bar: Edit & Delete
    act_col1, act_col2, act_col_space = st.columns([1, 1, 3])
    with act_col1:
        if st.button("✏️ Edit Fields / Folder", use_container_width=True, key="vc_btn_edit"):
            st.session_state[f"vc_edit_{chosen_id}"] = not st.session_state.get(f"vc_edit_{chosen_id}", False)
            st.session_state[f"vc_del_{chosen_id}"] = False
    with act_col2:
        if st.button("🗑️ Delete Invoice", use_container_width=True, key="vc_btn_del"):
            st.session_state[f"vc_del_{chosen_id}"] = not st.session_state.get(f"vc_del_{chosen_id}", False)
            st.session_state[f"vc_edit_{chosen_id}"] = False

    if st.session_state.get(f"vc_edit_{chosen_id}"):
        with st.container():
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1.5px solid #0071E3; border-radius: 16px; padding: 18px; margin-bottom: 14px; box-shadow: 0 4px 20px rgba(0, 113, 227, 0.08);">
                    <div style="font-weight: 700; font-size: 1rem; color: #1D1D1F;">✏️ Edit Invoice Document</div>
                    <div style="font-size: 0.8rem; color: #515154;">Modify invoice details and folder categorization.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            known_folders = ["Operations", "IT & Software", "Logistics", "Marketing", "Legal & Consulting", "Utilities"]
            cur_fld = doc.get("folder", "Operations")
            fld_opts = sorted(list(set(known_folders + [cur_fld])))
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                sel_fld_vc = st.selectbox("Assign Folder", options=fld_opts, index=fld_opts.index(cur_fld) if cur_fld in fld_opts else 0, key=f"vc_fld_{chosen_id}")
            with f_col2:
                new_fld_vc = st.text_input("Or Custom Folder Name", placeholder="Leave blank to use selected folder", key=f"vc_custom_fld_{chosen_id}")

            e1, e2, e3 = st.columns(3)
            with e1:
                v_inv = st.text_input("Invoice #", value=fields.get("invoiceNumber") or "", key=f"vc_e_inv_{chosen_id}")
                v_vend = st.text_input("Vendor", value=fields.get("vendor") or "", key=f"vc_e_vend_{chosen_id}")
            with e2:
                v_dt = st.text_input("Date", value=fields.get("date") or "", key=f"vc_e_dt_{chosen_id}")
                v_due = st.text_input("Due Date", value=fields.get("dueDate") or "", key=f"vc_e_due_{chosen_id}")
            with e3:
                v_sub = st.number_input("Subtotal ($)", value=float(fields.get("subtotal") or 0.0), step=10.0, format="%.2f", key=f"vc_e_sub_{chosen_id}")
                v_tot = st.number_input("Total ($)", value=float(fields.get("total") or 0.0), step=10.0, format="%.2f", key=f"vc_e_tot_{chosen_id}")

            sv1, sv2 = st.columns([1, 4])
            with sv1:
                if st.button("💾 Save Changes", type="primary", use_container_width=True, key=f"vc_save_{chosen_id}"):
                    final_fld = new_fld_vc.strip() if new_fld_vc.strip() else sel_fld_vc
                    up_f = dict(fields)
                    up_f["invoiceNumber"] = v_inv.strip()
                    up_f["vendor"] = v_vend.strip()
                    up_f["date"] = v_dt.strip()
                    up_f["dueDate"] = v_due.strip()
                    up_f["subtotal"] = v_sub
                    up_f["total"] = v_tot
                    firebase_service.update_document_fields(chosen_id, up_f, folder=final_fld, user=user)
                    st.session_state[f"vc_edit_{chosen_id}"] = False
                    st.toast(f"Invoice updated in folder '{final_fld}'!", icon="✅")
                    st.rerun()
            with sv2:
                if st.button("Cancel", key=f"vc_cancel_{chosen_id}"):
                    st.session_state[f"vc_edit_{chosen_id}"] = False
                    st.rerun()

    if st.session_state.get(f"vc_del_{chosen_id}"):
        with st.container():
            st.markdown(
                f"""
                <div style="background: #FEF2F2; border: 1.5px solid #DC2626; border-radius: 16px; padding: 18px; margin-bottom: 14px;">
                    <div style="font-weight: 700; font-size: 1rem; color: #991B1B;">🗑️ Confirm Invoice Deletion</div>
                    <div style="font-size: 0.84rem; color: #7F1D1D; margin-top: 4px;">
                        Permanently delete <b>{fields.get('invoiceNumber') or chosen_id}</b>? This will purge all verification metrics and audit history.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            del_c1, del_c2 = st.columns([1.5, 4])
            with del_c1:
                if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True, key=f"vc_confirm_del_{chosen_id}"):
                    firebase_service.delete_document(chosen_id, user=user)
                    st.session_state[f"vc_del_{chosen_id}"] = False
                    st.session_state.pop("selected_doc_id", None)
                    st.toast(f"Invoice {chosen_id} deleted.", icon="🗑️")
                    st.session_state["nav_page"] = "Document History"
                    st.rerun()
            with del_c2:
                if st.button("Cancel", key=f"vc_cancel_del_{chosen_id}"):
                    st.session_state[f"vc_del_{chosen_id}"] = False
                    st.rerun()

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
            <div style="background-color: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 14px; margin-top: 10px; font-size: 0.82rem; color: #515154; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                <div><b>Filename:</b> <span style="color: #1D1D1F;">{file_info.get('filename', 'N/A')}</span></div>
                <div><b>SHA-256:</b> <span style="font-family: monospace; color: #0071E3;">{file_info.get('sha256', 'N/A')[:20]}...</span></div>
                <div><b>pHash:</b> <span style="font-family: monospace; color: #7E22CE;">{file_info.get('pHash', 'N/A')}</span></div>
                <div><b>Created:</b> <span style="color: #1D1D1F;">{doc.get('createdAt', 'N/A')}</span></div>
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
                <div style="background-color: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 14px; margin-bottom: 12px; color: #1D1D1F; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                    <div><b>Model Verdict:</b> <span style="font-weight: 700; color: {'#DC2626' if verdict == 'suspicious' else '#16A34A'};">{verdict.upper()}</span></div>
                    <div><b>Suspiciousness Probability:</b> <span style="color: #0071E3; font-weight: 600;">{ml_prob * 100:.1f}%</span></div>
                    <div style="color: #6E6E73; font-size: 0.82rem; margin-top: 4px;"><b>Model Registry:</b> {ml.get('modelVersion', 'v1.0.0')} (Random Forest Baseline)</div>
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
