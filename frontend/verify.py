"""Document Verification & Review Workspace for FinVision."""

from decimal import Decimal
import streamlit as st
from PIL import Image

from frontend.components import render_risk_badge, render_rule_card
from src import firebase_service, verification, risk_assessment


def render_verify(user: dict):
    """Render interactive document verification, editable field audit, and human decision workspace."""
    st.markdown("## Document Audit & Verification Workspace")
    st.caption("Inspect document scans, review extracted financial fields, inspect deterministic rule outcomes, and submit formal reviewer sign-off.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=50)

    if not docs:
        st.info("No documents available for verification. Please upload an invoice first.")
        return

    # Select document to verify
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
    file_info = doc.get("file", {})
    review = doc.get("review") or {}

    # 1. Top Executive Status Bar (Card Grid)
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
                <div style="font-weight: 700; font-size: 1.15rem; color: #0F172A;">{doc.get('status', 'verified').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b3:
        dec = review.get("decision") or "unreviewed"
        dec_color = "#10B981" if dec == "approved" else ("#EF4444" if dec == "rejected" else "#F59E0B")
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
                <div style="font-weight: 700; font-size: 1.15rem; color: #0F172A;">{doc.get('overallConfidence', 0):.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Main Work Area: Scan Preview (Left) vs Editable Form (Right)
    col_left, col_right = st.columns([1, 1.25])

    with col_left:
        st.markdown("### Document Scan")
        storage_path = file_info.get("storagePath")
        scan_rendered = False

        if storage_path:
            img_bytes = firebase_service.download_invoice_bytes(storage_path)
            if img_bytes:
                try:
                    from src.image_processing import load_image_from_source
                    img_bgr, _ = load_image_from_source(img_bytes)
                    st.image(img_bgr, caption=f"Original File: {file_info.get('mimeType', 'Document')}", use_container_width=True)
                    scan_rendered = True
                except Exception:
                    pass

        if not scan_rendered:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 40px 20px; text-align: center; color: #64748B;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📑</div>
                    <div style="font-weight: 600; color: #1E293B; margin-bottom: 4px;">Original Scan Asset</div>
                    <div style="font-size: 0.85rem;">Document record verified via metadata & OCR pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Cryptographic file fingerprint card
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="fin-card" style="padding: 12px 16px;">
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-bottom: 4px;">DIGITAL FINGERPRINT</div>
                <div style="font-family: monospace; font-size: 0.75rem; color: #334155; word-break: break-all;">
                    <b>SHA-256:</b> {file_info.get('sha256', 'N/A')}<br/>
                    <b>pHash:</b> {file_info.get('pHash', 'N/A')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Extracted Financial Fields")
        st.caption("Review extracted values. Fields with confidence < 80% indicate potential visual or OCR ambiguity.")

        with st.form("edit_fields_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                conf_inv = confidences.get("invoiceNumber", 0.0)
                inv_num_val = st.text_input(
                    f"Invoice Number (Conf: {conf_inv:.0f}%)",
                    value=fields.get("invoiceNumber") or "",
                )
                conf_date = confidences.get("date", 0.0)
                date_val = st.text_input(
                    f"Invoice Date (Conf: {conf_date:.0f}%)",
                    value=fields.get("date") or "",
                )
                conf_sub = confidences.get("subtotal", 0.0)
                subtotal_val = st.number_input(
                    f"Subtotal $ (Conf: {conf_sub:.0f}%)",
                    value=float(fields.get("subtotal") or 0.0),
                    step=0.01,
                )

            with col_f2:
                conf_vend = confidences.get("vendor", 0.0)
                vendor_val = st.text_input(
                    f"Vendor (Conf: {conf_vend:.0f}%)",
                    value=fields.get("vendor") or "",
                )
                conf_cust = confidences.get("customer", 0.0)
                cust_val = st.text_input(
                    f"Customer (Conf: {conf_cust:.0f}%)",
                    value=fields.get("customer") or "",
                )
                conf_tax = confidences.get("tax", 0.0)
                tax_val = st.number_input(
                    f"Tax Amount $ (Conf: {conf_tax:.0f}%)",
                    value=float(fields.get("tax") or 0.0),
                    step=0.01,
                )

            conf_tot = confidences.get("total", 0.0)
            total_val = st.number_input(
                f"Total Stated Amount $ (Conf: {conf_tot:.0f}%)",
                value=float(fields.get("total") or 0.0),
                step=0.01,
            )

            save_corrections = st.form_submit_button("💾 Save Field Corrections & Re-verify", type="primary", use_container_width=True)
            if save_corrections:
                updated_fields = {
                    "invoiceNumber": inv_num_val,
                    "date": date_val,
                    "vendor": vendor_val,
                    "customer": cust_val,
                    "subtotal": subtotal_val,
                    "tax": tax_val,
                    "taxRate": round(tax_val / subtotal_val, 2) if subtotal_val > 0 else 0.10,
                    "total": total_val,
                    "currency": fields.get("currency", "USD"),
                }

                # Re-run rule engine and risk assessment on corrected values
                new_verif = verification.run_verification_rules(updated_fields, doc.get("lineItems", []))
                new_risk = risk_assessment.compute_document_risk(new_verif, ml_results=ml)

                # Persist updates
                firebase_service.save_document_analysis(
                    chosen_id,
                    updated_fields,
                    confidences,
                    doc.get("lineItems", []),
                    new_verif,
                    {"cvScore": verif.get("cvScore", 0.0), "cvRegions": verif.get("cvRegions", [])},
                    ml,
                    new_risk,
                )
                st.success("Field corrections persisted! Verification rules and risk assessment updated.")
                st.rerun()

    # 3. Diagnostic Tabs
    st.markdown("---")
    tab_rules, tab_cv, tab_ml, tab_risk, tab_decision = st.tabs([
        "Deterministic Rules",
        "Computer Vision Diagnostics",
        "Machine Learning Inference",
        "Risk Drivers & Actions",
        "Reviewer Decision Sign-Off",
    ])

    with tab_rules:
        st.markdown("### Deterministic Rule Engine Verification")
        st.caption("Pure mathematical & business logic checks evaluated with exact Decimal arithmetic.")
        rule_list = verif.get("ruleResults", [])
        if rule_list:
            for r in rule_list:
                render_rule_card(r)
        else:
            st.info("No rule results recorded for this document.")

    with tab_cv:
        st.markdown("### Explainable Computer Vision Indicators")
        st.caption("Heuristic image forensics. These indicators are advisory checks and do not prove tampering.")
        cv_score = verif.get("cvScore", 0.0)
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.metric("Composite CV Anomaly Score", f"{cv_score:.1f} / 100")
        with c_v2:
            regions = verif.get("cvRegions", [])
            st.metric("Flagged Outlier Clusters", f"{len(regions)} region(s)")

    with tab_ml:
        st.markdown("### Statistical Suspiciousness Classifier")
        st.caption("Random Forest model trained on documented, reproducible synthetic perturbations.")
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.metric("Model Version", ml.get("modelVersion", "v1.0.0"))
        with c_m2:
            st.metric("Model Status", "Experimental (Advisory)")
        with c_m3:
            st.metric("Probability of Suspiciousness", f"{ml.get('probability', 0.0)*100:.1f}%")

    with tab_risk:
        st.markdown("### Risk Evaluation Summary")
        st.markdown(
            f"""
            <div class="fin-card">
                <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A; margin-bottom: 6px;">
                    Overall Risk Rating: {render_risk_badge(risk.get('level', 'LOW'), risk.get('score', 0))}
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">
                    <b>Recommended Operational Action:</b><br/>{risk.get('recommendedAction', 'Standard manual review required.')}
                </div>
                <div style="font-weight: 600; font-size: 0.88rem; color: #1E293B; margin-bottom: 4px;">Identified Variance Drivers:</div>
                <ul style="color: #475569; font-size: 0.88rem; margin-top: 0;">
                    {''.join(f'<li>{reason}</li>' for reason in risk.get('reasons', []))}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_decision:
        st.markdown("### Human Reviewer Audit Sign-Off")
        st.caption("Formal auditor sign-off. The decision is committed with server timestamp to an append-only audit trail.")

        current_comment = review.get("comment", "")
        reviewer_comment = st.text_area("Audit Justification / Reviewer Notes", value=current_comment, placeholder="Enter any notes or justification for this decision...")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            if st.button("✅ Approve Document", type="primary", use_container_width=True):
                firebase_service.update_document_review_decision(
                    chosen_id, user.get("uid", "anonymous"), "approved", reviewer_comment
                )
                st.success("Document approved and audit trail updated.")
                st.rerun()
        with col_d2:
            if st.button("❌ Reject Document", use_container_width=True):
                firebase_service.update_document_review_decision(
                    chosen_id, user.get("uid", "anonymous"), "rejected", reviewer_comment
                )
                st.error("Document rejected and audit trail updated.")
                st.rerun()
        with col_d3:
            if st.button("⚠️ Flag for Secondary Review", use_container_width=True):
                firebase_service.update_document_review_decision(
                    chosen_id, user.get("uid", "anonymous"), "needs_review", reviewer_comment
                )
                st.warning("Document flagged for secondary review.")
                st.rerun()
