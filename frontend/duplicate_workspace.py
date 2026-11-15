"""Dedicated Multi-Modal Duplicate Detection Workspace for FinVision."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from frontend.components import render_risk_badge
from src import duplicate_detection, firebase_service


def render_duplicate_workspace(user: dict):
    """Render comprehensive duplicate invoice identification and comparison workspace."""
    st.markdown("## 🔍 Multi-Modal Duplicate Detection Workspace")
    st.caption(
        "Detect exact cryptographic replicas (SHA-256), perceptual image layout collisions (pHash), "
        "and semantic business duplicates (invoice number, vendor, date, and amount matching)."
    )

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    if not docs:
        st.info("No documents found in archive to evaluate duplicates.")
        return

    # Find all documents with identified duplicates
    duplicate_docs = []
    for d in docs:
        dup_matches = d.get("verification", {}).get("duplicateMatches", [])
        if dup_matches:
            duplicate_docs.append((d, dup_matches))

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Portfolio Documents</div>
                <div class="fin-card-value">{len(docs)}</div>
                <div class="fin-card-subtitle">Active repository records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Duplicate Collisions</div>
                <div class="fin-card-value" style="color: {'#EF4444' if duplicate_docs else '#10B981'};">{len(duplicate_docs)}</div>
                <div class="fin-card-subtitle">Invoices with matching candidates</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Perceptual Threshold</div>
                <div class="fin-card-value">≤ {duplicate_detection.HAMMING_THRESHOLD}</div>
                <div class="fin-card-subtitle">64-bit DCT Hamming distance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Matching Modalities</div>
                <div class="fin-card-value">4</div>
                <div class="fin-card-subtitle">SHA256, pHash, Semantic (2)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Document Selection for Duplicate Audit
    st.markdown("### Inspect Duplicate Collisions")
    tab_detected, tab_cross_scan = st.tabs(["Identified Collisions Queue", "On-Demand Cross-Scan"])

    with tab_detected:
        if not duplicate_docs:
            st.success("✓ No duplicate collisions detected across the current document archive.")
        else:
            doc_options = {
                d["docId"]: f"📄 {d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} | {d.get('fields', {}).get('vendor') or 'Unknown'} — {len(matches)} collision(s)"
                for d, matches in duplicate_docs
            }
            selected_dup_id = st.selectbox(
                "Select Flagged Document to Inspect",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
            )

            # Retrieve target document
            target = next((d for d, _ in duplicate_docs if d["docId"] == selected_dup_id), None)
            if target:
                t_fields = target.get("fields", {})
                t_matches = target.get("verification", {}).get("duplicateMatches", [])

                st.markdown(f"#### Target Document: `{target.get('docId')}`")
                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    st.text_input("Invoice #", value=t_fields.get("invoiceNumber", "N/A"), disabled=True, key="dup_t_inv")
                with tc2:
                    st.text_input("Vendor", value=t_fields.get("vendor", "N/A"), disabled=True, key="dup_t_vend")
                with tc3:
                    st.text_input("Date", value=t_fields.get("date", "N/A"), disabled=True, key="dup_t_date")
                with tc4:
                    tot_val = t_fields.get("total")
                    st.text_input("Total Amount", value=f"${float(tot_val):,.2f}" if tot_val else "N/A", disabled=True, key="dup_t_tot")

                st.markdown("##### Detected Candidate Matches")
                for i, match in enumerate(t_matches):
                    cand_id = match.get("docId", "unknown")
                    m_type = match.get("matchType", "UNKNOWN")
                    sim_score = match.get("similarityScore", 100.0)
                    reason = match.get("reason", "Potential duplicate document match.")
                    fields_matched = match.get("matchingFields", [])
                    orig_date = match.get("originalUploadDate", "Unknown")

                    with st.expander(f"Match #{i+1}: {m_type} (Similarity: {sim_score}%) — Candidate ID: {cand_id}", expanded=True):
                        col_m1, col_m2 = st.columns([1, 1])
                        with col_m1:
                            st.markdown(f"**Match Modality:** `{m_type}`")
                            st.markdown(f"**Similarity Score:** `{sim_score}%`")
                            st.markdown(f"**Matching Fields:** `{', '.join(fields_matched) if fields_matched else 'Hash'}`")
                            st.markdown(f"**Original Timestamp:** `{orig_date}`")
                        with col_m2:
                            st.markdown(f"**Detection Evidence:**\n{reason}")

    with tab_cross_scan:
        st.markdown("#### Run Real-Time Duplicate Evaluation on Any Document")
        all_doc_options = {
            d["docId"]: f"📄 {d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} | {d.get('fields', {}).get('vendor') or 'Unknown'} (${float(d.get('fields', {}).get('total', 0.0) or 0.0):,.2f})"
            for d in docs
        }
        test_doc_id = st.selectbox(
            "Select Target Document to Scan Against Portfolio",
            options=list(all_doc_options.keys()),
            format_func=lambda x: all_doc_options[x],
            key="cross_scan_select",
        )

        if st.button("🚀 Run Comprehensive Cross-Scan", type="primary"):
            target_cand = next((d for d in docs if d["docId"] == test_doc_id), None)
            if target_cand:
                t_payload = {
                    "docId": target_cand.get("docId"),
                    "sha256": target_cand.get("file", {}).get("sha256"),
                    "pHash": target_cand.get("file", {}).get("pHash"),
                    "fields": target_cand.get("fields", {}),
                }
                scan_res = duplicate_detection.find_duplicates(t_payload, docs)
                if scan_res.get("hasDuplicate"):
                    st.error(f"⚠️ Duplicate Detected! Found {scan_res.get('matchCount')} collision(s):")
                    for m in scan_res.get("duplicateMatches", []):
                        st.markdown(f"- **{m.get('matchType')}** (Similarity: `{m.get('similarityScore', 100)}%`): {m.get('reason')}")
                else:
                    st.success("✓ No duplicate matches found for this document across all 4 detection modalities.")

