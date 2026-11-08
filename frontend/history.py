"""Document History & Audit Trail Search for FinVision."""

import streamlit as st
import pandas as pd

from frontend.components import render_risk_badge
from src import firebase_service


def render_history(user: dict):
    """Render searchable history table with filters and cursor pagination."""
    st.markdown("## Document Archive & Portfolio History")
    st.markdown("Search across verified invoices, filter by vendor, risk rating, or review decision.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    if not docs:
        st.info("No documents found in archive.")
        return

    # Filter Controls
    col_search, col_risk_filt, col_status_filt = st.columns([1.5, 1, 1])

    with col_search:
        search_query = st.text_input("Search (Invoice #, Vendor, Customer)", placeholder="Search keyword...")

    with col_risk_filt:
        risk_filter = st.selectbox("Risk Level", options=["ALL", "LOW", "MEDIUM", "HIGH"])

    with col_status_filt:
        status_filter = st.selectbox("Decision Status", options=["ALL", "approved", "rejected", "needs_review", "unreviewed"])

    # Filter records in memory
    filtered = []
    for d in docs:
        fields = d.get("fields", {})
        risk = d.get("risk", {})
        review = d.get("review") or {}
        decision = review.get("decision") or "unreviewed"
        risk_lvl = risk.get("level", "LOW")

        # Text search matching
        match_search = True
        if search_query:
            sq = search_query.lower()
            inv_num = str(fields.get("invoiceNumber", "")).lower()
            vend = str(fields.get("vendor", "")).lower()
            cust = str(fields.get("customer", "")).lower()
            match_search = sq in inv_num or sq in vend or sq in cust

        # Risk match
        match_risk = risk_filter == "ALL" or risk_lvl == risk_filter

        # Status match
        match_status = status_filter == "ALL" or decision == status_filter

        if match_search and match_risk and match_status:
            filtered.append(d)

    st.markdown(f"**Showing {len(filtered)} matching document(s):**")

    if filtered:
        rows = []
        for d in filtered:
            fields = d.get("fields", {})
            risk = d.get("risk", {})
            review = d.get("review") or {}
            rows.append({
                "Doc ID": d.get("docId", "")[:12],
                "Invoice #": fields.get("invoiceNumber") or "N/A",
                "Vendor": fields.get("vendor") or "N/A",
                "Date": fields.get("date") or "N/A",
                "Total Amount": f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A",
                "Risk Level": risk.get("level", "LOW"),
                "Risk Score": f"{risk.get('score', 0):.0f}",
                "Review Decision": (review.get("decision") or "unreviewed").title(),
                "Created At": d.get("createdAt", "")[:10],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### Quick Navigate to Document Audit")
        selected_to_view = st.selectbox(
            "Select document to view in Verify Workspace",
            options=[d["docId"] for d in filtered],
            format_func=lambda did: f"{did[:12]} - {next((d.get('fields', {}).get('vendor') or 'Unknown') for d in filtered if d['docId'] == did)}",
        )
        if st.button("Open in Verify Workspace"):
            st.session_state["selected_doc_id"] = selected_to_view
            st.session_state["nav_page"] = "Verify Documents"
            st.rerun()
    else:
        st.warning("No documents matched the applied filter criteria.")

