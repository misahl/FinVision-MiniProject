"""Document History & Search Archive for FinVision."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from frontend.components import render_risk_badge
from src import firebase_service


def render_history(user: Dict[str, Any]):
    """Render searchable history table with filters, sorting, and pagination."""
    st.markdown("## 📜 Document History & Archive")
    st.caption("Search portfolio invoices, filter by vendor, date range, or audit outcome, and open detailed verification records.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=200)

    if not docs:
        st.info("No documents found in archive. Upload an invoice to build your document portfolio.")
        return

    # Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])

    with f_col1:
        search_query = st.text_input("Search (Invoice #, Vendor, Customer)", placeholder="Search keyword...")

    with f_col2:
        risk_filter = st.selectbox("Risk Rating", options=["ALL", "LOW", "MEDIUM", "HIGH"])

    with f_col3:
        status_filter = st.selectbox("Review Decision", options=["ALL", "approved", "rejected", "flagged", "needs_review", "unreviewed"])

    with f_col4:
        sort_order = st.selectbox("Sort by Upload Date", options=["Newest First", "Oldest First"])

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

    # Sort
    filtered.sort(
        key=lambda x: str(x.get("createdAt", "")),
        reverse=(sort_order == "Newest First"),
    )

    st.markdown(f"**Showing {len(filtered)} matching document(s):**")

    if filtered:
        # Pagination setup
        page_size = 15
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        curr_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            format_func=lambda p: f"Page {p} of {total_pages}",
        )
        start_idx = (curr_page - 1) * page_size
        paged_docs = filtered[start_idx:start_idx + page_size]

        rows = []
        for d in paged_docs:
            fields = d.get("fields", {})
            risk = d.get("risk", {})
            review = d.get("review") or {}
            rows.append({
                "Doc ID": d.get("docId", "")[:12],
                "Invoice #": fields.get("invoiceNumber") or "N/A",
                "Vendor": fields.get("vendor") or "N/A",
                "Date": fields.get("date") or "N/A",
                "Total Amount": f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A",
                "Risk Rating": risk.get("level", "LOW"),
                "Risk Score": f"{risk.get('score', 0):.0f}",
                "Review Decision": (review.get("decision") or "unreviewed").title(),
                "Upload Timestamp": str(d.get("createdAt", ""))[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Open Document in Verification Workspace")
        doc_select_map = {
            d["docId"]: f"📄 {d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} | {d.get('fields', {}).get('vendor') or 'Unknown'} (${float(d.get('fields', {}).get('total', 0.0) or 0.0):,.2f})"
            for d in paged_docs
        }
        chosen_view_id = st.selectbox(
            "Select Document to Open",
            options=list(doc_select_map.keys()),
            format_func=lambda x: doc_select_map[x],
            key="history_open_doc",
        )
        if st.button("🔍 Open in Verification Workspace", type="primary"):
            st.session_state["selected_doc_id"] = chosen_view_id
            st.session_state["nav_page"] = "Verification Results"
            st.rerun()
