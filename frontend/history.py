"""Document History, Folder Organization & Archive for FinVision."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from frontend.components import render_risk_badge
from src import firebase_service


def render_history(user: Dict[str, Any]):
    """Render searchable history table with folders, filters, editing, and deletion."""
    st.markdown("## 📜 Document History & Folder Archive")
    st.caption("Organize portfolio invoices into folders, search and filter records, edit invoice fields, or safely delete files.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=300)

    if not docs:
        st.info("No documents found in archive. Upload an invoice or take a camera scan to build your portfolio.")
        return

    # 1. Folder Management & Dynamic Category Filter Bar
    all_known_folders = ["Operations", "IT & Software", "Logistics", "Marketing", "Legal & Consulting", "Utilities"]
    for d in docs:
        fld = d.get("folder")
        if fld and fld not in all_known_folders:
            all_known_folders.append(fld)

    folder_options = ["📁 All Invoices (Default)"] + [f"📂 {f}" for f in sorted(all_known_folders)]
    selected_folder = st.radio(
        "Folder / Category Organization",
        options=folder_options,
        horizontal=True,
        key="hist_folder_filter",
        label_visibility="collapsed",
    )

    # Filter by folder if not All
    current_folder_filter = None
    if selected_folder != "📁 All Invoices (Default)":
        current_folder_filter = selected_folder.replace("📂 ", "")
        active_docs = [d for d in docs if d.get("folder", "Operations") == current_folder_filter]
    else:
        active_docs = docs

    # 2. Search & Advanced Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])

    with f_col1:
        search_query = st.text_input("Search (Invoice #, Vendor, Customer)", placeholder="Search keyword...", key="hist_search_kw")

    with f_col2:
        risk_filter = st.selectbox("Risk Rating", options=["ALL", "LOW", "MEDIUM", "HIGH"], key="hist_risk_sel")

    with f_col3:
        status_filter = st.selectbox("Review Decision", options=["ALL", "approved", "rejected", "flagged", "needs_review", "unreviewed"], key="hist_dec_sel")

    with f_col4:
        sort_order = st.selectbox("Sort by Upload Date", options=["Newest First", "Oldest First"], key="hist_sort_sel")

    # Filter records in memory
    filtered = []
    for d in active_docs:
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

    st.markdown(f"**Showing {len(filtered)} document(s) in {selected_folder}:**")

    if filtered:
        # Pagination setup
        page_size = 15
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        curr_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            format_func=lambda p: f"Page {p} of {total_pages}",
            key="hist_page_sel",
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
                "Folder": d.get("folder", "Operations"),
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

        st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 20px 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Document Actions & Management")

        doc_select_map = {
            d["docId"]: f"📄 {d.get('fields', {}).get('invoiceNumber') or 'Unnamed'} | {d.get('fields', {}).get('vendor') or 'Unknown'} (${float(d.get('fields', {}).get('total', 0.0) or 0.0):,.2f}) — [Folder: {d.get('folder', 'Operations')}]"
            for d in paged_docs
        }

        col_sel_doc, col_actions = st.columns([2.2, 1.8])
        with col_sel_doc:
            chosen_doc_id = st.selectbox(
                "Choose document to inspect, edit, or delete:",
                options=list(doc_select_map.keys()),
                format_func=lambda x: doc_select_map[x],
                key="history_target_doc",
            )

        target_doc = next((d for d in paged_docs if d["docId"] == chosen_doc_id), None)

        with col_actions:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("🔍 View", type="primary", use_container_width=True, key="btn_view_doc"):
                    st.session_state["selected_doc_id"] = chosen_doc_id
                    st.session_state["nav_page"] = "Verification Results"
                    st.rerun()
            with btn_col2:
                btn_edit = st.button("✏️ Edit", use_container_width=True, key="btn_edit_doc")
            with btn_col3:
                btn_del = st.button("🗑️ Delete", use_container_width=True, key="btn_del_doc")

        # Edit Form State Management
        if btn_edit:
            st.session_state[f"editing_doc_{chosen_doc_id}"] = not st.session_state.get(f"editing_doc_{chosen_doc_id}", False)
            st.session_state[f"deleting_doc_{chosen_doc_id}"] = False

        if btn_del:
            st.session_state[f"deleting_doc_{chosen_doc_id}"] = not st.session_state.get(f"deleting_doc_{chosen_doc_id}", False)
            st.session_state[f"editing_doc_{chosen_doc_id}"] = False

        # -------------------------------------------------------------
        # EDIT INVOICE PANEL
        # -------------------------------------------------------------
        if target_doc and st.session_state.get(f"editing_doc_{chosen_doc_id}"):
            with st.container():
                st.markdown(
                    """
                    <div style="background: #FFFFFF; border: 1.5px solid #0071E3; border-radius: 16px; padding: 18px; margin-top: 14px; box-shadow: 0 4px 20px rgba(0, 113, 227, 0.08);">
                        <div style="font-weight: 700; font-size: 1.05rem; color: #1D1D1F; margin-bottom: 4px;">✏️ Edit Invoice Document Fields & Folder</div>
                        <div style="font-size: 0.8rem; color: #515154; margin-bottom: 12px;">Updates are recorded to the tamper-evident audit log with automatic rule recalculation.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                t_fields = target_doc.get("fields", {})

                # Folder Selection / Custom Folder
                e_f1, e_f2 = st.columns(2)
                with e_f1:
                    current_doc_folder = target_doc.get("folder", "Operations")
                    folder_choices = sorted(list(set(all_known_folders + [current_doc_folder] + ["+ Create New Folder"])))
                    def_folder_idx = folder_choices.index(current_doc_folder) if current_doc_folder in folder_choices else 0
                    sel_fld = st.selectbox("Assign to Folder / Department", options=folder_choices, index=def_folder_idx, key=f"edit_fld_{chosen_doc_id}")
                with e_f2:
                    if sel_fld == "+ Create New Folder":
                        custom_folder_name = st.text_input("New Folder Name", placeholder="e.g. R&D Facilities", key=f"new_fld_{chosen_doc_id}")
                    else:
                        custom_folder_name = sel_fld

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    e_inv_num = st.text_input("Invoice Number", value=t_fields.get("invoiceNumber") or "", key=f"e_inv_{chosen_doc_id}")
                    e_vendor = st.text_input("Vendor Name", value=t_fields.get("vendor") or "", key=f"e_vend_{chosen_doc_id}")
                with ec2:
                    e_date = st.text_input("Invoice Date", value=t_fields.get("date") or "", key=f"e_dt_{chosen_doc_id}")
                    e_due = st.text_input("Due Date", value=t_fields.get("dueDate") or "", key=f"e_due_{chosen_doc_id}")
                with ec3:
                    e_subtotal = st.number_input("Subtotal ($)", value=float(t_fields.get("subtotal") or 0.0), step=10.0, format="%.2f", key=f"e_sub_{chosen_doc_id}")
                    e_tax = st.number_input("Tax / VAT ($)", value=float(t_fields.get("tax") or 0.0), step=5.0, format="%.2f", key=f"e_tax_{chosen_doc_id}")

                ec4, ec5 = st.columns(2)
                with ec4:
                    e_discount = st.number_input("Discount ($)", value=float(t_fields.get("discount") or 0.0), step=5.0, format="%.2f", key=f"e_disc_{chosen_doc_id}")
                with ec5:
                    e_total = st.number_input("Total Amount ($)", value=float(t_fields.get("total") or 0.0), step=10.0, format="%.2f", key=f"e_tot_{chosen_doc_id}")

                save_col1, save_col2 = st.columns([1, 4])
                with save_col1:
                    if st.button("💾 Save Changes", type="primary", use_container_width=True, key=f"btn_save_{chosen_doc_id}"):
                        assigned_fld = custom_folder_name.strip() if custom_folder_name else "Operations"
                        updated_f = dict(t_fields)
                        updated_f["invoiceNumber"] = e_inv_num.strip()
                        updated_f["vendor"] = e_vendor.strip()
                        updated_f["date"] = e_date.strip()
                        updated_f["dueDate"] = e_due.strip()
                        updated_f["subtotal"] = e_subtotal
                        updated_f["tax"] = e_tax
                        updated_f["discount"] = e_discount
                        updated_f["total"] = e_total
                        firebase_service.update_document_fields(chosen_doc_id, updated_f, folder=assigned_fld, user=user)
                        st.session_state[f"editing_doc_{chosen_doc_id}"] = False
                        st.toast(f"Invoice {e_inv_num} updated in '{assigned_fld}'!", icon="✅")
                        st.rerun()
                with save_col2:
                    if st.button("Cancel", use_container_width=False, key=f"btn_cancel_{chosen_doc_id}"):
                        st.session_state[f"editing_doc_{chosen_doc_id}"] = False
                        st.rerun()

        # -------------------------------------------------------------
        # DELETE INVOICE CONFIRMATION
        # -------------------------------------------------------------
        if target_doc and st.session_state.get(f"deleting_doc_{chosen_doc_id}"):
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #FEF2F2; border: 1.5px solid #DC2626; border-radius: 16px; padding: 18px; margin-top: 14px;">
                        <div style="font-weight: 700; font-size: 1.05rem; color: #991B1B; margin-bottom: 6px;">🗑️ Confirm Permanent Invoice Deletion</div>
                        <div style="font-size: 0.88rem; color: #7F1D1D; margin-bottom: 14px;">
                            Are you sure you want to permanently delete <b>{target_doc.get('fields', {}).get('invoiceNumber') or chosen_doc_id}</b> from <b>{target_doc.get('fields', {}).get('vendor') or 'Vendor'}</b>?
                            This action will erase the document scan, extracted line items, and risk telemetry.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                del_col1, del_col2 = st.columns([1.5, 4])
                with del_col1:
                    if st.button("🗑️ Yes, Permanently Delete", type="primary", use_container_width=True, key=f"btn_confirm_del_{chosen_doc_id}"):
                        firebase_service.delete_document(chosen_doc_id, user=user)
                        st.session_state[f"deleting_doc_{chosen_doc_id}"] = False
                        st.toast(f"Document {chosen_doc_id} permanently deleted.", icon="🗑️")
                        st.rerun()
                with del_col2:
                    if st.button("Cancel Deletion", use_container_width=False, key=f"btn_cancel_del_{chosen_doc_id}"):
                        st.session_state[f"deleting_doc_{chosen_doc_id}"] = False
                        st.rerun()
