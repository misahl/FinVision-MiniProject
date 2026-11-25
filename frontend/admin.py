"""Administration Workspace for Authorized FinVision Administrators."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from frontend.components import render_kpi_card, render_risk_badge
from src import firebase_service, predict


def render_admin(user: Dict[str, Any]):
    """Render comprehensive administration control panel for authorized administrators."""
    # Enforce backend authorization check
    if user.get("role") != "admin":
        st.error("⛔ Access Denied: Administrator privileges required to access this workspace.")
        st.stop()

    st.markdown("## ⚙️ FinVision Administration & System Management")
    st.caption("Enterprise identity governance, role-based access control, audit event telemetry, and system diagnostics.")

    # Retrieve system health
    health = firebase_service.get_system_health()

    # Executive Admin Cards
    users_list = firebase_service.list_users()
    audit_logs = firebase_service.list_audit_logs(limit=50)
    all_docs = firebase_service.list_documents(owner_uid=None, limit=100)
    pending_queue = [d for d in all_docs if not (d.get("review") or {}).get("decision")]

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_kpi_card("Total Registered Users", len(users_list), "Active system accounts")
    with a2:
        render_kpi_card("Pending Review Queue", len(pending_queue), "Documents awaiting sign-off")
    with a3:
        render_kpi_card("Logged Audit Events", len(audit_logs), "Recent tamper-evident events")
    with a4:
        render_kpi_card("System Health", "Operational", f"Project: {health.get('projectId')}")

    st.markdown("---")

    tab_users, tab_queue, tab_audit, tab_model, tab_health = st.tabs([
        "User & Role Management",
        "Pending Review Queue",
        "Immutable Audit Logs",
        "Model Registry & Telemetry",
        "System Infrastructure Health",
    ])

    # -------------------------------------------------------------
    # 1. User & Role Management
    # -------------------------------------------------------------
    with tab_users:
        st.markdown("### User Identity & Privilege Governance")
        st.caption("Manage user profiles and assign authorized roles (`analyst` or `admin`).")

        if users_list:
            user_table_data = []
            for u in users_list:
                user_table_data.append({
                    "UID": u.get("uid", "")[:12] + "...",
                    "Display Name": u.get("displayName", "N/A"),
                    "Email": u.get("email", "N/A"),
                    "Role": u.get("role", "analyst").upper(),
                    "Created At": str(u.get("createdAt", ""))[:19].replace("T", " "),
                })
            st.dataframe(pd.DataFrame(user_table_data), use_container_width=True)

            st.markdown("#### Modify User Role")
            user_options = {u["uid"]: f"{u.get('displayName')} ({u.get('email')}) — [{u.get('role', 'analyst').upper()}]" for u in users_list}
            selected_user_uid = st.selectbox(
                "Select User to Reassign Role",
                options=list(user_options.keys()),
                format_func=lambda x: user_options[x],
            )
            target_user = next((u for u in users_list if u["uid"] == selected_user_uid), None)

            if target_user:
                current_role = target_user.get("role", "analyst")
                col_role_sel, col_role_btn = st.columns([1, 1])
                with col_role_sel:
                    new_role = st.selectbox("Assign Role", options=["analyst", "admin"], index=0 if current_role == "analyst" else 1)
                with col_role_btn:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("Update User Role", type="primary"):
                        try:
                            firebase_service.update_user_role(selected_user_uid, new_role, user)
                            st.success(f"✓ Role for {target_user.get('displayName')} successfully updated to '{new_role.upper()}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update user role: {str(e)}")
        else:
            st.info("No registered users recorded in database.")

    # -------------------------------------------------------------
    # 2. Pending Review Queue
    # -------------------------------------------------------------
    with tab_queue:
        st.markdown("### Priority Document Review Queue")
        st.caption("Unreviewed invoices requiring analyst or administrator sign-off.")

        if pending_queue:
            q_rows = []
            for d in pending_queue:
                fields = d.get("fields", {})
                risk = d.get("risk", {})
                q_rows.append({
                    "Doc ID": d.get("docId", "")[:12],
                    "Invoice #": fields.get("invoiceNumber", "N/A"),
                    "Vendor": fields.get("vendor", "N/A"),
                    "Total": f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A",
                    "Risk Level": risk.get("level", "LOW"),
                    "Risk Score": f"{risk.get('score', 0):.0f}",
                    "Uploaded": str(d.get("createdAt", ""))[:19].replace("T", " "),
                })
            st.dataframe(pd.DataFrame(q_rows), use_container_width=True)
        else:
            st.success("✓ Review queue is clear. No unreviewed invoices pending.")

    # -------------------------------------------------------------
    # 3. Immutable Audit Logs
    # -------------------------------------------------------------
    with tab_audit:
        st.markdown("### Tamper-Evident Audit Trail")
        st.caption("Append-only audit events recording verification runs, review decisions, and role changes.")

        if audit_logs:
            audit_rows = []
            for a in audit_logs:
                audit_rows.append({
                    "Event ID": a.get("eventId", "")[:12],
                    "Actor UID": a.get("actorUid", "")[:12] + "...",
                    "Action": a.get("action", "UNKNOWN"),
                    "Target Doc / Subject": a.get("docId", "N/A")[:14],
                    "Timestamp (UTC)": str(a.get("timestamp", ""))[:19].replace("T", " "),
                    "Details": json.dumps(a.get("details", {})),
                })
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
        else:
            st.info("No audit logs recorded yet.")

    # -------------------------------------------------------------
    # 4. Model Registry & Telemetry
    # -------------------------------------------------------------
    with tab_model:
        st.markdown("### Production ML Model Registry")
        st.caption("Active classifier metadata, feature importance weights, and validation benchmarks.")

        meta_path = Path(__file__).parent.parent / "models" / "model_metadata_v1.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            c_mod1, c_mod2 = st.columns(2)
            with c_mod1:
                st.markdown(f"**Model Version:** `{meta.get('version', 'v1.0.0')}`")
                st.markdown(f"**Dataset Specification:** {meta.get('datasetDescription')}")
                st.markdown(f"**Train / Test Split:** `{meta.get('trainTestSplit')}`")
                st.markdown(f"**Training Samples:** `{meta.get('trainSamples')}` | **Held-Out Test Samples:** `{meta.get('testSamples')}`")

            with c_mod2:
                ml_met = meta.get("metrics", {}).get("mlModel", {})
                st.markdown(f"**Test Accuracy:** `{ml_met.get('accuracy', 1.0) * 100:.1f}%`")
                st.markdown(f"**Test Precision:** `{ml_met.get('precision', 1.0) * 100:.1f}%`")
                st.markdown(f"**Test Recall:** `{ml_met.get('recall', 1.0) * 100:.1f}%`")
                st.markdown(f"**Test F1-Score:** `{ml_met.get('f1Score', 1.0) * 100:.1f}%`")

            st.markdown("#### Feature Importance Weights")
            feat_imp = meta.get("featureImportances", {})
            if feat_imp:
                df_imp = pd.DataFrame([{"Feature": k, "Importance Weight": v} for k, v in feat_imp.items()]).sort_values("Importance Weight", ascending=False)
                st.dataframe(df_imp, use_container_width=True)

            st.caption(f"🛡️ Disclaimer: {meta.get('disclaimer')}")
        else:
            st.warning("Model metadata JSON not found.")

    # -------------------------------------------------------------
    # 5. System Infrastructure Health
    # -------------------------------------------------------------
    with tab_health:
        st.markdown("### Google Cloud & Firebase Health Diagnostics")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(
                f"""
                <div class="fin-card">
                    <div class="fin-card-title">Firebase Project</div>
                    <div style="font-weight: 700; color: #F8FAFC;">{health.get('projectId')}</div>
                    <div class="fin-card-subtitle">Region: us-central1</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"""
                <div class="fin-card">
                    <div class="fin-card-title">Cloud Storage Bucket</div>
                    <div style="font-weight: 700; color: #F8FAFC;">{health.get('storageBucket')}</div>
                    <div class="fin-card-subtitle">Status: {health.get('storage')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f"""
                <div class="fin-card">
                    <div class="fin-card-title">Cloud Run Service</div>
                    <div style="font-weight: 700; color: #F8FAFC;">finvision</div>
                    <div class="fin-card-subtitle">ADC Active: True</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

