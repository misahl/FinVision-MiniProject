"""Executive overview for FinVision."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import render_kpi_card
from src import firebase_service


def render_dashboard(user: dict):
    """Render an actionable portfolio dashboard using persisted metrics."""
    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    kpis = firebase_service.get_dashboard_metrics(owner_uid)
    total = int(kpis.get("total_documents", 0))
    low = int(kpis.get("normal_count", 0))
    medium = int(kpis.get("medium_risk_count", 0))
    high = int(kpis.get("high_risk_count", 0))
    duplicates = int(kpis.get("duplicates_count", 0))
    reviewed = int(kpis.get("approved_count", 0)) + int(kpis.get("rejected_count", 0))

    st.markdown(
        "<div class='fv-eyebrow'>FINANCIAL OPERATIONS / OVERVIEW</div>"
        "<div class='fv-header'><div><h1>Document intelligence</h1>"
        "<p>Monitor invoice integrity, exceptions and review activity in one workspace.</p>"
        "</div><div class='fv-live'><span></span> Workspace</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='fv-banner'><div><strong>Know what changed. Know what needs attention.</strong>"
        "<p>FinVision combines extracted invoice data, deterministic checks, duplicate signals and model-assisted risk indicators.</p></div></div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    cards = [
        ("DOCUMENTS PROCESSED", total, "Invoices in your workspace"),
        ("LOW RISK", low, "No elevated signals detected"),
        ("REVIEW REQUIRED", medium + high, "Medium or high risk indicators"),
        ("DUPLICATE SIGNALS", duplicates, "Potential matches identified"),
    ]
    for column, (label, value, caption) in zip(columns, cards):
        with column:
            render_kpi_card(label, value, caption)

    st.markdown("<div class='fv-section-head'><div><h2>Portfolio activity</h2><p>Based on saved verification records</p></div></div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.4], gap="large")
    with left:
        st.markdown("#### Risk indicators")
        if total:
            risk_df = pd.DataFrame({"Risk level": ["Low", "Medium", "High"], "Documents": [low, medium, high]})
            risk_df = risk_df[risk_df["Documents"] > 0]
            fig = px.pie(risk_df, names="Risk level", values="Documents", hole=0.72,
                         color="Risk level", color_discrete_map={"Low": "#0F9F75", "Medium": "#D99A25", "High": "#D9534F"})
            fig.update_layout(height=270, margin=dict(l=8, r=8, t=8, b=8), showlegend=True,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("<div class='fv-empty'><div class='fv-empty-icon'>⌁</div><strong>Your portfolio starts here</strong><p>Upload an invoice to see its risk indicators and verification status appear in this space.</p></div>", unsafe_allow_html=True)
            if st.button("＋ Upload your first invoice", type="primary", use_container_width=True):
                st.session_state["nav_page"] = "Upload Invoice"
                st.rerun()

    with right:
        st.markdown("#### Review throughput")
        pending = max(0, total - reviewed)
        if total:
            review_df = pd.DataFrame({"Status": ["Reviewed", "Awaiting review"], "Documents": [reviewed, pending]})
            fig = px.bar(review_df, x="Status", y="Documents", color="Status",
                         color_discrete_map={"Reviewed": "#244B78", "Awaiting review": "#D7E3F0"}, text="Documents")
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(height=270, margin=dict(l=8, r=8, t=16, b=8), showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(showgrid=True, gridcolor="#E8EDF3", zeroline=False), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("<div class='fv-empty'><div class='fv-empty-icon'>✓</div><strong>Review activity will appear here</strong><p>Once documents are processed, this chart tracks reviewed and pending cases.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='fv-section-head'><div><h2>Recent documents</h2><p>Your latest verification records</p></div></div>", unsafe_allow_html=True)
    recents = kpis.get("recent_documents", [])
    if recents:
        rows = []
        for document in recents:
            fields = document.get("fields", {}) or {}
            risk = document.get("risk", {}) or {}
            review = document.get("review") or {}
            amount = fields.get("total")
            rows.append({
                "Invoice": fields.get("invoiceNumber") or document.get("docId", "")[:12],
                "Vendor": fields.get("vendor") or "Not extracted",
                "Amount": f"{fields.get('currency', 'USD')} {float(amount):,.2f}" if amount is not None else "—",
                "Risk": (risk.get("level") or "Pending").title(),
                "Review": (review.get("decision") or "Awaiting review").replace("_", " ").title(),
                "Date": str(document.get("createdAt", ""))[:10] or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No documents have been processed yet. Your latest activity will appear here after the first upload.")

    st.caption("Risk indicators support prioritisation and human review. They are not a determination of fraud or financial approval.")
