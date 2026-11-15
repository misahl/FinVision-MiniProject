"""Executive portfolio dashboard for FinVision."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import render_kpi_card
from src import firebase_service


def render_dashboard(user: dict):
    """Render a calm, enterprise-style document portfolio overview."""
    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    kpis = firebase_service.get_dashboard_metrics(owner_uid)
    total = int(kpis.get("total_documents", 0) or 0)
    low = int(kpis.get("normal_count", 0) or 0)
    medium = int(kpis.get("medium_risk_count", 0) or 0)
    high = int(kpis.get("high_risk_count", 0) or 0)
    duplicates = int(kpis.get("duplicates_count", 0) or 0)

    st.markdown(
        """
        <div class="fv-page-heading">
          <div class="fv-eyebrow">FINANCIAL OPERATIONS / OVERVIEW</div>
          <h1>Document portfolio</h1>
          <p>Monitor invoice processing, review exceptions and keep an auditable view of your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="fv-welcome">
          <div>
            <div class="fv-welcome-kicker">WORKSPACE OVERVIEW</div>
            <div class="fv-welcome-title">Welcome back, {user.get('displayName', 'Analyst')}</div>
            <div class="fv-welcome-copy">Your verification activity and review queue, in one place.</div>
          </div>
          <div class="fv-welcome-meta"><span class="fv-live-dot"></span> Workspace data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Portfolio at a glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Documents processed", total, "All uploaded invoices")
    with c2:
        render_kpi_card("Low risk", low, "No major indicators")
    with c3:
        render_kpi_card("Needs attention", medium, "Review recommended")
    with c4:
        render_kpi_card("High risk", high, "Prioritise manual review")
    with c5:
        render_kpi_card("Possible duplicates", duplicates, "Potential matches")

    st.markdown("---")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Risk distribution")
        if total:
            risk_df = pd.DataFrame(
                [
                    {"Risk level": "Low", "Documents": low},
                    {"Risk level": "Medium", "Documents": medium},
                    {"Risk level": "High", "Documents": high},
                ]
            )
            risk_df = risk_df[risk_df["Documents"] > 0]
            fig = px.pie(
                risk_df,
                values="Documents",
                names="Risk level",
                hole=0.68,
                color="Risk level",
                color_discrete_map={"Low": "#16A37A", "Medium": "#D99A26", "High": "#D9534F"},
            )
            fig.update_layout(
                height=280,
                margin=dict(t=8, b=8, l=8, r=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
                font=dict(color="#344054"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="fv-empty-panel"><div class="fv-empty-icon">◌</div>'
                '<strong>Your portfolio starts here</strong><p>Risk insights will appear after your first invoice is processed.</p></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Review outcomes")
        approved = int(kpis.get("approved_count", 0) or 0)
        rejected = int(kpis.get("rejected_count", 0) or 0)
        pending = max(0, total - approved - rejected)
        if total:
            review_df = pd.DataFrame(
                [
                    {"Decision": "Approved", "Documents": approved},
                    {"Decision": "Rejected", "Documents": rejected},
                    {"Decision": "Pending review", "Documents": pending},
                ]
            )
            fig = px.bar(
                review_df,
                x="Decision",
                y="Documents",
                color="Decision",
                color_discrete_map={
                    "Approved": "#16A37A",
                    "Rejected": "#D9534F",
                    "Pending review": "#5277B8",
                },
                text="Documents",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=280,
                margin=dict(t=18, b=8, l=8, r=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
                font=dict(color="#344054"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="fv-empty-panel"><div class="fv-empty-icon">✓</div>'
                '<strong>Review activity will show here</strong><p>Approved, rejected and pending documents will be summarised here.</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Recent document activity")
    recents = kpis.get("recent_documents", [])
    if recents:
        rows = []
        for doc in recents:
            fields = doc.get("fields", {})
            risk = doc.get("risk", {})
            review = doc.get("review") or {}
            rows.append(
                {
                    "Invoice": fields.get("invoiceNumber") or "Not extracted",
                    "Vendor": fields.get("vendor") or "Not extracted",
                    "Amount": f"{fields.get('currency', 'USD')} {float(fields.get('total') or 0):,.2f}",
                    "Risk": str(risk.get("level", "PENDING")).title(),
                    "Review": str(review.get("decision") or "Pending").replace("_", " ").title(),
                    "Date added": str(doc.get("createdAt", ""))[:10] or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="fv-first-run">
              <div class="fv-first-run-mark">＋</div>
              <div>
                <strong>No documents have been processed yet</strong>
                <p>Upload an invoice to extract its financial details, run verification checks and create your first audit record.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Upload your first invoice", type="primary", use_container_width=False):
            st.session_state["nav_page"] = "Upload Invoice"
            st.rerun()

    st.caption("Risk indicators are decision-support signals. They are not, by themselves, proof of fraud or document authenticity.")
