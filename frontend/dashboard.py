"""Dashboard Page for FinVision."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.components import render_kpi_card, render_risk_badge
from src import firebase_service


def render_dashboard(user: dict):
    """Render the executive financial document overview."""
    st.markdown("## Overview & Document Portfolio")
    st.markdown("Live verification metrics, risk classification distribution, and audit summaries.")

    # Retrieve real metrics from Firestore aggregation queries
    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    kpis = firebase_service.get_dashboard_metrics(owner_uid)

    # 5 KPI cards in columns
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Invoices", kpis["total_documents"], "Processed to date")
    with c2:
        render_kpi_card("Standard (Low)", kpis["normal_count"], "Eligible for straight-thru")
    with c3:
        render_kpi_card("Medium Variance", kpis["medium_risk_count"], "Secondary review recommended")
    with c4:
        render_kpi_card("High Risk", kpis["high_risk_count"], "Manual review required")
    with c5:
        render_kpi_card("Duplicate Matches", kpis["duplicates_count"], "Collisions identified")

    st.markdown("---")

    # Interactive Visualizations
    col_chart1, col_chart2 = st.columns([1, 1.3])

    with col_chart1:
        st.markdown("### Risk Level Distribution")
        risk_counts = {
            "Low Risk": kpis["normal_count"],
            "Medium Risk": kpis["medium_risk_count"],
            "High Risk": kpis["high_risk_count"],
        }
        df_risk = pd.DataFrame([{"Risk Level": k, "Count": v} for k, v in risk_counts.items()])
        if df_risk["Count"].sum() > 0:
            fig_pie = px.pie(
                df_risk,
                values="Count",
                names="Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "Low Risk": "#10B981",
                    "Medium Risk": "#F59E0B",
                    "High Risk": "#EF4444",
                },
                hole=0.55,
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No documents uploaded yet to display risk distribution.")

    with col_chart2:
        st.markdown("### Review Decision Breakdown")
        rev_counts = {
            "Approved": kpis["approved_count"],
            "Rejected": kpis["rejected_count"],
            "Pending Review": max(0, kpis["total_documents"] - kpis["approved_count"] - kpis["rejected_count"]),
        }
        df_rev = pd.DataFrame([{"Status": k, "Count": v} for k, v in rev_counts.items()])
        if df_rev["Count"].sum() > 0:
            fig_bar = px.bar(
                df_rev,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map={
                    "Approved": "#10B981",
                    "Rejected": "#EF4444",
                    "Pending Review": "#3B82F6",
                },
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No decision records available.")

    # Recent Documents Table
    st.markdown("### Recent Document Queue")
    recents = kpis.get("recent_documents", [])
    if recents:
        table_rows = []
        for d in recents:
            fields = d.get("fields", {})
            risk = d.get("risk", {})
            review = d.get("review") or {}
            table_rows.append({
                "Doc ID": d.get("docId", "")[:12],
                "Invoice #": fields.get("invoiceNumber") or "N/A",
                "Vendor": fields.get("vendor") or "N/A",
                "Total": f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A",
                "Risk Level": risk.get("level", "PENDING"),
                "Risk Score": f"{risk.get('score', 0):.0f}",
                "Review Decision": (review.get("decision") or "unreviewed").title(),
                "Uploaded": d.get("createdAt", "")[:10],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No documents in history. Upload your first invoice to begin.")

