"""Executive Overview & Document Portfolio Dashboard for FinVision."""

from datetime import datetime
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components import render_kpi_card, render_risk_badge
from src import firebase_service


def render_dashboard(user: Dict[str, Any]):
    """Render the executive financial document overview."""
    st.markdown("## 📊 Executive Overview & Portfolio Health")
    st.caption("Real-time portfolio intelligence, arithmetic verification rates, anomaly distribution, and audit summaries.")

    # Quick Launchpad
    col_act1, col_act2, col_act3, col_act4 = st.columns(4)
    with col_act1:
        if st.button("⚡ Ingest & Verify Document", use_container_width=True, type="primary"):
            st.session_state["nav_page"] = "Upload & Analyze"
            st.rerun()
    with col_act2:
        if st.button("🔍 Review Flagged Queue", use_container_width=True):
            st.session_state["nav_page"] = "Verification Results"
            st.rerun()
    with col_act3:
        if st.button("🛡️ Duplicate Shield Scan", use_container_width=True):
            st.session_state["nav_page"] = "Duplicate Detection"
            st.rerun()
    with col_act4:
        if st.button("📄 Export Audit Reports", use_container_width=True):
            st.session_state["nav_page"] = "Reports"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # Retrieve real metrics from Firestore aggregation queries
    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    kpis = firebase_service.get_dashboard_metrics(owner_uid)

    # 1. Primary Executive KPI Row (6 cards)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card("Total Invoices", kpis.get("total_documents", 0), "Portfolio intake", "↑ active")
    with c2:
        render_kpi_card("Audited Value", f"${kpis.get('total_invoice_value', 0.0):,.2f}", "Total transaction volume")
    with c3:
        render_kpi_card("Verified Clean", kpis.get("verified_count", 0), "100% arithmetic pass", "✓ clean")
    with c4:
        render_kpi_card("Flagged High Risk", kpis.get("high_risk_count", 0), "Discrepancy / Collision", "⚠ review")
    with c5:
        render_kpi_card("Duplicates", kpis.get("duplicates_count", 0), "pHash / SHA matches")
    with c6:
        render_kpi_card("Success Rate", f"{kpis.get('verification_success_rate', 100.0):.1f}%", "Straight-through pass")

    st.markdown("---")

    # 2. Charts Section (Risk Distribution, Review Decisions & Monthly Trends)
    ch1, ch2, ch3 = st.columns([1, 1, 1.2])

    with ch1:
        st.markdown("### Risk Level Breakdown")
        risk_counts = {
            "Low Risk": kpis.get("normal_count", 0),
            "Medium Variance": kpis.get("medium_risk_count", 0),
            "High Risk": kpis.get("high_risk_count", 0),
        }
        df_risk = pd.DataFrame([{"Risk Level": k, "Count": v} for k, v in risk_counts.items()])
        df_risk_nonzero = df_risk[df_risk["Count"] > 0]
        if not df_risk_nonzero.empty:
            fig_pie = px.pie(
                df_risk_nonzero,
                values="Count",
                names="Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "Low Risk": "#34C759",
                    "Medium Variance": "#FF9500",
                    "High Risk": "#FF3B30",
                },
                hole=0.62,
                template="plotly_white",
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent",
                hoverinfo="label+value+percent",
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )
            fig_pie.update_layout(
                margin=dict(t=15, b=25, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1D1D1F", family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', Inter, sans-serif"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No documents uploaded yet to display risk breakdown.")

    with ch2:
        st.markdown("### Review Decision Breakdown")
        rev_counts = {
            "Approved": kpis.get("approved_count", 0),
            "Rejected": kpis.get("rejected_count", 0),
            "Pending Review": kpis.get("pending_reviews", 0),
        }
        df_rev = pd.DataFrame([{"Status": k, "Count": v} for k, v in rev_counts.items()])
        if df_rev["Count"].sum() > 0:
            fig_bar = px.bar(
                df_rev,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map={
                    "Approved": "#34C759",
                    "Rejected": "#FF3B30",
                    "Pending Review": "#0071E3",
                },
                template="plotly_white",
            )
            fig_bar.update_layout(
                margin=dict(t=15, b=25, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1D1D1F", family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', Inter, sans-serif"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No review decision records available.")

    with ch3:
        st.markdown("### Monthly Processing Volume")
        trends = kpis.get("monthly_trends", {})
        if trends:
            formatted_trends = []
            for k, v in trends.items():
                try:
                    dt = datetime.strptime(str(k)[:7], "%Y-%m")
                    m_label = dt.strftime("%b %Y")
                except Exception:
                    m_label = str(k)
                formatted_trends.append({"Month": m_label, "Invoices": v})

            df_trends = pd.DataFrame(formatted_trends)
            fig_trend = px.line(
                df_trends,
                x="Month",
                y="Invoices",
                markers=True,
                color_discrete_sequence=["#0071E3"],
                template="plotly_white",
            )
            fig_trend.update_xaxes(type="category", tickangle=0)
            max_inv = max(df_trends["Invoices"]) if not df_trends.empty else 5
            fig_trend.update_yaxes(rangemode="tozero", dtick=max(1, int(max_inv / 4) if max_inv > 4 else 1))
            fig_trend.update_layout(
                margin=dict(t=15, b=25, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1D1D1F", family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', Inter, sans-serif"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Processing trends will populate as documents are recorded across months.")

    st.markdown("---")

    # 3. Queues (Recent Uploads & Recent Suspicious Documents)
    q_col1, q_col2 = st.columns(2)

    with q_col1:
        st.markdown("### 📥 Recent Document Intake Queue")
        recents = kpis.get("recent_documents", [])
        if recents:
            table_rows = []
            for d in recents:
                fields = d.get("fields", {})
                risk = d.get("risk", {})
                rev = d.get("review") or {}
                table_rows.append({
                    "Doc ID": d.get("docId", "")[:12],
                    "Invoice #": fields.get("invoiceNumber", "N/A"),
                    "Vendor": fields.get("vendor", "N/A"),
                    "Total": f"${float(fields.get('total', 0.0)):,.2f}" if fields.get("total") else "N/A",
                    "Risk": risk.get("level", "LOW"),
                    "Status": (rev.get("decision") or "unreviewed").title(),
                })
            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
                height=280,
                column_config={
                    "Doc ID": st.column_config.TextColumn("Doc ID", width="small"),
                    "Invoice #": st.column_config.TextColumn("Invoice #", width="medium"),
                    "Vendor": st.column_config.TextColumn("Vendor", width="medium"),
                    "Total": st.column_config.TextColumn("Total", width="small"),
                    "Risk": st.column_config.TextColumn("Risk Level", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                },
            )
        else:
            st.info("No documents in portfolio yet. Upload an invoice to get started.")

    with q_col2:
        st.markdown("### ⚠️ Priority Suspicious Queue (Requires Review)")
        suspicious = kpis.get("recent_suspicious", [])
        if suspicious:
            susp_rows = []
            for d in suspicious:
                fields = d.get("fields", {})
                risk = d.get("risk", {})
                susp_rows.append({
                    "Doc ID": d.get("docId", "")[:12],
                    "Invoice #": fields.get("invoiceNumber", "N/A"),
                    "Vendor": fields.get("vendor", "N/A"),
                    "Score": f"{risk.get('score', 0):.0f}/100",
                    "Primary Flag": risk.get("reasons", ["Flagged"])[0][:45] + "...",
                })
            st.dataframe(
                pd.DataFrame(susp_rows),
                use_container_width=True,
                hide_index=True,
                height=280,
                column_config={
                    "Doc ID": st.column_config.TextColumn("Doc ID", width="small"),
                    "Invoice #": st.column_config.TextColumn("Invoice #", width="medium"),
                    "Vendor": st.column_config.TextColumn("Vendor", width="medium"),
                    "Score": st.column_config.TextColumn("Score", width="small"),
                    "Primary Flag": st.column_config.TextColumn("Primary Flag", width="large"),
                },
            )
        else:
            st.success("✓ No high-risk documents currently flagged for investigation.")
