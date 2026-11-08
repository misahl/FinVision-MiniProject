"""Analytics & Fraud Risk Patterns Page for FinVision."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import firebase_service


def render_analytics(user: dict):
    """Render financial risk analytics, rule failure frequencies, and model insights."""
    st.markdown("## Financial Verification Analytics & Insights")
    st.markdown("Aggregated risk metrics, frequent rule violations, and extraction quality diagnostics.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=100)

    if not docs:
        st.info("No documents available for analytics yet. Upload invoices to populate charts.")
        return

    # Extract analytics data
    rule_failure_counts = {}
    cv_scores = []
    ocr_confidences = []
    dates = []
    risk_scores = []

    for d in docs:
        # Rule failures
        verif = d.get("verification", {})
        for r in verif.get("ruleResults", []):
            if not r.get("passed", True):
                rid = r.get("rule_id", "OTHER")
                rule_failure_counts[rid] = rule_failure_counts.get(rid, 0) + 1

        # Scores
        cv_scores.append(verif.get("cvScore", 0.0))
        ocr_confidences.append(d.get("overallConfidence", 0.0))
        risk_scores.append(d.get("risk", {}).get("score", 0.0))
        dates.append(d.get("createdAt", "")[:10])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Most Frequent Rule Variances")
        if rule_failure_counts:
            df_rules = pd.DataFrame([
                {"Rule ID": k, "Failure Count": v}
                for k, v in sorted(rule_failure_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            fig_rules = px.bar(
                df_rules,
                x="Rule ID",
                y="Failure Count",
                color="Failure Count",
                color_continuous_scale="Reds",
            )
            fig_rules.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.success("No rule failures recorded across current portfolio.")

    with col2:
        st.markdown("### Risk Score Distribution")
        df_scores = pd.DataFrame({"Risk Score": risk_scores})
        fig_hist = px.histogram(
            df_scores,
            x="Risk Score",
            nbins=10,
            color_discrete_sequence=["#2563EB"],
        )
        fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Extraction Quality & OCR Diagnostics")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        avg_ocr = sum(ocr_confidences) / max(1, len(ocr_confidences))
        st.metric("Average OCR Confidence", f"{avg_ocr:.1f}%")
    with col_q2:
        avg_cv = sum(cv_scores) / max(1, len(cv_scores))
        st.metric("Average CV Anomaly Indicator Score", f"{avg_cv:.1f} / 100")

