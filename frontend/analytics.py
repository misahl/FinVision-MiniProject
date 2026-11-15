"""Financial Risk Analytics & Fraud Pattern Diagnostics for FinVision."""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import firebase_service


def render_analytics(user: Dict[str, Any]):
    """Render financial risk analytics, rule failure frequencies, and model insights."""
    st.markdown("## 📈 Financial Risk Analytics & Anomaly Diagnostics")
    st.caption("Deep-dive into portfolio risk scores, frequent rule violations, visual forensic anomalies, and OCR extraction diagnostics.")

    owner_uid = None if user.get("role") == "admin" else user.get("uid")
    docs = firebase_service.list_documents(owner_uid=owner_uid, limit=200)

    if not docs:
        st.info("No documents available for analytics yet. Upload invoices to populate charts.")
        return

    # Extract analytics data
    rule_failure_counts = {}
    cv_scores = []
    ocr_confidences = []
    risk_scores = []
    factor_weights = {"rules": 0.0, "duplicates": 0.0, "cv": 0.0, "ml": 0.0}

    for d in docs:
        verif = d.get("verification", {})
        for r in verif.get("ruleResults", []):
            if not r.get("passed", True):
                rid = r.get("rule_id", "OTHER")
                rule_failure_counts[rid] = rule_failure_counts.get(rid, 0) + 1

        cv_scores.append(d.get("cv", {}).get("cvScore", 0.0))
        ocr_confidences.append(d.get("overallConfidence", 0.0))
        risk_obj = d.get("risk", {})
        risk_scores.append(risk_obj.get("score", 0.0))

        bd = risk_obj.get("breakdown", {})
        factor_weights["rules"] += bd.get("rules_contribution", 0.0)
        factor_weights["duplicates"] += bd.get("duplicate_contribution", 0.0)
        factor_weights["cv"] += bd.get("cv_contribution", 0.0)
        factor_weights["ml"] += bd.get("ml_contribution", 0.0)

    # 1. Top Diagnostic Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Average Risk Score</div>
                <div class="fin-card-value">{avg_risk:.1f}/100</div>
                <div class="fin-card-subtitle">Portfolio composite mean</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        total_violations = sum(rule_failure_counts.values())
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Total Rule Variances</div>
                <div class="fin-card-value" style="color: {'#EF4444' if total_violations > 0 else '#10B981'};">{total_violations}</div>
                <div class="fin-card-subtitle">Flagged arithmetic discrepancies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        avg_ocr = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Mean OCR Confidence</div>
                <div class="fin-card-value">{avg_ocr:.1f}%</div>
                <div class="fin-card-subtitle">Tesseract extraction accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        avg_cv = sum(cv_scores) / len(cv_scores) if cv_scores else 0.0
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-title">Mean Forensic Anomaly</div>
                <div class="fin-card-value">{avg_cv:.1f}/100</div>
                <div class="fin-card-subtitle">Computer vision outlier score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Charts: Rule Failures & Factor Breakdown
    col1, col2 = st.columns([1.2, 1])

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
            fig_rules.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.success("✓ No rule failures recorded across the current portfolio.")

    with col2:
        st.markdown("### Risk Factor Driver Contribution")
        df_factors = pd.DataFrame([
            {"Risk Factor": "Rule Invariants", "Weighted Points": factor_weights["rules"]},
            {"Risk Factor": "Duplicate Collisions", "Weighted Points": factor_weights["duplicates"]},
            {"Risk Factor": "Computer Vision Anomalies", "Weighted Points": factor_weights["cv"]},
            {"Risk Factor": "ML Statistical Variance", "Weighted Points": factor_weights["ml"]},
        ])
        if df_factors["Weighted Points"].sum() > 0:
            fig_factors = px.pie(
                df_factors,
                names="Risk Factor",
                values="Weighted Points",
                color="Risk Factor",
                color_discrete_map={
                    "Rule Invariants": "#EF4444",
                    "Duplicate Collisions": "#F59E0B",
                    "Computer Vision Anomalies": "#3B82F6",
                    "ML Statistical Variance": "#8B5CF6",
                },
                hole=0.55,
            )
            fig_factors.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.info("No risk points accumulated in portfolio yet.")

    st.markdown("---")

    # 3. Distribution Charts: Risk Scores & OCR Quality
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### Risk Score Distribution (Histogram)")
        df_scores = pd.DataFrame({"Risk Score": risk_scores})
        fig_hist = px.histogram(
            df_scores,
            x="Risk Score",
            nbins=10,
            color_discrete_sequence=["#2563EB"],
        )
        fig_hist.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        st.markdown("### OCR Extraction Quality Distribution")
        df_ocr = pd.DataFrame({"OCR Confidence (%)": ocr_confidences})
        fig_ocr = px.histogram(
            df_ocr,
            x="OCR Confidence (%)",
            nbins=10,
            color_discrete_sequence=["#10B981"],
        )
        fig_ocr.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ocr, use_container_width=True)
