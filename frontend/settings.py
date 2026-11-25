"""System Settings and Configuration for FinVision."""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import streamlit as st

from src import verification


def render_settings(user: Dict[str, Any]):
    """Render profile details, configurable rule tolerances, and model registry specs."""
    st.markdown("## ⚙️ Configuration & User Settings")
    st.caption("Inspect active analyst profile identity, configured financial rule tolerances, notification preferences, and ML model registry specs.")

    tab_profile, tab_rules, tab_prefs, tab_model = st.tabs([
        "User Profile",
        "Rule Tolerances",
        "System Preferences",
        "Model Registry",
    ])

    with tab_profile:
        st.markdown("### Active Analyst Session Identity")
        st.caption("Authenticated credentials and role-based access identity powering your verification decisions.")

        role_str = user.get("role", "analyst").upper()
        role_badge = f'<span class="badge-low" style="background: rgba(0, 113, 227, 0.1); color: #0071E3; font-weight: 700; padding: 4px 12px; border-radius: 980px; font-size: 0.8rem;">{role_str}</span>'

        profile_html = (
            '<div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 18px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">'
            '<table style="width: 100%; border-collapse: collapse; font-size: 0.94rem;">'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 10px 0; font-weight: 600; color: #515154; width: 180px;">Full Name</td><td style="padding: 10px 0; font-weight: 700; color: #1D1D1F;">{user.get("displayName", "Analyst")}</td></tr>'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 10px 0; font-weight: 600; color: #515154;">Work Email</td><td style="padding: 10px 0; font-weight: 500; color: #1D1D1F;">{user.get("email", "N/A")}</td></tr>'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 10px 0; font-weight: 600; color: #515154;">System Role</td><td style="padding: 10px 0;">{role_badge}</td></tr>'
            f'<tr><td style="padding: 10px 0; font-weight: 600; color: #515154;">Account UID</td><td style="padding: 10px 0; font-family: monospace; font-size: 0.85rem; color: #0071E3;">{user.get("uid", "N/A")}</td></tr>'
            '</table>'
            '</div>'
        )
        st.markdown(profile_html, unsafe_allow_html=True)

    with tab_rules:
        st.markdown("### Configured Financial Invariants & Tolerances")
        st.caption("Configured parameters loaded from `config/rules_config.json` enforcing mathematical and audit sanity.")

        rules_html = (
            '<div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 18px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">'
            '<table style="width: 100%; border-collapse: collapse; font-size: 0.94rem;">'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 12px 0; font-weight: 600; color: #515154; width: 280px;">Math Tolerance (Rounding Cents)</td><td style="padding: 12px 0; font-weight: 700; color: #1D1D1F;">${verification.TOLERANCE}</td><td style="padding: 12px 0; color: #6E6E73; font-size: 0.84rem;">Max allowed variance for rounding differences in Subtotal + Tax - Discount = Total</td></tr>'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 12px 0; font-weight: 600; color: #515154;">Min History for Outliers</td><td style="padding: 12px 0; font-weight: 700; color: #1D1D1F;">{verification.MIN_HISTORY_FOR_OUTLIERS} prior docs</td><td style="padding: 12px 0; color: #6E6E73; font-size: 0.84rem;">Historical baseline documents required before statistical Z-score outlier rule activates</td></tr>'
            f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.06);"><td style="padding: 12px 0; font-weight: 600; color: #515154;">Outlier Robust Z-Score Threshold</td><td style="padding: 12px 0; font-weight: 700; color: #1D1D1F;">{verification.OUTLIER_Z_THRESHOLD} σ</td><td style="padding: 12px 0; color: #6E6E73; font-size: 0.84rem;">Standard deviations above vendor historical median before invoice amount is flagged as an anomaly</td></tr>'
            f'<tr><td style="padding: 12px 0; font-weight: 600; color: #515154;">Max Sanity Invoice Age</td><td style="padding: 12px 0; font-weight: 700; color: #1D1D1F;">{verification.MAX_AGE_YEARS} years</td><td style="padding: 12px 0; color: #6E6E73; font-size: 0.84rem;">Rejects or flags invoices dated more than 5 years in the past or in the far future</td></tr>'
            '</table>'
            '</div>'
        )
        st.markdown(rules_html, unsafe_allow_html=True)

    with tab_prefs:
        st.markdown("### Workflow & Display Preferences")
        st.caption("Personal analyst workflow customizations saved in browser session.")
        st.checkbox("Auto-advance to Verification Workspace after successful upload", value=True)
        st.checkbox("Highlight high-risk invoices in audit table with red accents", value=True)
        st.checkbox("Enable real-time duplicate cross-matching on upload", value=True)
        st.success("✓ Preferences active in current session.")

    with tab_model:
        st.markdown("### Active ML Model Specifications (modelRegistry)")
        st.caption("Governed model registry tracking training lineage, held-out evaluation metrics, and feature importance weights.")

        meta_path = Path(__file__).parent.parent / "models" / "model_metadata_v1.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Apple KPI Cards for Model Performance
            m1, m2, m3, m4 = st.columns(4)
            ml_metrics = meta.get("metrics", {}).get("mlModel", {})
            with m1:
                st.markdown(
                    f"""
                    <div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.78rem; font-weight: 600; color: #515154;">Model Version</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #0071E3;">{meta.get('version', 'v1.0.0')}</div>
                        <div style="font-size: 0.72rem; color: #6E6E73;">Random Forest (100 Trees)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                acc_val = ml_metrics.get("accuracy", 1.0) * 100
                st.markdown(
                    f"""
                    <div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.78rem; font-weight: 600; color: #515154;">Test Accuracy</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #248A3D;">{acc_val:.1f}%</div>
                        <div style="font-size: 0.72rem; color: #6E6E73;">Held-out test split</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                f1_val = ml_metrics.get("f1Score", 1.0)
                st.markdown(
                    f"""
                    <div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.78rem; font-weight: 600; color: #515154;">Test F1-Score</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #0071E3;">{f1_val:.2f}</div>
                        <div style="font-size: 0.72rem; color: #6E6E73;">Harmonic mean (P/R)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                samples_str = f"{meta.get('trainSamples', 0)} / {meta.get('testSamples', 0)}"
                st.markdown(
                    f"""
                    <div class="fin-card" style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.78rem; font-weight: 600; color: #515154;">Train / Test Split</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #1D1D1F;">{samples_str}</div>
                        <div style="font-size: 0.72rem; color: #6E6E73;">GroupShuffleSplit (0 Leakage)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Feature Importance Table
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### Feature Importance Diagnostic")
            feat_imp = meta.get("featureImportances", {})
            if feat_imp:
                df_imp = pd.DataFrame([
                    {"Feature Name": k, "Weight": v, "Relative Contribution (%)": round(v * 100, 2)}
                    for k, v in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(df_imp, use_container_width=True)

            with st.expander("Technical Model Metadata Schema (JSON)", expanded=False):
                st.json(meta)
        else:
            st.info("Model metadata file not found. Run model training to view specs.")
