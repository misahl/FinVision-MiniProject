"""System Settings and Configuration for FinVision."""

import json
from pathlib import Path
import streamlit as st

from src import verification


def render_settings(user: dict):
    """Render profile details, rule tolerances, and model registry specs."""
    st.markdown("## Configuration & System Settings")
    st.caption("Inspect active analyst profile identity, configured financial rule tolerances, and serialized ML model registry metadata.")

    tab_profile, tab_rules, tab_model = st.tabs(["User Profile", "Rule Tolerances", "Model Registry"])

    with tab_profile:
        st.markdown("### Active Session Identity")
        st.markdown(
            f"""
            <div class="fin-card">
                <div style="display: grid; grid-template-columns: 140px 1fr; row-gap: 12px; font-size: 0.92rem;">
                    <div style="font-weight: 600; color: #64748B;">Display Name:</div>
                    <div style="font-weight: 600; color: #0F172A;">{user.get('displayName', 'Analyst')}</div>
                    
                    <div style="font-weight: 600; color: #64748B;">Work Email:</div>
                    <div style="font-weight: 500; color: #0F172A;">{user.get('email', 'N/A')}</div>
                    
                    <div style="font-weight: 600; color: #64748B;">Assigned Role:</div>
                    <div><span class="badge-low">{user.get('role', 'analyst').upper()}</span></div>
                    
                    <div style="font-weight: 600; color: #64748B;">User UID:</div>
                    <div style="font-family: monospace; font-size: 0.85rem; color: #334155;">{user.get('uid', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_rules:
        st.markdown("### Configured Financial Rules & Tolerances")
        st.caption("Loaded from config/rules_config.json")
        st.markdown(
            f"""
            <div class="fin-card">
                <div style="display: grid; grid-template-columns: 260px 1fr; row-gap: 14px; font-size: 0.92rem;">
                    <div style="font-weight: 600; color: #64748B;">Math Tolerance (Rounding Cents):</div>
                    <div style="font-weight: 700; color: #0F172A;">${verification.TOLERANCE}</div>
                    
                    <div style="font-weight: 600; color: #64748B;">Min History for Outliers:</div>
                    <div style="font-weight: 700; color: #0F172A;">{verification.MIN_HISTORY_FOR_OUTLIERS} prior documents</div>
                    
                    <div style="font-weight: 600; color: #64748B;">Outlier Robust Z-Score:</div>
                    <div style="font-weight: 700; color: #0F172A;">{verification.OUTLIER_Z_THRESHOLD}</div>
                    
                    <div style="font-weight: 600; color: #64748B;">Max Sanity Invoice Age:</div>
                    <div style="font-weight: 700; color: #0F172A;">{verification.MAX_AGE_YEARS} years</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_model:
        st.markdown("### Active ML Model Specifications (modelRegistry)")
        meta_path = Path(__file__).parent.parent / "models" / "model_metadata_v1.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.json(meta)
        else:
            st.info("Model metadata file not found. Run model training to view specs.")
