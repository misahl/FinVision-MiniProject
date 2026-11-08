"""Reusable UI Components for FinVision Streamlit Application."""

from typing import Any, Dict
import streamlit as st


def render_kpi_card(title: str, value: Any, subtitle: str = "", trend: str = ""):
    """Render a fintech-style KPI metric card."""
    st.markdown(
        f"""
        <div class="fin-card">
            <div class="fin-card-title">{title}</div>
            <div class="fin-card-value">{value}</div>
            <div class="fin-card-subtitle">{subtitle} {trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(level: str, score: float = 0.0) -> str:
    """Return HTML string for modern risk level pill."""
    lvl = str(level).upper()
    if lvl == "HIGH":
        css_class = "badge-high"
    elif lvl == "MEDIUM":
        css_class = "badge-medium"
    else:
        css_class = "badge-low"

    score_str = f" ({score:.0f})" if score > 0 else ""
    return f'<span class="{css_class}">{lvl}{score_str}</span>'


def render_rule_card(rule: Dict[str, Any]):
    """Render an individual rule outcome block."""
    passed = rule.get("passed", True)
    severity = rule.get("severity", "INFO")
    rule_id = rule.get("rule_id", "RULE")
    message = rule.get("message", "")

    if passed:
        card_class = "rule-item-pass"
        icon = "✓"
        status_text = "PASSED"
        status_color = "#10B981"
    elif severity == "CRITICAL":
        card_class = "rule-item-fail"
        icon = "✕"
        status_text = "FAILED (CRITICAL)"
        status_color = "#EF4444"
    else:
        card_class = "rule-item-warn"
        icon = "⚠"
        status_text = "WARNING"
        status_color = "#F59E0B"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 600; font-size: 0.9rem; color: #0F172A;">{rule_id}</span>
                <span style="font-weight: 700; font-size: 0.8rem; color: {status_color};">{icon} {status_text}</span>
            </div>
            <div style="font-size: 0.85rem; color: #475569;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

