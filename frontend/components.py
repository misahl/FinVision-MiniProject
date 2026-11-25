"""Reusable UI Components for FinVision Streamlit Application."""

from typing import Any, Dict
import streamlit as st


def render_topbar(user: Dict[str, Any]):
    """Render the ultra-modern top navigation header ribbon."""
    user_name = user.get("displayName") or "Analyst"
    role = user.get("role", "analyst").upper()
    role_color = "#3B82F6" if role == "ADMIN" else "#10B981"

    st.markdown(
        f"""
        <div class="fin-topbar">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div class="fin-brand-title">🛡️ FinVision <span style="font-size: 0.75rem; font-weight: 600; color: #60A5FA; letter-spacing: 0.08em; background: rgba(59, 130, 246, 0.15); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.3);">2.0 PRO</span></div>
                <div class="fin-status-pill">
                    <span class="fin-status-dot"></span>
                    <span>System Active & Verified</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="text-align: right; line-height: 1.2;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #F8FAFC;">{user_name}</span>
                    <span style="display: block; font-size: 0.72rem; font-weight: 700; color: {role_color}; letter-spacing: 0.05em;">{role}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: Any, subtitle: str = "", trend: str = ""):
    """Render a fintech-style KPI metric card."""
    trend_html = f'<span style="color: #34D399; font-weight: 600; margin-left: 4px;">{trend}</span>' if trend else ""
    st.markdown(
        f"""
        <div class="fin-card">
            <div class="fin-card-title">{title}</div>
            <div class="fin-card-value">{value}</div>
            <div class="fin-card-subtitle">{subtitle} {trend_html}</div>
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
        status_color = "#34D399"
    elif severity == "CRITICAL":
        card_class = "rule-item-fail"
        icon = "✕"
        status_text = "FAILED (CRITICAL)"
        status_color = "#F87171"
    else:
        card_class = "rule-item-warn"
        icon = "⚠"
        status_text = "WARNING"
        status_color = "#FBBF24"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 0.88rem; color: #F8FAFC;">{rule_id}</span>
                <span style="font-weight: 700; font-size: 0.8rem; color: {status_color};">{icon} {status_text}</span>
            </div>
            <div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.4;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
