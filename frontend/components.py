"""Reusable UI Components for FinVision Streamlit Application."""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_topbar(user: Dict[str, Any], active_section: str = "Ingest & Verify"):
    """Render enterprise fintech topbar with breadcrumb, global search, notifications, and profile."""
    user_name = user.get("displayName") or "Analyst"
    role = user.get("role", "analyst").upper()
    role_color = "#3B82F6" if role == "ADMIN" else "#10B981"

    # Map active section to category breadcrumb
    category = "Workflow"
    if active_section in ["Risk Analytics", "Document History", "Reports"]:
        category = "Insights"
    elif active_section in ["Settings", "Administration"]:
        category = "System"

    st.markdown(
        f"""
        <div class="fin-topbar">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="display: flex; align-items: center; gap: 6px; font-size: 0.82rem; color: #64748B;">
                    <span style="font-weight: 600; color: #94A3B8;">{category}</span>
                    <span>/</span>
                    <span style="font-weight: 700; color: #F8FAFC;">{active_section}</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 18px;">
                <div class="fin-topbar-search">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    <span style="color: #64748B; font-size: 0.8rem; margin-left: 6px;">Search portfolio invoices, vendors...</span>
                    <span style="background: rgba(255,255,255,0.08); font-size: 0.68rem; padding: 1px 6px; border-radius: 4px; color: #94A3B8; margin-left: 12px;">⌘K</span>
                </div>
                <div class="fin-notification-bell" title="System alerts">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
                    <span class="fin-bell-badge">3</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 16px;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: #FFFFFF;">
                        {user_name[:1].upper()}
                    </div>
                    <div style="line-height: 1.15;">
                        <span style="font-weight: 700; font-size: 0.84rem; color: #F8FAFC; display: block;">{user_name}</span>
                        <span style="font-size: 0.7rem; font-weight: 600; color: {role_color};">{role}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current_step: int = 1):
    """Render horizontal verification lifecycle stepper (Upload > Extract > Validate > Risk Score)."""
    steps = [
        ("1", "Upload", current_step >= 1, current_step == 1),
        ("2", "Extract", current_step >= 2, current_step == 2),
        ("3", "Validate", current_step >= 3, current_step == 3),
        ("4", "Risk Score", current_step >= 4, current_step == 4),
    ]

    html_parts = ['<div class="fin-stepper">']
    for i, (num, label, is_completed_or_curr, is_active) in enumerate(steps):
        step_class = "fin-step active" if is_active else ("fin-step completed" if is_completed_or_curr else "fin-step")
        circle_content = "✓" if (is_completed_or_curr and not is_active and current_step > int(num)) else num

        html_parts.append(
            f"""
            <div class="{step_class}">
                <div class="fin-step-circle">{circle_content}</div>
                <span class="fin-step-label">{num}. {label}</span>
            </div>
            """
        )
        if i < len(steps) - 1:
            line_class = "fin-step-line active" if current_step > (i + 1) else "fin-step-line"
            html_parts.append(f'<div class="{line_class}"></div>')

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_confidence_chip(score: float, label: str = "") -> str:
    """Return HTML chip for per-field confidence rating with amber review flag on low scores."""
    if score >= 85.0:
        cls = "conf-chip conf-chip-high"
        tag = f"High • {score:.0f}%"
    elif score >= 65.0:
        cls = "conf-chip conf-chip-med"
        tag = f"Medium • {score:.0f}%"
    else:
        cls = "conf-chip conf-chip-low"
        tag = f"Low • {score:.0f}% • Review"

    lbl_html = f'<span style="color: #94A3B8; margin-right: 4px;">{label}:</span>' if label else ""
    return f'<div class="{cls}">{lbl_html}<span>{tag}</span></div>'


def render_kpi_card(title: str, value: Any, subtitle: str = "", trend: str = ""):
    """Render an executive dark fintech KPI metric card."""
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
