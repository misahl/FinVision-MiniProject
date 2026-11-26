"""Reusable UI Components for FinVision Streamlit Application (Apple Pro Light Mode)."""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_topbar(user: Dict[str, Any], active_section: str = "Ingest & Verify"):
    """Render Apple-styled light fintech topbar with breadcrumb, search, notifications, and profile."""
    user_name = user.get("displayName") or "Analyst"
    role = user.get("role", "analyst").upper()
    role_color = "#0071E3" if role == "ADMIN" else "#248A3D"

    # Map active section to category breadcrumb
    category = "Workflow"
    if active_section in ["Risk Analytics", "Document History", "Reports"]:
        category = "Insights"
    elif active_section in ["Settings", "Administration"]:
        category = "System"

    st.markdown(
        f"""
        <div class="fin-topbar">
            <div class="fin-topbar-left" style="display: flex; align-items: center; gap: 14px;">
                <div class="fin-topbar-breadcrumb" style="display: flex; align-items: center; gap: 6px; font-size: 0.84rem; color: #6E6E73;">
                    <span class="fin-breadcrumb-cat" style="font-weight: 600; color: #6E6E73;">{category}</span>
                    <span class="fin-breadcrumb-sep">/</span>
                    <span class="fin-breadcrumb-active" style="font-weight: 700; color: #1D1D1F;">{active_section}</span>
                </div>
            </div>
            <div class="fin-topbar-right" style="display: flex; align-items: center; gap: 16px;">
                <div class="fin-topbar-search">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6E6E73" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    <span class="fin-search-text" style="color: #6E6E73; font-size: 0.8rem; margin-left: 6px;">Search portfolio invoices, vendors...</span>
                    <span class="fin-search-shortcut" style="background: rgba(0,0,0,0.06); font-size: 0.68rem; padding: 1px 6px; border-radius: 4px; color: #6E6E73; margin-left: 12px; font-weight: 600;">⌘K</span>
                </div>
                <div class="fin-notification-bell" title="System alerts">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1D1D1F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
                    <span class="fin-bell-badge">3</span>
                </div>
                <div class="fin-user-profile-widget" style="display: flex; align-items: center; gap: 10px; border-left: 1px solid rgba(0,0,0,0.08); padding-left: 16px;">
                    <div class="fin-user-avatar" style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0071E3 0%, #0284C7 100%); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: #FFFFFF;">
                        {user_name[:1].upper()}
                    </div>
                    <div class="fin-user-meta" style="line-height: 1.15;">
                        <span style="font-weight: 700; font-size: 0.84rem; color: #1D1D1F; display: block;">{user_name}</span>
                        <span style="font-size: 0.7rem; font-weight: 700; color: {role_color};">{role}</span>
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

    lbl_html = f'<span style="color: #515154; margin-right: 4px; font-weight: 600;">{label}:</span>' if label else ""
    return f'<div class="{cls}">{lbl_html}<span>{tag}</span></div>'


def render_kpi_card(title: str, value: Any, subtitle: str = "", trend: str = ""):
    """Render an executive light Apple KPI metric card."""
    trend_html = f'<span style="color: #248A3D; font-weight: 700; margin-left: 4px;">{trend}</span>' if trend else ""
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
    """Render an individual rule outcome block with friendly names, 4 statuses, and breakdown tables."""
    passed = rule.get("passed", True)
    severity = rule.get("severity", "INFO")
    rule_id = rule.get("rule_id", "RULE")
    display_name = rule.get("display_name") or rule_id
    message = rule.get("message", "")
    raw_status = rule.get("status")

    if raw_status == "NOT CHECKED":
        card_class = "rule-item-not-checked"
        icon = "⚪"
        status_text = "NOT CHECKED"
        status_color = "#64748B"
    elif raw_status == "PASSED" or (raw_status is None and passed):
        card_class = "rule-item-pass"
        icon = "✓"
        status_text = "PASSED"
        status_color = "#16A34A"
    elif raw_status == "FAILED" or (raw_status is None and not passed and severity == "CRITICAL"):
        card_class = "rule-item-fail"
        icon = "✕"
        status_text = "FAILED (CRITICAL)" if severity == "CRITICAL" else "FAILED"
        status_color = "#DC2626"
    else:
        card_class = "rule-item-warn"
        icon = "⚠"
        status_text = "WARNING"
        status_color = "#D97706"

    # Build line item / breakdown table if details are available
    details = rule.get("details", [])
    details_html = ""
    if details and isinstance(details, list):
        rows_html = []
        for d in details:
            d_desc = d.get("description", "Item")
            d_qty = f"{d['quantity']:g}" if d.get("quantity") is not None else "-"
            d_price = f"{d['unit_price']:,.2f}" if d.get("unit_price") is not None else "-"
            d_exp = f"{d['expected_total']:,.2f}" if d.get("expected_total") is not None else "-"
            d_act = f"{d['line_total']:,.2f}" if d.get("line_total") is not None else "-"
            d_diff = f"{d['difference']:,.2f}" if d.get("difference") is not None else "-"
            d_stat = d.get("status", "NOT CHECKED")

            if d_stat == "PASSED":
                stat_badge = '<span style="color: #16A34A; font-weight: 700;">✓ PASSED</span>'
            elif d_stat == "FAILED":
                stat_badge = '<span style="color: #DC2626; font-weight: 700;">✕ FAILED</span>'
            elif d_stat == "WARNING":
                stat_badge = '<span style="color: #D97706; font-weight: 700;">⚠ WARNING</span>'
            else:
                stat_badge = '<span style="color: #64748B; font-weight: 600;">⚪ NOT CHECKED</span>'

            row = (
                f'<tr style="border-bottom: 1px solid rgba(0,0,0,0.05);">'
                f'<td style="padding: 6px 8px; font-weight: 600; color: #1D1D1F;">{d_desc}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: #374151;">{d_qty}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: #374151;">{d_price}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: #16A34A; font-weight: 600;">{d_exp}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: #1D1D1F; font-weight: 600;">{d_act}</td>'
                f'<td style="padding: 6px 8px; text-align: right; color: #6E6E73;">{d_diff}</td>'
                f'<td style="padding: 6px 8px; text-align: center;">{stat_badge}</td>'
                f'</tr>'
            )
            rows_html.append(row)

        details_html = (
            f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(0,0,0,0.08); overflow-x: auto;">'
            f'<div style="font-size: 0.76rem; font-weight: 700; color: #6E6E73; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em;">Itemized Calculation Breakdown (Qty × Unit Price = Stated Total)</div>'
            f'<table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; background: rgba(255,255,255,0.6); border-radius: 6px;">'
            f'<thead>'
            f'<tr style="background: rgba(0,0,0,0.03); color: #6E6E73; border-bottom: 1px solid rgba(0,0,0,0.08); font-size: 0.74rem;">'
            f'<th style="padding: 6px 8px; text-align: left;">Item Description</th>'
            f'<th style="padding: 6px 8px; text-align: right;">Quantity</th>'
            f'<th style="padding: 6px 8px; text-align: right;">Unit Price</th>'
            f'<th style="padding: 6px 8px; text-align: right;">Expected</th>'
            f'<th style="padding: 6px 8px; text-align: right;">Stated</th>'
            f'<th style="padding: 6px 8px; text-align: right;">Diff</th>'
            f'<th style="padding: 6px 8px; text-align: center;">Status</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>{"".join(rows_html)}</tbody>'
            f'</table>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; flex-wrap: wrap; gap: 4px;">
                <div style="display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-weight: 700; font-size: 0.92rem; color: #1D1D1F;">{display_name}</span>
                    <span style="font-family: monospace; font-size: 0.72rem; color: #86868B; background: rgba(0,0,0,0.04); padding: 2px 6px; border-radius: 4px;">{rule_id}</span>
                </div>
                <span style="font-weight: 700; font-size: 0.8rem; color: {status_color};">{icon} {status_text}</span>
            </div>
            <div style="font-size: 0.84rem; color: #424245; line-height: 1.45;">{message}</div>
            {details_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
