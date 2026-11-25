"""FinVision: Intelligent Financial Document Verification System.

Streamlit application entry point, authentication gate, and page router.
"""

from pathlib import Path
import streamlit as st

from src import firebase_service
from frontend import (
    components,
    dashboard,
    upload,
    verify,
    history,
    duplicate_workspace,
    analytics,
    reports,
    settings,
    admin,
)

# -------------------------------------------------------------------------
# Streamlit Application Configuration
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="FinVision | Financial Document Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern fintech stylesheet once
CSS_PATH = Path(__file__).parent / "frontend" / "styles.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Session State & Authentication Gate
# -------------------------------------------------------------------------

if "user" not in st.session_state:
    st.session_state["user"] = None
if "id_token" not in st.session_state:
    st.session_state["id_token"] = None
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"


def render_auth_screen():
    """Render modern login and registration interface."""
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    st.markdown("# 🛡️ FinVision 2.0")
    st.markdown("### Intelligent Financial Document Verification System")
    st.markdown(
        "<p style='color: #64748B; max-width: 580px; margin: 0 auto;'>"
        "Enterprise-grade financial document verification combining Tesseract OCR layout heuristics, "
        "deterministic arithmetic invariants, multi-modal duplicate detection, and explainable ML risk assessment."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_space1, col_center, col_space2 = st.columns([1, 1.25, 1])

    with col_center:
        st.markdown("<br/>", unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Work Email", value="analyst@finvision.test")
                password = st.text_input("Password", type="password", value="FinVision2026!")
                submit_login = st.form_submit_button("Sign In to FinVision", type="primary", use_container_width=True)

                if submit_login:
                    try:
                        auth_res = firebase_service.sign_in_user(email, password)
                        claims = firebase_service.verify_user_token(auth_res["idToken"])
                        st.session_state["user"] = {
                            "uid": auth_res["localId"],
                            "email": auth_res["email"],
                            "displayName": auth_res.get("displayName", "Analyst"),
                            "role": auth_res.get("role", "analyst"),
                        }
                        st.session_state["id_token"] = auth_res["idToken"]
                        st.success("Authentication successful!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign-in error: {str(e)}")

            # 1-Click Fast Evaluation Logins
            st.markdown("<div style='text-align: center; margin-top: 14px;'>", unsafe_allow_html=True)
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                if st.button("⚡ Analyst Demo", use_container_width=True):
                    try:
                        try:
                            auth_res = firebase_service.sign_in_user("demo_analyst@finvision.test", "DemoPassword123!")
                        except Exception:
                            auth_res = firebase_service.sign_up_user("demo_analyst@finvision.test", "DemoPassword123!", "Demo Financial Analyst", "analyst")
                        st.session_state["user"] = {
                            "uid": auth_res["localId"],
                            "email": auth_res["email"],
                            "displayName": auth_res.get("displayName", "Demo Financial Analyst"),
                            "role": auth_res.get("role", "analyst"),
                        }
                        st.session_state["id_token"] = auth_res["idToken"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Demo sign-in error: {str(e)}")

            with col_demo2:
                if st.button("🛡️ Admin Demo", use_container_width=True):
                    try:
                        try:
                            auth_res = firebase_service.sign_in_user("demo_admin@finvision.test", "AdminPassword123!")
                        except Exception:
                            auth_res = firebase_service.sign_up_user("demo_admin@finvision.test", "AdminPassword123!", "System Administrator", "admin")
                        st.session_state["user"] = {
                            "uid": auth_res["localId"],
                            "email": auth_res["email"],
                            "displayName": auth_res.get("displayName", "System Administrator"),
                            "role": "admin",
                        }
                        st.session_state["id_token"] = auth_res["idToken"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Admin sign-in error: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_register:
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", value="Alex Rivera")
                reg_email = st.text_input("Email", value="")
                reg_pass = st.text_input("Password (min 8 chars)", type="password")
                reg_role = "analyst"
                st.caption("Role: **Financial Analyst** (Administrator privileges must be assigned by system administrator)")
                submit_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submit_register:
                    if len(reg_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        try:
                            auth_res = firebase_service.sign_up_user(reg_email, reg_pass, reg_name, reg_role)
                            st.session_state["user"] = {
                                "uid": auth_res["localId"],
                                "email": auth_res["email"],
                                "displayName": reg_name,
                                "role": reg_role,
                            }
                            st.session_state["id_token"] = auth_res["idToken"]
                            st.success("Account created successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Registration error: {str(e)}")


# -------------------------------------------------------------------------
# Authenticated Navigation & Routing
# -------------------------------------------------------------------------

import time

def render_app():
    user = st.session_state["user"]

    # Verify ID token validity periodically (every 10 minutes) rather than on every widget interaction
    last_auth = st.session_state.get("last_auth_check", 0.0)
    now = time.time()
    if now - last_auth > 600.0:
        try:
            firebase_service.verify_user_token(st.session_state["id_token"])
            st.session_state["last_auth_check"] = now
        except Exception:
            st.session_state["user"] = None
            st.session_state["id_token"] = None
            st.warning("Your session has expired. Please log in again.")
            st.rerun()

    # Sidebar: Grouped Navigation, User Card & Single Status Indicator
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 18px 0;">
                <div style="font-weight: 700; color: #1D1D1F; font-size: 1.25rem; display: flex; align-items: center; gap: 8px; letter-spacing: -0.024em;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0071E3" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    <span>FinVision</span>
                    <span style="font-size: 0.65rem; font-weight: 700; color: #0071E3; background: rgba(0,113,227,0.1); padding: 2px 8px; border-radius: 980px; border: 1px solid rgba(0,113,227,0.25); letter-spacing: 0.04em;">PRO</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_page = st.session_state.get("nav_page", "Overview")

        # 1. Workflow Group
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #6E6E73; text-transform: uppercase; letter-spacing: 0.08em; margin: 10px 0 6px 6px;'>Workflow</div>", unsafe_allow_html=True)
        workflow_pages = [
            ("Overview", "Overview"),
            ("Ingest & Verify", "Upload & Analyze"),
            ("Review Console", "Verification Results"),
            ("Duplicate Shield", "Duplicate Detection"),
        ]
        for label, page_key in workflow_pages:
            is_active = current_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state["nav_page"] = page_key
                st.rerun()

        # 2. Insights Group
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #6E6E73; text-transform: uppercase; letter-spacing: 0.08em; margin: 16px 0 6px 6px;'>Insights</div>", unsafe_allow_html=True)
        insight_pages = [
            ("Risk Analytics", "Risk Analytics"),
            ("Audit History", "Document History"),
            ("Reports", "Reports"),
        ]
        for label, page_key in insight_pages:
            is_active = current_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state["nav_page"] = page_key
                st.rerun()

        # 3. System Group
        st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #6E6E73; text-transform: uppercase; letter-spacing: 0.08em; margin: 16px 0 6px 6px;'>System</div>", unsafe_allow_html=True)
        system_pages = [("Settings", "Settings")]
        if user.get("role") == "admin":
            system_pages.append(("Administration", "Administration"))
        for label, page_key in system_pages:
            is_active = current_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state["nav_page"] = page_key
                st.rerun()

        # Bottom section: User card, single status indicator, logout
        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border-radius: 14px; padding: 14px; margin-bottom: 12px; border: 1px solid rgba(0,0,0,0.08); box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                    <span style="width: 7px; height: 7px; background-color: #248A3D; border-radius: 50%; box-shadow: 0 0 6px #248A3D; display: inline-block;"></span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #248A3D;">Operational</span>
                </div>
                <div style="font-weight: 700; color: #1D1D1F; font-size: 0.88rem;">{user.get('displayName', 'Analyst')}</div>
                <div style="color: #515154; font-size: 0.74rem; word-break: break-all;">{user.get('email', '')}</div>
                <div style="margin-top: 8px;">
                    <span class="badge-low" style="font-size: 0.68rem; padding: 2px 8px;">{user.get('role', 'analyst').upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button("Sync Data", use_container_width=True):
                firebase_service.invalidate_cache()
                st.toast("Portfolio data refreshed", icon="⚡")
                st.rerun()

        with col_sb2:
            if st.button("Logout", use_container_width=True):
                st.session_state["user"] = None
                st.session_state["id_token"] = None
                st.session_state["selected_doc_id"] = None
                st.rerun()

    # Top Bar with page title, breadcrumb, global search, notifications, profile
    page_title_map = {
        "Overview": "Overview",
        "Upload & Analyze": "Ingest & Verify",
        "Verification Results": "Review Console",
        "Document History": "Audit History",
        "Duplicate Detection": "Duplicate Shield",
        "Risk Analytics": "Risk Analytics",
        "Reports": "Reports",
        "Settings": "Settings",
        "Administration": "Administration",
    }
    display_title = page_title_map.get(current_page, current_page)
    components.render_topbar(user, active_section=display_title)

    # Page Router
    if current_page == "Overview":
        dashboard.render_dashboard(user)
    elif current_page == "Upload & Analyze":
        upload.render_upload(user)
    elif current_page == "Verification Results":
        verify.render_verify(user)
    elif current_page == "Document History":
        history.render_history(user)
    elif current_page == "Duplicate Detection":
        duplicate_workspace.render_duplicate_workspace(user)
    elif current_page == "Risk Analytics":
        analytics.render_analytics(user)
    elif current_page == "Reports":
        reports.render_reports(user)
    elif current_page == "Settings":
        settings.render_settings(user)
    elif current_page == "Administration":
        admin.render_admin(user)


# -------------------------------------------------------------------------
# Main Execution Gate
# -------------------------------------------------------------------------

if st.session_state["user"] is None:
    render_auth_screen()
else:
    render_app()
