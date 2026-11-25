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

    # Render modern top navigation ribbon
    components.render_topbar(user)

    # Navigation structure
    nav_mapping = {
        "📊 Overview": "Overview",
        "📤 Ingest & Verify": "Upload & Analyze",
        "🔍 Review Console": "Verification Results",
        "📁 Audit History": "Document History",
        "⚡ Duplicate Shield": "Duplicate Detection",
        "📈 Risk Analytics": "Risk Analytics",
        "📄 Reports": "Reports",
        "⚙️ Settings": "Settings",
    }

    if user.get("role") == "admin":
        nav_mapping["🛡️ Administration"] = "Administration"

    options_keys = list(nav_mapping.keys())
    current_page = st.session_state.get("nav_page", "Overview")
    matching_key = next((k for k, v in nav_mapping.items() if v == current_page), options_keys[0])

    selected_key = st.segmented_control(
        "Navigation Bar",
        options=options_keys,
        default=matching_key,
        label_visibility="collapsed",
    )
    if selected_key and selected_key in nav_mapping:
        selected_page = nav_mapping[selected_key]
        st.session_state["nav_page"] = selected_page
    else:
        selected_page = current_page

    # Sidebar: System telemetry & Quick sync
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 6px 0 12px 0;">
                <h3 style="margin: 0; color: #FFFFFF; font-size: 1.15rem;">⚡ Telemetry</h3>
                <span style="color: #60A5FA; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;">CLOUD RUN US-CENTRAL1</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: #111827; border-radius: 8px; padding: 12px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 0.75rem; color: #94A3B8;">ACTIVE ACCOUNT</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.86rem;">{user.get('displayName', 'Analyst')}</div>
                <div style="color: #64748B; font-size: 0.75rem; word-break: break-all;">{user.get('email', '')}</div>
                <div style="margin-top: 8px;">
                    <span class="badge-low" style="font-size: 0.7rem; padding: 2px 8px;">{user.get('role', 'analyst').upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button("🔄 Sync", use_container_width=True, help="Invalidate cache and fetch real-time fresh records"):
                firebase_service.invalidate_cache()
                st.toast("Portfolio data synchronized with Firestore", icon="⚡")
                st.rerun()

        with col_sb2:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state["user"] = None
                st.session_state["id_token"] = None
                st.session_state["selected_doc_id"] = None
                st.rerun()

        st.markdown(
            """
            <div style="margin-top: 20px; font-size: 0.72rem; color: #64748B; line-height: 1.4;">
                <div>Engine: <b>FinVision v2.1</b></div>
                <div>OCR: <b>PyMuPDF + Tesseract</b></div>
                <div>Storage: <b>GCS Verified</b></div>
                <div>Rules: <b>Deterministic Math</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Page Router
    if selected_page == "Overview":
        dashboard.render_dashboard(user)
    elif selected_page == "Upload & Analyze":
        upload.render_upload(user)
    elif selected_page == "Verification Results":
        verify.render_verify(user)
    elif selected_page == "Document History":
        history.render_history(user)
    elif selected_page == "Duplicate Detection":
        duplicate_workspace.render_duplicate_workspace(user)
    elif selected_page == "Risk Analytics":
        analytics.render_analytics(user)
    elif selected_page == "Reports":
        reports.render_reports(user)
    elif selected_page == "Settings":
        settings.render_settings(user)
    elif selected_page == "Administration":
        admin.render_admin(user)


# -------------------------------------------------------------------------
# Main Execution Gate
# -------------------------------------------------------------------------

if st.session_state["user"] is None:
    render_auth_screen()
else:
    render_app()
