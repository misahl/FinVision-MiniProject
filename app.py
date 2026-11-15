"""FinVision: Intelligent Financial Document Verification System.

Streamlit application entry point, authentication gate, and page router.
"""

from pathlib import Path
import streamlit as st

from src import firebase_service
from frontend import (
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

def render_app():
    user = st.session_state["user"]

    # Verify ID token validity on every request
    try:
        firebase_service.verify_user_token(st.session_state["id_token"])
    except Exception:
        st.session_state["user"] = None
        st.session_state["id_token"] = None
        st.warning("Your session has expired. Please log in again.")
        st.rerun()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0;">
                <h2 style="margin: 0; color: #FFFFFF; font-size: 1.35rem;">🛡️ FinVision</h2>
                <span style="color: #60A5FA; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;">VERIFICATION PLATFORM</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background-color: #1E293B; border-radius: 8px; padding: 12px; margin-bottom: 16px; border: 1px solid #334155;">
                <div style="font-weight: 600; color: #F8FAFC; font-size: 0.88rem;">{user.get('displayName', 'Analyst')}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">{user.get('email', '')}</div>
                <div style="margin-top: 6px;"><span class="badge-low">{user.get('role', 'analyst').upper()}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "Overview",
            "Upload & Analyze",
            "Verification Results",
            "Document History",
            "Duplicate Detection",
            "Risk Analytics",
            "Reports",
            "Settings",
        ]

        # Enforce RBAC: Administration only visible for administrators
        if user.get("role") == "admin":
            nav_options.append("Administration")

        # Sync nav selection
        current_page = st.session_state.get("nav_page", "Overview")
        if current_page not in nav_options:
            current_page = "Overview"

        selected_page = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=nav_options.index(current_page),
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["user"] = None
            st.session_state["id_token"] = None
            st.session_state["selected_doc_id"] = None
            st.rerun()

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
