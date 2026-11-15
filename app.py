"""FinVision: Intelligent Financial Document Verification.

Streamlit application entry point, authentication gate, and page router.
"""

from html import escape
from pathlib import Path
import os

import streamlit as st

from src import firebase_service
from frontend import dashboard, upload, verify, history, analytics, reports, settings


st.set_page_config(
    page_title="FinVision | Financial Document Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "frontend" / "styles.css"
if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

for key, default in {
    "user": None,
    "id_token": None,
    "nav_page": "Dashboard",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _establish_session(auth_res: dict) -> None:
    """Create a session using the verified Firebase token claims."""
    claims = firebase_service.verify_user_token(auth_res["idToken"])
    st.session_state["user"] = {
        "uid": claims["uid"],
        "email": claims.get("email") or auth_res.get("email", ""),
        "displayName": auth_res.get("displayName", "Analyst"),
        # Privileges come from verified server-side claims, never a form field.
        "role": claims.get("role", "analyst"),
    }
    st.session_state["id_token"] = auth_res["idToken"]


def render_auth_screen():
    """Render the public sign-in and analyst registration interface."""
    st.markdown(
        """
        <div class="fv-auth-intro">
          <div class="fv-auth-mark">FV</div>
          <div class="fv-eyebrow">FINANCIAL DOCUMENT INTELLIGENCE</div>
          <h1>Clarity behind every invoice.</h1>
          <p>Extract, verify and review financial documents with an explainable workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        login_tab, register_tab = st.tabs(["Sign in", "Create analyst account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Work email", placeholder="name@company.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button(
                    "Sign in to FinVision", type="primary", use_container_width=True
                )
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter your work email and password.")
                else:
                    try:
                        auth_res = firebase_service.sign_in_user(email.strip(), password)
                        _establish_session(auth_res)
                        st.rerun()
                    except Exception:
                        st.error("Sign-in failed. Check your credentials or contact your workspace administrator.")

            # Demo access is opt-in and must use an account provisioned by the owner.
            if os.getenv("FINVISION_ENABLE_DEMO_LOGIN", "false").lower() in {"1", "true", "yes"}:
                st.caption("Demonstration access")
                with st.form("demo_login_form"):
                    demo_email = st.text_input("Demo account email", key="demo_email")
                    demo_password = st.text_input("Demo account password", type="password", key="demo_password")
                    demo_submitted = st.form_submit_button("Sign in to demo workspace", use_container_width=True)
                if demo_submitted:
                    try:
                        auth_res = firebase_service.sign_in_user(demo_email.strip(), demo_password)
                        _establish_session(auth_res)
                        st.rerun()
                    except Exception:
                        st.error("Demo sign-in failed. Please contact the demonstration administrator.")

        with register_tab:
            st.caption("New accounts receive analyst access. Administrator privileges are assigned separately by the workspace owner.")
            with st.form("register_form"):
                reg_name = st.text_input("Full name", placeholder="Your name")
                reg_email = st.text_input("Work email", placeholder="name@company.com")
                reg_password = st.text_input("Password (minimum 8 characters)", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                register = st.form_submit_button(
                    "Create analyst account", type="primary", use_container_width=True
                )

            if register:
                if not reg_name.strip() or not reg_email.strip():
                    st.error("Enter your name and work email.")
                elif len(reg_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                elif reg_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    try:
                        auth_res = firebase_service.sign_up_user(
                            reg_email.strip(),
                            reg_password,
                            reg_name.strip(),
                            role="analyst",
                        )
                        _establish_session(auth_res)
                        st.success("Your analyst account has been created.")
                        st.rerun()
                    except Exception:
                        st.error("Account creation failed. Check the email address or contact your workspace administrator.")


def render_app():
    user = st.session_state["user"]

    try:
        claims = firebase_service.verify_user_token(st.session_state["id_token"])
        if claims.get("uid") != user.get("uid"):
            raise PermissionError("Session identity mismatch")
        # Refresh the role from verified claims on every rerun.
        user["role"] = claims.get("role", "analyst")
    except Exception:
        st.session_state["user"] = None
        st.session_state["id_token"] = None
        st.warning("Your session has expired. Please sign in again.")
        st.rerun()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0;">
              <h2 style="margin:0;color:#FFFFFF;font-size:1.4rem;">🛡️ FinVision</h2>
              <span style="color:#93B4DD;font-size:.75rem;font-weight:650;letter-spacing:.08em;">DOCUMENT INTELLIGENCE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        safe_name = escape(str(user.get("displayName", "Analyst")))
        safe_email = escape(str(user.get("email", "")))
        safe_role = escape(str(user.get("role", "analyst")).upper())
        st.markdown(
            f"""
            <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:13px;margin-bottom:18px;">
              <div style="font-weight:650;color:#F8FAFC;font-size:.9rem;">{safe_name}</div>
              <div style="color:#AAB8CC;font-size:.76rem;overflow-wrap:anywhere;">{safe_email}</div>
              <div style="margin-top:8px;"><span class="badge-low">{safe_role}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "Dashboard",
            "Upload Invoice",
            "Verify Documents",
            "Document History",
            "Analytics",
            "Reports",
            "Settings",
        ]
        current_page = st.session_state.get("nav_page", "Dashboard")
        if current_page not in nav_options:
            current_page = "Dashboard"
        selected_page = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(current_page),
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("---")
        if st.button("Sign out", use_container_width=True):
            for key in ("user", "id_token", "selected_doc_id"):
                st.session_state[key] = None
            st.rerun()

    page_map = {
        "Dashboard": dashboard.render_dashboard,
        "Upload Invoice": upload.render_upload,
        "Verify Documents": verify.render_verify,
        "Document History": history.render_history,
        "Analytics": analytics.render_analytics,
        "Reports": reports.render_reports,
        "Settings": settings.render_settings,
    }
    page_map[selected_page](user)


if st.session_state["user"] is None:
    render_auth_screen()
else:
    render_app()
