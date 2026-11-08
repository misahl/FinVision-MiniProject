"""FinVision: Intelligent Financial Document Verification.

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
    analytics,
    reports,
    settings,
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
    st.session_state["nav_page"] = "Dashboard"


def render_auth_screen():
    """Render modern login and registration interface."""
    st.markdown("<div style='text-align: center; margin-top: 40px;'>", unsafe_allow_html=True)
    st.markdown("# 🛡️ FinVision")
    st.markdown("### Intelligent Financial Document Verification System")
    st.markdown(
        "<p style='color: #64748B; max-width: 500px; margin: 0 auto;'>"
        "Firebase-native AI/ML invoice verification with explainable computer vision anomaly detection "
        "and deterministic financial rules."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_space1, col_center, col_space2 = st.columns([1, 1.2, 1])

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

            # Quick 1-Click Demo Login
            st.markdown("<div style='text-align: center; margin-top: 10px;'>", unsafe_allow_html=True)
            if st.button("⚡ Quick Demo Analyst Sign-In", use_container_width=True):
                auth_res = firebase_service.sign_up_user(
                    email="demo_analyst@finvision.test",
                    password="DemoPassword123!",
                    display_name="Demo Financial Analyst",
                    role="analyst",
                )
                st.session_state["user"] = {
                    "uid": auth_res["localId"],
                    "email": auth_res["email"],
                    "displayName": auth_res.get("displayName", "Demo Analyst"),
                    "role": "analyst",
                }
                st.session_state["id_token"] = auth_res["idToken"]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_register:
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", value="Alex Rivera")
                reg_email = st.text_input("Email", value="")
                reg_pass = st.text_input("Password (min 8 chars)", type="password")
                reg_role = st.selectbox("Role", options=["analyst", "admin"])
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
            <div style="padding: 10px 0 20px 0;">
                <h2 style="margin: 0; color: #FFFFFF; font-size: 1.4rem;">🛡️ FinVision</h2>
                <span style="color: #60A5FA; font-size: 0.8rem; font-weight: 500;">DOCUMENT VERIFICATION</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background-color: #1E293B; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
                <div style="font-weight: 600; color: #F8FAFC; font-size: 0.9rem;">{user.get('displayName', 'Analyst')}</div>
                <div style="color: #94A3B8; font-size: 0.78rem;">{user.get('email', '')}</div>
                <div style="margin-top: 6px;"><span class="badge-low">{user.get('role', 'analyst').upper()}</span></div>
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

        # Sync nav selection
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
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["user"] = None
            st.session_state["id_token"] = None
            st.session_state["selected_doc_id"] = None
            st.rerun()

    # Page Router
    if selected_page == "Dashboard":
        dashboard.render_dashboard(user)
    elif selected_page == "Upload Invoice":
        upload.render_upload(user)
    elif selected_page == "Verify Documents":
        verify.render_verify(user)
    elif selected_page == "Document History":
        history.render_history(user)
    elif selected_page == "Analytics":
        analytics.render_analytics(user)
    elif selected_page == "Reports":
        reports.render_reports(user)
    elif selected_page == "Settings":
        settings.render_settings(user)


# -------------------------------------------------------------------------
# Main Execution Gate
# -------------------------------------------------------------------------

if st.session_state["user"] is None:
    render_auth_screen()
else:
    render_app()

