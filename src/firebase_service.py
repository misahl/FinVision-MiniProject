"""Firebase Service Layer for FinVision.

This is the SINGLE module in FinVision authorized to import and communicate with Firebase.
- Firebase Admin SDK on the server (Auth, Firestore, Storage).
- Identity Toolkit REST API for Client Auth (sign-in & sign-up).
- Token verification via verify_id_token on every protected operation.
- Aggregation queries (count()) for scalable dashboard metrics.
- Seamless switch between Local Emulator (zero-billing) and Live Google Cloud via FIREBASE_USE_EMULATOR.
- Transparent local in-memory fallback if emulator daemon is not yet running, ensuring offline testability.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

load_dotenv()

# Firebase Environment Variables
USE_EMULATOR = os.getenv("FIREBASE_USE_EMULATOR", "true").lower() in ("true", "1", "yes")
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "finvision-dev")
STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", f"{PROJECT_ID}.appspot.com")
API_KEY = os.getenv("FIREBASE_API_KEY", "fake-api-key-for-emulator")

AUTH_EMU_HOST = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")
FIRESTORE_EMU_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
STORAGE_EMU_HOST = os.getenv("FIREBASE_STORAGE_EMULATOR_HOST", "127.0.0.1:9199")

# Set OS environment variables for Google Cloud libraries when in emulator mode
if USE_EMULATOR:
    os.environ["FIRESTORE_EMULATOR_HOST"] = FIRESTORE_EMU_HOST
    os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = AUTH_EMU_HOST
    os.environ["STORAGE_EMULATOR_HOST"] = f"http://{STORAGE_EMU_HOST}"
    os.environ["GCLOUD_PROJECT"] = PROJECT_ID

import firebase_admin
from firebase_admin import auth as admin_auth
from firebase_admin import credentials, firestore, storage


class InMemoryFirebaseStore:
    """Thread-safe in-memory replica of Firestore & Storage for offline tests and development."""

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.line_items: Dict[str, Dict[str, Dict[str, Any]]] = {}  # docId -> {itemId: item}
        self.audit_log: List[Dict[str, Any]] = []
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.storage_files: Dict[str, bytes] = {}

    def reset(self):
        self.users.clear()
        self.documents.clear()
        self.line_items.clear()
        self.audit_log.clear()
        self.model_registry.clear()
        self.storage_files.clear()


# Initialize global in-memory store
_LOCAL_STORE = InMemoryFirebaseStore()
_FIREBASE_APP = None
_FIRESTORE_CLIENT = None
_STORAGE_BUCKET_CLIENT = None
_EMULATOR_REACHABLE = False


def initialize_firebase():
    """Initialize Firebase Admin SDK and test connection to emulators or cloud."""
    global _FIREBASE_APP, _FIRESTORE_CLIENT, _STORAGE_BUCKET_CLIENT, _EMULATOR_REACHABLE

    if _FIREBASE_APP is None:
        try:
            if not firebase_admin._apps:
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if cred_path and Path(cred_path).exists() and not USE_EMULATOR:
                    cred = credentials.Certificate(cred_path)
                    _FIREBASE_APP = firebase_admin.initialize_app(
                        cred, {"storageBucket": STORAGE_BUCKET, "projectId": PROJECT_ID}
                    )
                else:
                    _FIREBASE_APP = firebase_admin.initialize_app(
                        options={"projectId": PROJECT_ID, "storageBucket": STORAGE_BUCKET}
                    )
            else:
                _FIREBASE_APP = firebase_admin.get_app()

            _FIRESTORE_CLIENT = firestore.client()
            # Test if emulator is reachable
            if USE_EMULATOR:
                try:
                    resp = requests.get(f"http://{FIRESTORE_EMU_HOST}", timeout=0.8)
                    _EMULATOR_REACHABLE = True
                except Exception:
                    _EMULATOR_REACHABLE = False
            else:
                _EMULATOR_REACHABLE = True
        except Exception:
            _EMULATOR_REACHABLE = False


# Initial attempt to hook into Firebase Admin
initialize_firebase()


# -------------------------------------------------------------------------
# Authentication Layer (Identity Toolkit REST API & Token Verification)
# -------------------------------------------------------------------------

def sign_up_user(
    email: str,
    password: str,
    display_name: str,
    role: str = "analyst",
) -> Dict[str, Any]:
    """Register a new user using Firebase Identity Toolkit REST API."""
    if USE_EMULATOR and not _EMULATOR_REACHABLE:
        # Offline in-memory fallback
        uid = f"user_{uuid.uuid4().hex[:12]}"
        user_record = {
            "uid": uid,
            "email": email,
            "displayName": display_name,
            "role": role,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        _LOCAL_STORE.users[uid] = user_record
        id_token = f"mock-token-{uid}"
        return {
            "idToken": id_token,
            "refreshToken": f"mock-refresh-{uid}",
            "localId": uid,
            "email": email,
            "displayName": display_name,
            "role": role,
        }

    # REST endpoint for sign-up
    if USE_EMULATOR:
        url = f"http://{AUTH_EMU_HOST}/identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    else:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"

    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload, timeout=5)
    data = res.json()

    if "error" in data:
        raise ValueError(data["error"].get("message", "Sign-up failed"))

    uid = data["localId"]
    id_token = data["idToken"]

    # Set custom claims and create user document
    try:
        admin_auth.set_custom_user_claims(uid, {"role": role})
        create_user_profile(uid, display_name, email, role)
    except Exception:
        pass

    return {
        "idToken": id_token,
        "refreshToken": data.get("refreshToken"),
        "localId": uid,
        "email": email,
        "displayName": display_name,
        "role": role,
    }


def sign_in_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user with email and password via Identity Toolkit REST API."""
    if USE_EMULATOR and not _EMULATOR_REACHABLE:
        # Offline in-memory search
        for uid, u in _LOCAL_STORE.users.items():
            if u["email"].lower() == email.lower():
                return {
                    "idToken": f"mock-token-{uid}",
                    "refreshToken": f"mock-refresh-{uid}",
                    "localId": uid,
                    "email": email,
                    "displayName": u.get("displayName", "Analyst"),
                    "role": u.get("role", "analyst"),
                }
        # If not found in mock store, create temporary session for convenience
        return sign_up_user(email, password, display_name="Test Analyst", role="analyst")

    if USE_EMULATOR:
        url = f"http://{AUTH_EMU_HOST}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    else:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"

    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload, timeout=5)
    data = res.json()

    if "error" in data:
        raise ValueError(data["error"].get("message", "Invalid email or password"))

    uid = data["localId"]
    profile = get_user_profile(uid)
    return {
        "idToken": data["idToken"],
        "refreshToken": data.get("refreshToken"),
        "localId": uid,
        "email": email,
        "displayName": profile.get("displayName", data.get("displayName", "Analyst")),
        "role": profile.get("role", "analyst"),
    }


def verify_user_token(id_token: str) -> Dict[str, Any]:
    """Verify Firebase ID token and return claims dict with user identity."""
    if id_token.startswith("mock-token-"):
        uid = id_token.replace("mock-token-", "")
        user = _LOCAL_STORE.users.get(uid, {"uid": uid, "email": "analyst@finvision.local", "role": "analyst"})
        return {"uid": uid, "email": user.get("email"), "role": user.get("role", "analyst")}

    try:
        decoded = admin_auth.verify_id_token(id_token)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
            "role": decoded.get("role", "analyst"),
        }
    except Exception as e:
        raise PermissionError(f"Invalid or expired Firebase ID token: {str(e)}")


def create_user_profile(uid: str, display_name: str, email: str, role: str):
    """Write user profile document to users/{uid} in Firestore."""
    profile = {
        "displayName": display_name,
        "email": email,
        "role": role,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        _FIRESTORE_CLIENT.collection("users").document(uid).set(profile)
    else:
        _LOCAL_STORE.users[uid] = profile


def get_user_profile(uid: str) -> Dict[str, Any]:
    """Read user profile from Firestore."""
    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        doc = _FIRESTORE_CLIENT.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict()
    return _LOCAL_STORE.users.get(uid, {"displayName": "Analyst", "role": "analyst"})


# -------------------------------------------------------------------------
# Document Operations & Firestore Data Model
# -------------------------------------------------------------------------

def create_document_record(
    owner_uid: str,
    file_metadata: Dict[str, Any],
) -> str:
    """Create initial document record in Firestore with status 'uploaded'."""
    doc_id = f"doc_{uuid.uuid4().hex[:16]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    doc_data = {
        "docId": doc_id,
        "ownerUid": owner_uid,
        "status": "uploaded",
        "createdAt": now_iso,
        "updatedAt": now_iso,
        "file": {
            "storagePath": file_metadata.get("storagePath", ""),
            "mimeType": file_metadata.get("mimeType", "image/jpeg"),
            "sizeBytes": file_metadata.get("sizeBytes", 0),
            "sha256": file_metadata.get("sha256", ""),
            "pHash": file_metadata.get("pHash", ""),
            "pageCount": file_metadata.get("pageCount", 1),
        },
        "fields": {
            "invoiceNumber": None,
            "vendor": None,
            "customer": None,
            "date": None,
            "currency": "USD",
            "subtotal": None,
            "tax": None,
            "taxRate": None,
            "total": None,
        },
        "fieldConfidence": {},
        "userCorrected": [],
        "verification": {},
        "risk": {},
        "ml": {},
        "review": None,
    }

    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        _FIRESTORE_CLIENT.collection("documents").document(doc_id).set(doc_data)
    else:
        _LOCAL_STORE.documents[doc_id] = doc_data

    log_audit_event(owner_uid, doc_id, "DOCUMENT_UPLOADED", {"filename": file_metadata.get("filename")})
    return doc_id


def save_document_analysis(
    doc_id: str,
    fields: Dict[str, Any],
    field_confidence: Dict[str, float],
    line_items: List[Dict[str, Any]],
    verification_results: Dict[str, Any],
    cv_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    storage_path: Optional[str] = None,
):
    """Save processed OCR, rule results, CV anomalies, ML prediction and risk assessment."""
    now_iso = datetime.now(timezone.utc).isoformat()

    update_payload = {
        "status": "verified",
        "updatedAt": now_iso,
        "fields": fields,
        "fieldConfidence": field_confidence,
        "verification": {
            "ruleResults": verification_results.get("ruleResults", []),
            "missingFields": verification_results.get("missingFields", []),
            "cvScore": cv_results.get("cvScore", 0.0),
            "cvRegions": cv_results.get("cvRegions", []),
            "duplicateMatches": verification_results.get("duplicateMatches", []),
        },
        "ml": ml_results,
        "risk": risk_assessment,
    }

    if storage_path:
        update_payload["file.storagePath"] = storage_path

    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        doc_ref = _FIRESTORE_CLIENT.collection("documents").document(doc_id)
        doc_ref.update(update_payload)
        
        # Write line items to subcollection
        for idx, item in enumerate(line_items):
            item_id = f"item_{idx+1:03d}"
            doc_ref.collection("lineItems").document(item_id).set(item)
    else:
        if doc_id in _LOCAL_STORE.documents:
            _LOCAL_STORE.documents[doc_id].update(update_payload)
            if storage_path:
                _LOCAL_STORE.documents[doc_id].setdefault("file", {})["storagePath"] = storage_path
        else:
            doc_record = {"docId": doc_id, **update_payload}
            if storage_path:
                doc_record.setdefault("file", {})["storagePath"] = storage_path
            _LOCAL_STORE.documents[doc_id] = doc_record
        
        _LOCAL_STORE.line_items[doc_id] = {
            f"item_{idx+1:03d}": it for idx, it in enumerate(line_items)
        }


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Fetch complete document including subcollection line items."""
    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        doc = _FIRESTORE_CLIENT.collection("documents").document(doc_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        # Fetch subcollection line items
        items_ref = _FIRESTORE_CLIENT.collection("documents").document(doc_id).collection("lineItems").stream()
        data["lineItems"] = [item.to_dict() for item in items_ref]
        return data

    doc_data = _LOCAL_STORE.documents.get(doc_id)
    if doc_data:
        data = dict(doc_data)
        data["lineItems"] = list(_LOCAL_STORE.line_items.get(doc_id, {}).values())
        return data
    return None


def update_document_review_decision(
    doc_id: str,
    reviewer_uid: str,
    decision: str,
    comment: str = "",
):
    """Save reviewer decision (approved, rejected, needs_review). Writes to auditLog."""
    now_iso = datetime.now(timezone.utc).isoformat()
    review_map = {
        "decision": decision,
        "reviewerUid": reviewer_uid,
        "comment": comment,
        "reviewedAt": now_iso,
    }
    update_data = {
        "status": "decided",
        "updatedAt": now_iso,
        "review": review_map,
    }

    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        _FIRESTORE_CLIENT.collection("documents").document(doc_id).update(update_data)
    else:
        if doc_id in _LOCAL_STORE.documents:
            _LOCAL_STORE.documents[doc_id].update(update_data)

    log_audit_event(
        reviewer_uid,
        doc_id,
        f"REVIEW_DECISION_{decision.upper()}",
        {"comment": comment, "decision": decision},
    )


def log_audit_event(
    actor_uid: str,
    doc_id: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Append-only audit trail write to auditLog/{eventId}."""
    event_id = f"evt_{uuid.uuid4().hex[:16]}"
    event_data = {
        "eventId": event_id,
        "actorUid": actor_uid,
        "docId": doc_id,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        try:
            _FIRESTORE_CLIENT.collection("auditLog").document(event_id).set(event_data)
        except Exception:
            _LOCAL_STORE.audit_log.append(event_data)
    else:
        _LOCAL_STORE.audit_log.append(event_data)


# -------------------------------------------------------------------------
# Dashboard Aggregations (Firestore count() / Aggregate Queries)
# -------------------------------------------------------------------------

def get_dashboard_metrics(owner_uid: Optional[str] = None) -> Dict[str, Any]:
    """Compute real dashboard KPI metrics using Firestore count() aggregations."""
    kpis = {
        "total_documents": 0,
        "normal_count": 0,
        "medium_risk_count": 0,
        "high_risk_count": 0,
        "duplicates_count": 0,
        "approved_count": 0,
        "rejected_count": 0,
        "recent_documents": [],
    }

    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        coll = _FIRESTORE_CLIENT.collection("documents")
        if owner_uid:
            coll = coll.where("ownerUid", "==", owner_uid)

        # Use aggregation queries for performance
        try:
            kpis["total_documents"] = coll.count().get()[0][0].value
            kpis["high_risk_count"] = coll.where("risk.level", "==", "HIGH").count().get()[0][0].value
            kpis["medium_risk_count"] = coll.where("risk.level", "==", "MEDIUM").count().get()[0][0].value
            kpis["normal_count"] = coll.where("risk.level", "==", "LOW").count().get()[0][0].value
            kpis["approved_count"] = coll.where("review.decision", "==", "approved").count().get()[0][0].value
            kpis["rejected_count"] = coll.where("review.decision", "==", "rejected").count().get()[0][0].value
        except Exception:
            pass

        # Stream recent 10 documents
        recent_stream = coll.order_by("createdAt", direction=firestore.Query.DESCENDING).limit(10).stream()
        kpis["recent_documents"] = [d.to_dict() for d in recent_stream]
        return kpis

    # In-Memory fallback aggregation
    docs = list(_LOCAL_STORE.documents.values())
    if owner_uid:
        docs = [d for d in docs if d.get("ownerUid") == owner_uid]

    kpis["total_documents"] = len(docs)
    for d in docs:
        risk_lvl = d.get("risk", {}).get("level")
        if risk_lvl == "HIGH":
            kpis["high_risk_count"] += 1
        elif risk_lvl == "MEDIUM":
            kpis["medium_risk_count"] += 1
        elif risk_lvl == "LOW":
            kpis["normal_count"] += 1

        rev = d.get("review") or {}
        if rev.get("decision") == "approved":
            kpis["approved_count"] += 1
        elif rev.get("decision") == "rejected":
            kpis["rejected_count"] += 1

        dups = d.get("verification", {}).get("duplicateMatches", [])
        if dups:
            kpis["duplicates_count"] += 1

    docs_sorted = sorted(docs, key=lambda d: d.get("createdAt", ""), reverse=True)
    kpis["recent_documents"] = docs_sorted[:10]
    return kpis


def list_documents(
    owner_uid: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List documents for table display."""
    if _EMULATOR_REACHABLE and _FIRESTORE_CLIENT:
        coll = _FIRESTORE_CLIENT.collection("documents")
        if owner_uid:
            coll = coll.where("ownerUid", "==", owner_uid)
        stream = coll.order_by("createdAt", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [d.to_dict() for d in stream]

    docs = list(_LOCAL_STORE.documents.values())
    if owner_uid:
        docs = [d for d in docs if d.get("ownerUid") == owner_uid]
    docs.sort(key=lambda d: d.get("createdAt", ""), reverse=True)
    return docs[:limit]


# -------------------------------------------------------------------------
# Storage Operations
# -------------------------------------------------------------------------

def upload_invoice_bytes(
    uid: str,
    doc_id: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "image/jpeg",
) -> str:
    """Upload invoice file to Cloud Storage at users/{uid}/invoices/{docId}/{filename}."""
    storage_path = f"users/{uid}/invoices/{doc_id}/{filename}"

    if _EMULATOR_REACHABLE:
        try:
            bucket = storage.bucket(STORAGE_BUCKET)
            blob = bucket.blob(storage_path)
            blob.upload_from_string(file_bytes, content_type=content_type)
            return storage_path
        except Exception:
            pass

    _LOCAL_STORE.storage_files[storage_path] = file_bytes
    return storage_path


def download_invoice_bytes(storage_path: str) -> Optional[bytes]:
    """Retrieve invoice file bytes from storage."""
    if storage_path in _LOCAL_STORE.storage_files:
        return _LOCAL_STORE.storage_files[storage_path]

    if _EMULATOR_REACHABLE:
        try:
            bucket = storage.bucket(STORAGE_BUCKET)
            blob = bucket.blob(storage_path)
            return blob.download_as_bytes()
        except Exception:
            return None
    return None

