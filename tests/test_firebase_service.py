"""Unit tests for FinVision Firebase Service Layer."""

import pytest
from src.firebase_service import (
    sign_up_user,
    sign_in_user,
    verify_user_token,
    create_document_record,
    save_document_analysis,
    get_document,
    update_document_review_decision,
    get_dashboard_metrics,
    upload_invoice_bytes,
    download_invoice_bytes,
    _LOCAL_STORE,
)


@pytest.fixture(autouse=True)
def clean_store():
    """Ensure clean store before each test."""
    _LOCAL_STORE.reset()
    yield
    _LOCAL_STORE.reset()


def test_auth_registration_and_token_verification():
    """Verify sign-up, sign-in and token claims."""
    user = sign_up_user(
        email="test_analyst@finvision.test",
        password="ValidPassword123!",
        display_name="Jane Doe",
        role="analyst",
    )
    assert user["email"] == "test_analyst@finvision.test"
    assert "idToken" in user

    # Token verification
    claims = verify_user_token(user["idToken"])
    assert claims["uid"] == user["localId"]
    assert claims["role"] == "analyst"


def test_document_lifecycle_and_analysis_persistence():
    """Verify document upload, analysis update, and review lifecycle."""
    owner_uid = "user_test_123"
    meta = {
        "filename": "invoice_001.jpg",
        "sha256": "abc12345",
        "pHash": "fedcba98",
        "mimeType": "image/jpeg",
        "sizeBytes": 10240,
    }

    # 1. Create document
    doc_id = create_document_record(owner_uid, meta)
    assert doc_id.startswith("doc_")

    doc = get_document(doc_id)
    assert doc["status"] == "uploaded"
    assert doc["ownerUid"] == owner_uid

    # 2. Save analysis
    fields = {"invoiceNumber": "INV-100", "total": 250.0, "vendor": "Acme"}
    line_items = [{"description": "Service A", "quantity": 1.0, "unitPrice": 250.0, "lineTotal": 250.0}]
    verification = {"ruleResults": [], "missingFields": [], "duplicateMatches": []}
    cv_res = {"cvScore": 5.0, "cvRegions": []}
    ml_res = {"prediction": 0, "probability": 0.05, "experimental": True}
    risk_res = {"level": "LOW", "score": 5.0, "reasons": [], "recommendedAction": "Approve"}

    save_document_analysis(doc_id, fields, {}, line_items, verification, cv_res, ml_res, risk_res)

    updated_doc = get_document(doc_id)
    assert updated_doc["status"] == "verified"
    assert updated_doc["risk"]["level"] == "LOW"
    assert len(updated_doc["lineItems"]) == 1

    # 3. Save review decision
    update_document_review_decision(doc_id, owner_uid, "approved", "Looks good")
    reviewed_doc = get_document(doc_id)
    assert reviewed_doc["status"] == "decided"
    assert reviewed_doc["review"]["decision"] == "approved"


def test_dashboard_kpis_aggregation():
    """Verify dashboard KPI metrics reflect persisted documents."""
    owner = "user_kpi_test"
    meta = {"filename": "test.jpg", "sizeBytes": 100}

    d1 = create_document_record(owner, meta)
    save_document_analysis(d1, {}, {}, [], {}, {}, {}, {"level": "LOW"})

    d2 = create_document_record(owner, meta)
    save_document_analysis(d2, {}, {}, [], {}, {}, {}, {"level": "HIGH"})

    kpis = get_dashboard_metrics(owner)
    assert kpis["total_documents"] == 2
    assert kpis["normal_count"] == 1
    assert kpis["high_risk_count"] == 1


def test_storage_upload_and_download():
    """Verify file storage round-trip."""
    raw_payload = b"Sample Invoice Image Data"
    path = upload_invoice_bytes("user_1", "doc_1", "invoice.jpg", raw_payload)
    downloaded = download_invoice_bytes(path)
    assert downloaded == raw_payload

