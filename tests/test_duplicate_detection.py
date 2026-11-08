"""Unit tests for FinVision Duplicate Detection Engine."""

import hashlib
import numpy as np
from PIL import Image
import pytest

from src.duplicate_detection import (
    compute_sha256,
    compute_perceptual_hashes,
    compute_phash_hamming_distance,
    normalize_string,
    find_duplicates,
)


def test_sha256_computation():
    """Verify SHA-256 byte hashing."""
    data = b"FinVision Invoice Payload"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_sha256(data) == expected


def test_perceptual_hashes_identical_and_near_identical():
    """Verify pHash Hamming distance is 0 for identical and small (<= 4) for recompressed JPEG."""
    import io

    # Create a realistic document-like image with text shapes
    img1 = Image.new("RGB", (300, 300), color=(255, 255, 255))
    import cv2
    arr = np.array(img1)
    cv2.putText(arr, "INVOICE 1001", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(arr, "Total: $500.00", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    img1 = Image.fromarray(arr)

    h1 = compute_perceptual_hashes(img1)

    # Identical image
    dist_self = compute_phash_hamming_distance(h1["pHash"], h1["pHash"])
    assert dist_self == 0

    # Recompressed JPEG (quality 85)
    buf = io.BytesIO()
    img1.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    img2 = Image.open(buf)
    h2 = compute_perceptual_hashes(img2)
    dist_near = compute_phash_hamming_distance(h1["pHash"], h2["pHash"])
    assert dist_near <= 4


def test_find_duplicates_exact_sha():
    """Verify exact SHA-256 duplicate detection."""
    target = {"docId": "doc1", "sha256": "abc12345"}
    candidate = {"docId": "doc2", "sha256": "abc12345"}

    res = find_duplicates(target, [candidate])
    assert res["hasDuplicate"] is True
    assert res["duplicateMatches"][0]["matchType"] == "EXACT_SHA256"
    assert res["duplicateMatches"][0]["docId"] == "doc2"


def test_find_duplicates_semantic():
    """Verify semantic duplicate detection by (invoiceNumber, vendor)."""
    target = {
        "docId": "doc1",
        "fields": {"invoiceNumber": "INV-8899", "vendor": "Stripe Inc."},
    }
    candidate = {
        "docId": "doc2",
        "fields": {"invoiceNumber": "inv 8899", "vendor": "stripe, inc"},
    }

    res = find_duplicates(target, [candidate])
    assert res["hasDuplicate"] is True
    assert res["duplicateMatches"][0]["matchType"] == "SEMANTIC_INVOICE_VENDOR"


def test_find_duplicates_semantic_vendor_date_total():
    """Verify semantic duplicate detection by (vendor, date, total)."""
    target = {
        "docId": "doc1",
        "fields": {"vendor": "Acme Corp", "date": "2024-03-01", "total": 450.0},
    }
    candidate = {
        "docId": "doc2",
        "fields": {"vendor": "Acme Corp", "date": "2024-03-01", "total": 450.0},
    }

    res = find_duplicates(target, [candidate])
    assert res["hasDuplicate"] is True
    assert res["duplicateMatches"][0]["matchType"] == "SEMANTIC_VENDOR_DATE_TOTAL"
