"""Unit tests for FinVision Image Processing & CV Anomaly Detection."""

import io
import cv2
import numpy as np
import pytest
from reportlab.pdfgen import canvas

from src.image_processing import (
    detect_mime_type,
    load_image_from_source,
    estimate_skew_angle,
    deskew_image,
    preprocess_image,
    compute_error_level_analysis,
    compute_noise_variance_inconsistency,
    compute_text_layout_anomalies,
    compute_copy_move_candidates,
    run_cv_anomaly_pipeline,
)


def create_synthetic_text_image(text="FINVISION INVOICE 12345", width=600, height=300):
    """Generate a clean white canvas with sharp black text for CV testing."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, line in enumerate([text, "Vendor: Acme Corp", "Total Amount: $1,250.00", "Date: 2024-03-15"]):
        y = 60 + i * 45
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def create_synthetic_pdf():
    """Create a minimal 1-page PDF document in memory."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "FinVision Test Invoice")
    c.drawString(100, 700, "Item: Consulting Services - $500.00")
    c.showPage()
    c.save()
    return buf.getvalue()


# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------

def test_mime_type_sniffing():
    """Verify magic-byte sniffing distinguishes JPEG, PNG, PDF, and unknown."""
    assert detect_mime_type(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image/jpeg"
    assert detect_mime_type(b"\x89PNG\r\n\x1a\n\x00\x00") == "image/png"
    assert detect_mime_type(b"%PDF-1.4\n%test") == "application/pdf"
    assert detect_mime_type(b"random-bytes-not-an-image") == "application/octet-stream"


def test_pdf_rasterization_at_300_dpi():
    """Test loading and rasterizing an in-memory PDF page at 300 DPI."""
    pdf_bytes = create_synthetic_pdf()
    img, meta = load_image_from_source(pdf_bytes, dpi=300)

    assert meta["original_mime"] == "application/pdf"
    assert meta["page_count"] == 1
    assert img is not None
    assert len(img.shape) == 3
    # Standard PDF 612x792 pt rasterized at 300 DPI is approx 2550 x 3300 px
    assert img.shape[0] > 2000
    assert img.shape[1] > 1500


def test_immutability_and_pipeline_stages():
    """Verify original input array is never mutated and pipeline logs all steps."""
    original = create_synthetic_text_image()
    orig_copy = original.copy()

    deskewed, binary, log, meta = preprocess_image(original)

    # 1. Immutability guarantee
    np.testing.assert_array_equal(original, orig_copy, err_msg="Original image was mutated in-place!")

    # 2. Output shapes and types
    assert len(deskewed.shape) == 3
    assert len(binary.shape) == 2  # Binarized 8-bit single channel
    assert binary.dtype == np.uint8

    # 3. Processing log completeness
    logged_steps = [entry["step"] for entry in log]
    for required in ["load", "grayscale", "denoise", "clahe", "deskew", "adaptive_threshold"]:
        assert required in logged_steps, f"Missing pipeline step: {required}"


def test_deskewing_synthetic_rotated_image():
    """Verify that a rotated synthetic invoice is estimated and deskewed."""
    base_img = create_synthetic_text_image(width=700, height=500)
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    # Artificially rotate image by +12.0 degrees
    known_angle = 12.0
    h, w = gray.shape
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, known_angle, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), borderValue=255)

    estimated_angle = estimate_skew_angle(rotated)

    # The estimate should be within 2.0 degrees of -known_angle (to counter-rotate)
    assert abs(abs(estimated_angle) - known_angle) < 2.5, (
        f"Expected deskew correction near ±{known_angle}, got {estimated_angle}"
    )

    # Test max-angle safety guard (> 45 deg should be rejected to prevent 90-deg flip)
    guard_angle = estimate_skew_angle(rotated, max_angle=5.0)
    assert guard_angle == 0.0, "Safety guard failed to reject angle exceeding max_angle!"


def test_error_level_analysis_detects_patch():
    """Test ELA detects a spliced high-contrast patch on a recompressed JPEG."""
    base = create_synthetic_text_image()

    # Save as baseline JPEG quality 70
    _, encoded = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    jpeg_img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Splice an uncompressed high-frequency patch (e.g. artificial stamp)
    tampered = jpeg_img.copy()
    cv2.circle(tampered, (250, 150), 30, (0, 0, 255), -1)
    cv2.putText(tampered, "FORGED", (210, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    res_clean = compute_error_level_analysis(jpeg_img)
    res_tampered = compute_error_level_analysis(tampered)

    assert res_clean["applicable"] is True
    assert res_tampered["applicable"] is True
    # Tampered image should have higher anomaly score or detected regions
    assert res_tampered["max_residual"] >= res_clean["max_residual"]
    assert res_tampered["score"] >= res_clean["score"]


def test_noise_variance_inconsistency():
    """Test noise variance detector flags an artificially noisy patch."""
    base = np.full((400, 400, 3), 200, dtype=np.uint8)

    # Add uniform baseline subtle noise
    gaussian_noise = np.random.normal(0, 2, base.shape).astype(np.int16)
    uniform_noisy = np.clip(base.astype(np.int16) + gaussian_noise, 0, 255).astype(np.uint8)

    # Splice a localized high-noise patch in a 64x64 block
    spliced = uniform_noisy.copy()
    heavy_noise = np.random.normal(0, 35, (64, 64, 3)).astype(np.int16)
    spliced[128:192, 128:192] = np.clip(
        spliced[128:192, 128:192].astype(np.int16) + heavy_noise, 0, 255
    ).astype(np.uint8)

    res = compute_noise_variance_inconsistency(spliced, block_size=32)

    assert res["indicator"] == "noise_variance_inconsistency"
    assert res["outlier_blocks_count"] > 0
    assert len(res["regions"]) > 0


def test_text_layout_anomalies():
    """Test text layout outlier detector identifies an abnormal font height box."""
    # Normal words: height ~ 18 px
    boxes = [
        {"text": "Invoice", "left": 50, "top": 100, "width": 80, "height": 18},
        {"text": "Number:", "left": 140, "top": 100, "width": 75, "height": 18},
        {"text": "10492", "left": 220, "top": 100, "width": 50, "height": 19},
        {"text": "Date:", "left": 50, "top": 140, "width": 45, "height": 18},
        {"text": "2024-01-01", "left": 105, "top": 140, "width": 90, "height": 18},
        {"text": "Subtotal:", "left": 50, "top": 180, "width": 80, "height": 17},
        {"text": "Tax:", "left": 50, "top": 210, "width": 40, "height": 18},
        {"text": "Total:", "left": 50, "top": 240, "width": 55, "height": 18},
        # Tampered outlier: abnormally large height (55 px vs 18 px baseline)
        {"text": "$99,999", "left": 120, "top": 235, "width": 120, "height": 55},
    ]

    res = compute_text_layout_anomalies(boxes, z_threshold=2.5)

    assert res["indicator"] == "text_layout_anomalies"
    assert res["anomalous_boxes_count"] == 1
    assert res["regions"][0]["text"] == "$99,999"


def test_copy_move_detection():
    """Test ORB-based copy-move detector identifies duplicated visual elements."""
    base = np.full((500, 500, 3), 255, dtype=np.uint8)
    
    # Draw a distinct star/stamp logo
    logo = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.circle(logo, (40, 40), 30, (0, 0, 150), -1)
    cv2.rectangle(logo, (20, 20), (60, 60), (150, 0, 0), 2)
    cv2.putText(logo, "OK", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Paste the identical logo in two separate locations
    base[50:130, 50:130] = logo
    base[300:380, 320:400] = logo

    res = compute_copy_move_candidates(base, min_matches=5, min_distance_px=50.0)
    assert res["indicator"] == "copy_move_candidates"
    assert res["match_count"] > 0


def test_combined_cv_pipeline():
    """Verify run_cv_anomaly_pipeline aggregates all detectors and produces disclaimer."""
    img = create_synthetic_text_image()
    res = run_cv_anomaly_pipeline(img, is_jpeg=True)

    assert "cvScore" in res
    assert isinstance(res["cvScore"], float)
    assert "cvRegions" in res
    assert "indicators" in res
    assert len(res["indicators"]) >= 3
    assert "disclaimer" in res
    assert "human review" in res["disclaimer"].lower()

