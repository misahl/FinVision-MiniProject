"""Image Processing & Computer Vision Pipeline for FinVision.

This module provides:
1. Input validation & PDF-to-image rasterization (300 DPI).
2. Clean, reproducible preprocessing pipeline (Grayscale -> Denoise -> CLAHE -> Deskew -> Adaptive Threshold).
3. Explainable CV anomaly indicators (Error Level Analysis, Local Noise Variance Inconsistency,
   Text layout/spacing anomalies, Copy-Move candidates).

Zero Firebase dependencies: purely functional and unit-testable offline.
Original image data is strictly immutable (never modified in-place).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pymupdf  # Modern PyMuPDF import


# -------------------------------------------------------------------------
# 1. Image Loading, Conversion & Sniffing
# -------------------------------------------------------------------------

def detect_mime_type(data: bytes) -> str:
    """Sniff magic bytes to identify file format reliably.
    
    Do not trust file extensions alone.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif data.startswith(b"%PDF"):
        return "application/pdf"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def load_image_from_source(
    source: Union[str, Path, bytes, np.ndarray],
    pdf_page: int = 0,
    dpi: int = 300,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load an image from a file path, raw bytes, or existing numpy array.
    
    If given a PDF, rasterizes the requested page at specified DPI (default 300).
    Returns (BGR image array, metadata dict).
    """
    metadata: Dict[str, Any] = {
        "original_mime": "unknown",
        "page_count": 1,
        "selected_page": pdf_page,
    }

    if isinstance(source, np.ndarray):
        img = source.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        metadata["original_mime"] = "image/numpy"
        metadata["width"] = img.shape[1]
        metadata["height"] = img.shape[0]
        return img, metadata

    # If file path, read bytes
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        raw_bytes = source_path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    mime = detect_mime_type(raw_bytes)
    metadata["original_mime"] = mime

    if mime == "application/pdf":
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        metadata["page_count"] = len(doc)
        if len(doc) == 0:
            raise ValueError("PDF document contains no pages.")
        page_index = max(0, min(pdf_page, len(doc) - 1))
        page = doc[page_index]
        
        # Standard PDF is 72 DPI. Target DPI (300) requires zoom = 300 / 72 ≈ 4.1667
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to numpy BGR array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
    else:
        # Decode image bytes using OpenCV
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes into OpenCV matrix.")

    metadata["width"] = img.shape[1]
    metadata["height"] = img.shape[0]
    return img, metadata


# -------------------------------------------------------------------------
# 2. Deskewing Heuristic with Max Angle Safety Guard
# -------------------------------------------------------------------------

def estimate_skew_angle(gray_image: np.ndarray, max_angle: float = 45.0) -> float:
    """Estimate skew angle in degrees using text contour orientation.
    
    A safety guard enforces |angle| <= max_angle so that valid vertical
    receipts or margin lines are not incorrectly flipped 90 degrees.
    """
    # Invert image: text becomes white on black background
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Dilate horizontally to join words into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find non-zero points (text pixels)
    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) < 50:
        return 0.0

    # minAreaRect gives ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize OpenCV angle conventions
    if angle < -45.0:
        angle = -(90.0 + angle)
    elif angle > 45.0:
        angle = 90.0 - angle
    else:
        angle = -angle

    # Enforce safety guard
    if abs(angle) > max_angle:
        return 0.0

    return float(angle)


def deskew_image(
    image: np.ndarray,
    angle: float,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Rotate image by the given angle around its center with white border padding."""
    if abs(angle) < 0.2:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions to avoid clipping
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    border_val = 255 if len(image.shape) == 2 else border_color
    rotated = cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


# -------------------------------------------------------------------------
# 3. Main Preprocessing Pipeline
# -------------------------------------------------------------------------

def preprocess_image(
    source: Union[str, Path, bytes, np.ndarray],
    pdf_page: int = 0,
    dpi: int = 300,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run full OpenCV preprocessing pipeline on document image.
    
    Steps:
      1. Load & format normalization (immutable copy).
      2. Grayscale conversion.
      3. Denoising (bilateral filter preserves sharp text edges while removing noise).
      4. Contrast Limited Adaptive Histogram Equalization (CLAHE).
      5. Deskewing with max-angle protection.
      6. Adaptive binarization / thresholding.

    Returns:
      (deskewed_color_image, binarized_ocr_ready_image, processing_log, metadata)
    """
    original_bgr, metadata = load_image_from_source(source, pdf_page=pdf_page, dpi=dpi)
    processing_log: List[Dict[str, Any]] = []

    # Immutable copy: ensure we never touch original input array
    working_bgr = original_bgr.copy()
    processing_log.append({
        "step": "load",
        "details": f"Loaded image with shape {working_bgr.shape}",
        "shape": list(working_bgr.shape),
    })

    # Step 1: Grayscale
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    processing_log.append({
        "step": "grayscale",
        "details": "Converted BGR color to 8-bit single-channel grayscale",
        "shape": list(gray.shape),
    })

    # Step 2: Denoise
    # Bilateral filter removes background noise while preserving sharp font edges
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    processing_log.append({
        "step": "denoise",
        "details": "Applied edge-preserving bilateral filter (d=7, sigma=50)",
    })

    # Step 3: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    processing_log.append({
        "step": "clahe",
        "details": "Applied CLAHE (clipLimit=2.0, tileGrid=(8,8)) to balance uneven scan lighting",
    })

    # Step 4: Deskewing
    skew_angle = estimate_skew_angle(enhanced)
    if abs(skew_angle) >= 0.2:
        deskewed_gray = deskew_image(enhanced, skew_angle, border_color=(255, 255, 255))
        deskewed_bgr = deskew_image(working_bgr, skew_angle, border_color=(255, 255, 255))
        processing_log.append({
            "step": "deskew",
            "details": f"Corrected skew by {skew_angle:.2f} degrees",
            "angle_degrees": round(skew_angle, 2),
        })
    else:
        deskewed_gray = enhanced
        deskewed_bgr = working_bgr
        processing_log.append({
            "step": "deskew",
            "details": f"Skew angle {skew_angle:.2f} degrees within neutral tolerance; no rotation needed",
            "angle_degrees": round(skew_angle, 2),
        })

    # Step 5: Adaptive Gaussian Thresholding (optimal for OCR text isolation)
    binary = cv2.adaptiveThreshold(
        deskewed_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )
    processing_log.append({
        "step": "adaptive_threshold",
        "details": "Applied adaptive Gaussian thresholding (blockSize=15, C=8) for clean text binarization",
    })

    return deskewed_bgr, binary, processing_log, metadata


# -------------------------------------------------------------------------
# 4. Explainable Computer-Vision Anomaly Indicators
# -------------------------------------------------------------------------

def compute_error_level_analysis(
    image_bgr: np.ndarray,
    quality: int = 90,
    scale: int = 15,
    threshold: int = 40,
) -> Dict[str, Any]:
    """Error Level Analysis (ELA) for JPEG recompression residual detection.
    
    Explainability:
      When a JPEG image is modified (e.g. text spliced in or modified digitally)
      and re-saved, the pasted region has a different error level / compression
      rate than the rest of the image. ELA highlights compression delta.
    
    Notice:
      This is an explainable heuristic, NOT proof of tampering.
    """
    # Encode to JPEG in memory at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
    if not success:
        return {
            "indicator": "error_level_analysis",
            "score": 0.0,
            "applicable": False,
            "regions": [],
            "message": "Failed to re-encode image for ELA",
        }

    # Decompress back to array
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Calculate absolute difference and amplify
    diff = cv2.absdiff(image_bgr, recompressed)
    diff_amplified = cv2.scaleAdd(diff, scale, np.zeros_like(diff))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Statistical evaluation
    mean_val = float(np.mean(diff_gray))
    std_val = float(np.std(diff_gray))
    max_val = float(np.max(diff_gray))

    # Identify anomalous high-residual regions
    # Pixels where difference is significantly higher than normal recompression noise
    outlier_thresh = int(mean_val + 3.0 * std_val)
    outlier_thresh = max(outlier_thresh, threshold)
    _, mask = cv2.threshold(diff_gray, outlier_thresh, 255, cv2.THRESH_BINARY)

    # Find bounding contours of outlier regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict[str, int]] = []
    
    # Filter small noise specks (min area 100 px)
    h, w = image_bgr.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area > 120:
            rx, ry, rw, rh = cv2.boundingRect(c)
            # Avoid full-page border false positives
            if rw < 0.95 * w and rh < 0.95 * h:
                regions.append({"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)})

    # Normalized score between 0 and 100 based on standard deviation and outlier area
    total_pixels = h * w
    outlier_pixels = int(np.count_nonzero(mask))
    outlier_ratio = outlier_pixels / max(1, total_pixels)

    score = min(100.0, (outlier_ratio * 1000.0) + (std_val * 4.0))

    return {
        "indicator": "error_level_analysis",
        "score": round(float(score), 2),
        "mean_residual": round(mean_val, 2),
        "std_residual": round(std_val, 2),
        "max_residual": round(max_val, 2),
        "outlier_regions_count": len(regions),
        "regions": regions[:15],  # Top 15 bounding boxes
        "applicable": True,
        "details": (
            f"ELA detected {len(regions)} inconsistent compression cluster(s). "
            f"Mean residual: {mean_val:.2f}, Std: {std_val:.2f}."
        ),
    }


def compute_noise_variance_inconsistency(
    image_bgr: np.ndarray,
    block_size: int = 32,
    z_threshold: float = 2.5,
) -> Dict[str, Any]:
    """Compute local high-frequency noise variance across a block grid.
    
    Explainability:
      Documents produced by a single scanner/camera share a uniform noise distribution.
      If digital text is pasted or a region is airbrushed/smoothed, its local
      noise variance will be an extreme statistical outlier (z-score > threshold).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    h, w = gray.shape[:2]

    # Compute Laplacian to isolate high-frequency noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Tile image into blocks and compute variance of Laplacian in each block
    variances: List[float] = []
    block_coords: List[Tuple[int, int]] = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            patch = laplacian[y : y + block_size, x : x + block_size]
            var = float(np.var(patch))
            variances.append(var)
            block_coords.append((x, y))

    if not variances:
        return {
            "indicator": "noise_variance_inconsistency",
            "score": 0.0,
            "regions": [],
            "details": "Image too small for block grid analysis",
        }

    arr = np.array(variances)
    median_var = float(np.median(arr))
    # Robust standard deviation via Median Absolute Deviation (MAD)
    mad = float(np.median(np.abs(arr - median_var)))
    robust_std = 1.4826 * mad if mad > 1e-4 else float(np.std(arr))

    regions: List[Dict[str, int]] = []
    outlier_count = 0

    if robust_std > 1e-4:
        for idx, var in enumerate(variances):
            z = abs(var - median_var) / robust_std
            if z > z_threshold:
                bx, by = block_coords[idx]
                regions.append({"x": bx, "y": by, "w": block_size, "h": block_size})
                outlier_count += 1

    total_blocks = len(variances)
    outlier_pct = (outlier_count / max(1, total_blocks)) * 100.0
    score = min(100.0, outlier_pct * 8.0)

    return {
        "indicator": "noise_variance_inconsistency",
        "score": round(float(score), 2),
        "total_blocks_analyzed": total_blocks,
        "outlier_blocks_count": outlier_count,
        "median_noise_var": round(median_var, 2),
        "robust_std": round(robust_std, 2),
        "regions": regions[:20],
        "details": (
            f"Analyzed {total_blocks} blocks ({block_size}x{block_size}px). "
            f"Found {outlier_count} block(s) ({outlier_pct:.1f}%) with inconsistent noise variance."
        ),
    }


def compute_text_layout_anomalies(
    ocr_boxes: List[Dict[str, Any]],
    z_threshold: float = 2.5,
) -> Dict[str, Any]:
    """Detect anomalous font heights and baseline spacing in OCR bounding boxes.
    
    Explainability:
      Fraudulent alteration often inserts numbers with mismatched font sizes
      or perturbed vertical baselines.
    """
    if not ocr_boxes or len(ocr_boxes) < 8:
        return {
            "indicator": "text_layout_anomalies",
            "score": 0.0,
            "regions": [],
            "details": "Insufficient OCR word boxes to compute robust statistics (needs >= 8 words).",
        }

    # Extract word box heights
    heights = [float(b["height"]) for b in ocr_boxes if b.get("height", 0) > 4]
    if len(heights) < 8:
        return {
            "indicator": "text_layout_anomalies",
            "score": 0.0,
            "regions": [],
            "details": "Insufficient valid word heights.",
        }

    arr_h = np.array(heights)
    med_h = float(np.median(arr_h))
    mad_h = float(np.median(np.abs(arr_h - med_h)))
    robust_std = 1.4826 * mad_h if mad_h > 1e-4 else float(np.std(arr_h))

    regions: List[Dict[str, int]] = []
    if robust_std > 1e-4:
        for b in ocr_boxes:
            h_val = float(b.get("height", 0))
            z = abs(h_val - med_h) / robust_std
            if z > z_threshold and b.get("text", "").strip():
                regions.append({
                    "x": int(b.get("left", 0)),
                    "y": int(b.get("top", 0)),
                    "w": int(b.get("width", 0)),
                    "h": int(b.get("height", 0)),
                    "text": str(b.get("text", "")),
                })

    score = min(100.0, (len(regions) / len(ocr_boxes)) * 250.0)
    return {
        "indicator": "text_layout_anomalies",
        "score": round(float(score), 2),
        "median_height": round(med_h, 2),
        "anomalous_boxes_count": len(regions),
        "regions": regions[:15],
        "details": f"Identified {len(regions)} word(s) with anomalous bounding box height vs document baseline.",
    }


def compute_copy_move_candidates(
    image_bgr: np.ndarray,
    min_matches: int = 12,
    min_distance_px: float = 40.0,
) -> Dict[str, Any]:
    """Detect candidate duplicated / copy-pasted regions using ORB feature matching.
    
    Explainability:
      Copy-move tampering occurs when a region of an image is duplicated to hide
      or forge content (e.g. duplicating numbers or stamps).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < min_matches:
        return {
            "indicator": "copy_move_candidates",
            "score": 0.0,
            "regions": [],
            "match_count": 0,
            "details": "Insufficient keypoints for copy-move matching.",
        }

    # Match descriptors against themselves using BFMatcher with k=3
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    duplicate_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    regions: List[Dict[str, int]] = []

    for m in matches:
        if len(m) >= 2:
            best = m[1]  # m[0] is self-match (distance = 0)
            if best.distance < 35:  # Good Hamming distance threshold
                pt1 = keypoints[best.queryIdx].pt
                pt2 = keypoints[best.trainIdx].pt
                # Enforce physical spatial distance to avoid adjacent pixel matches
                dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist > min_distance_px:
                    duplicate_pairs.append((
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                    ))

    # If substantial clustered duplicate keypoints exist, add bounding regions
    if len(duplicate_pairs) >= min_matches:
        all_pts = [p for pair in duplicate_pairs for p in pair]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        rx, ry = min(xs), min(ys)
        rw, rh = max(xs) - rx, max(ys) - ry
        regions.append({"x": rx, "y": ry, "w": rw, "h": rh})

    score = min(100.0, len(duplicate_pairs) * 4.0)

    return {
        "indicator": "copy_move_candidates",
        "score": round(float(score), 2),
        "match_count": len(duplicate_pairs),
        "regions": regions,
        "details": f"Found {len(duplicate_pairs)} distant identical feature pair(s).",
    }


def run_cv_anomaly_pipeline(
    image_bgr: np.ndarray,
    ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    is_jpeg: bool = True,
) -> Dict[str, Any]:
    """Execute all explainable CV anomaly detectors and return combined assessment."""
    indicators: List[Dict[str, Any]] = []
    all_regions: List[Dict[str, Any]] = []

    # 1. Error Level Analysis (JPEG sources)
    if is_jpeg:
        ela_res = compute_error_level_analysis(image_bgr)
        indicators.append(ela_res)
        for r in ela_res.get("regions", []):
            all_regions.append({**r, "indicator": "error_level_analysis"})

    # 2. Local Noise Variance Inconsistency
    noise_res = compute_noise_variance_inconsistency(image_bgr)
    indicators.append(noise_res)
    for r in noise_res.get("regions", []):
        all_regions.append({**r, "indicator": "noise_variance_inconsistency"})

    # 3. Text Layout Outliers (if OCR bounding boxes provided)
    if ocr_boxes:
        layout_res = compute_text_layout_anomalies(ocr_boxes)
        indicators.append(layout_res)
        for r in layout_res.get("regions", []):
            all_regions.append({**r, "indicator": "text_layout_anomalies"})

    # 4. Copy-Move Candidates
    cm_res = compute_copy_move_candidates(image_bgr)
    indicators.append(cm_res)
    for r in cm_res.get("regions", []):
        all_regions.append({**r, "indicator": "copy_move_candidates"})

    # Aggregate CV score (weighted average of active indicators)
    scores = [ind["score"] for ind in indicators if ind.get("score") is not None]
    overall_cv_score = float(np.mean(scores)) if scores else 0.0

    return {
        "cvScore": round(overall_cv_score, 2),
        "cvRegions": all_regions,
        "indicators": indicators,
        "disclaimer": (
            "These indicators are explainable computer-vision heuristics and "
            "do not constitute proof of tampering or fraud. All flagged items require human review."
        ),
    }

