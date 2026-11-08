"""Multi-Modal Duplicate Detection Engine for FinVision.

Detects duplicates across three distinct complementary levels:
1. Exact Cryptographic Hash: SHA-256 byte-for-byte identity.
2. Near-Exact Perceptual Hash (pHash & dHash):
   - Computes robust 64-bit DCT perceptual hashes.
   - Measures Hamming distance between image hashes.
   - Threshold tuned: Hamming distance <= 5 (documented in config/rules_config.json)
     allows for compression artifacts, slight crops, or lighting changes while avoiding
     false positive collisions across different document layouts.
3. Semantic Business Identity:
   - Same normalized (invoiceNumber, vendor)
   - Same normalized (vendor, date, total)

Pure Python domain module: zero Firebase dependencies (can accept any document collection or query results).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imagehash
from PIL import Image

# Read configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "rules_config.json"
DEFAULT_HAMMING_THRESHOLD = 5

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            HAMMING_THRESHOLD = int(cfg.get("phash_hamming_threshold", DEFAULT_HAMMING_THRESHOLD))
    except Exception:
        HAMMING_THRESHOLD = DEFAULT_HAMMING_THRESHOLD
else:
    HAMMING_THRESHOLD = DEFAULT_HAMMING_THRESHOLD


def compute_sha256(data_bytes: bytes) -> str:
    """Compute standard SHA-256 hex digest of file bytes."""
    return hashlib.sha256(data_bytes).hexdigest()


def compute_perceptual_hashes(image: Union[Image.Image, bytes, Path, str]) -> Dict[str, str]:
    """Compute 64-bit pHash (DCT-based) and dHash (gradient-based) strings."""
    if isinstance(image, (bytes, Path, str)):
        from src.image_processing import load_image_from_source
        img_bgr, _ = load_image_from_source(image)
        pil_img = Image.fromarray(img_bgr)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type for perceptual hashing: {type(image)}")

    phash = str(imagehash.phash(pil_img))
    dhash = str(imagehash.dhash(pil_img))

    return {
        "pHash": phash,
        "dHash": dhash,
    }


def compute_phash_hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Calculate bit-level Hamming distance between two perceptual hash hex strings."""
    h1 = imagehash.hex_to_hash(hash1_hex)
    h2 = imagehash.hex_to_hash(hash2_hex)
    return int(h1 - h2)


def normalize_string(val: Optional[str]) -> str:
    """Strip punctuation, whitespace, and normalize case for semantic matching."""
    if not val:
        return ""
    cleaned = re.sub(r"[^\w\d]", "", str(val).lower().strip())
    return cleaned


def find_duplicates(
    target_doc: Dict[str, Any],
    candidate_docs: List[Dict[str, Any]],
    hamming_threshold: int = HAMMING_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate target document against candidate documents.
    
    Returns structured duplicate report matching Firestore verification schema:
    {
        "hasDuplicate": bool,
        "duplicateMatches": [
            {
                "docId": str,
                "matchType": "EXACT_SHA256" | "PERCEPTUAL_PHASH" | "SEMANTIC_INVOICE_VENDOR" | "SEMANTIC_VENDOR_DATE_TOTAL",
                "hammingDistance": int | None,
                "reason": str,
            }
        ]
    }
    """
    matches: List[Dict[str, Any]] = []

    target_sha = target_doc.get("sha256") or target_doc.get("file", {}).get("sha256")
    target_phash = target_doc.get("pHash") or target_doc.get("file", {}).get("pHash")
    
    target_fields = target_doc.get("fields", {})
    t_inv_num = normalize_string(target_fields.get("invoiceNumber"))
    t_vendor = normalize_string(target_fields.get("vendor"))
    t_date = str(target_fields.get("date", "")).strip()
    try:
        t_total = float(target_fields.get("total", 0.0) or 0.0)
    except Exception:
        t_total = 0.0

    target_doc_id = target_doc.get("docId", target_doc.get("id", ""))

    for cand in candidate_docs:
        cand_id = cand.get("docId", cand.get("id", "unknown"))
        # Do not compare document with itself
        if target_doc_id and cand_id == target_doc_id:
            continue

        cand_sha = cand.get("sha256") or cand.get("file", {}).get("sha256")
        cand_phash = cand.get("pHash") or cand.get("file", {}).get("pHash")
        cand_fields = cand.get("fields", {})

        # 1. Exact Cryptographic Match
        if target_sha and cand_sha and target_sha == cand_sha:
            matches.append({
                "docId": cand_id,
                "matchType": "EXACT_SHA256",
                "hammingDistance": 0,
                "reason": f"Exact binary match: identical SHA-256 hash ({target_sha[:12]}...).",
            })
            continue

        # 2. Near-Exact Perceptual Match (pHash Hamming distance)
        if target_phash and cand_phash:
            try:
                dist = compute_phash_hamming_distance(target_phash, cand_phash)
                if dist <= hamming_threshold:
                    matches.append({
                        "docId": cand_id,
                        "matchType": "PERCEPTUAL_PHASH",
                        "hammingDistance": dist,
                        "reason": (
                            f"Near-exact visual match: perceptual pHash Hamming distance is {dist} "
                            f"(threshold: <= {hamming_threshold}). Indicates identical or slightly modified scan."
                        ),
                    })
                    continue
            except Exception:
                pass

        # 3. Semantic Business Match: (invoiceNumber, vendor)
        c_inv_num = normalize_string(cand_fields.get("invoiceNumber"))
        c_vendor = normalize_string(cand_fields.get("vendor"))
        if t_inv_num and c_inv_num and t_vendor and c_vendor:
            if t_inv_num == c_inv_num and t_vendor == c_vendor:
                matches.append({
                    "docId": cand_id,
                    "matchType": "SEMANTIC_INVOICE_VENDOR",
                    "hammingDistance": None,
                    "reason": f"Semantic duplicate: identical invoice number ('{t_inv_num}') and vendor ('{t_vendor}').",
                })
                continue

        # 4. Semantic Business Match: (vendor, date, total)
        c_date = str(cand_fields.get("date", "")).strip()
        try:
            c_total = float(cand_fields.get("total", 0.0) or 0.0)
        except Exception:
            c_total = 0.0

        if t_vendor and c_vendor and t_date and c_date and t_total > 0 and c_total > 0:
            if t_vendor == c_vendor and t_date == c_date and abs(t_total - c_total) < 0.05:
                matches.append({
                    "docId": cand_id,
                    "matchType": "SEMANTIC_VENDOR_DATE_TOTAL",
                    "hammingDistance": None,
                    "reason": f"Semantic duplicate: identical vendor ('{t_vendor}'), date ('{t_date}'), and total (${t_total:.2f}).",
                })
                continue

    return {
        "hasDuplicate": len(matches) > 0,
        "duplicateMatches": matches,
        "matchCount": len(matches),
    }

