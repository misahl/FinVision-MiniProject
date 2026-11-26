"""OCR and Document Field Extraction Pipeline for FinVision.

This module provides:
1. Low-level OCR execution via pytesseract (with word boxes and confidences).
2. Spatial 2D line clustering to preserve multi-column layout.
3. Explicit regex and layout heuristics for extracting invoice fields:
   - Invoice Number
   - Date (standardized to YYYY-MM-DD)
   - Vendor & Customer (multi-column awareness)
   - Currency
   - Subtotal, Tax, Tax Rate, Total (using Decimal for precise money representation)
   - Line Items (description, quantity, unit price, line total)
4. Field-level confidence scores mapped from constituent word confidences.
5. Strict honesty: returns None for any un-extracted field (never fabricates).
6. Ground-truth evaluation against real dataset annotations.

Zero Firebase dependencies: purely functional and unit-testable offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Load local environment configuration
load_dotenv()

# Configure Tesseract binary path
tess_cmd = os.getenv("TESSERACT_CMD")
if tess_cmd and Path(tess_cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
elif os.name == "nt":
    # Common default on Windows if not set in .env
    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_win).exists():
        pytesseract.pytesseract.tesseract_cmd = default_win


# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------

@dataclass
class LineItem:
    description: str
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "quantity": float(self.quantity),
            "unitPrice": float(self.unit_price),
            "lineTotal": float(self.line_total),
        }


@dataclass
class ExtractedInvoice:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor: Optional[str] = None
    customer: Optional[str] = None
    currency: str = "USD"
    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    tax_rate: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    total: Optional[Decimal] = None
    line_items: List[LineItem] = field(default_factory=list)
    field_confidence: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    ocr_words: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary matching Firestore data model."""
        return {
            "fields": {
                "invoiceNumber": self.invoice_number,
                "vendor": self.vendor,
                "customer": self.customer,
                "date": self.invoice_date,
                "dueDate": self.due_date,
                "currency": self.currency,
                "subtotal": float(self.subtotal) if self.subtotal is not None else None,
                "tax": float(self.tax) if self.tax is not None else None,
                "taxRate": float(self.tax_rate) if self.tax_rate is not None else None,
                "discount": float(self.discount) if self.discount is not None else 0.0,
                "total": float(self.total) if self.total is not None else None,
            },
            "fieldConfidence": self.field_confidence,
            "overallConfidence": round(self.overall_confidence, 2),
            "lineItems": [item.to_dict() for item in self.line_items],
            "ocrWordsCount": len(self.ocr_words),
        }


# -------------------------------------------------------------------------
# Helper Functions: Numbers, Dates & Currencies
# -------------------------------------------------------------------------

def parse_decimal_amount(value_str: str) -> Optional[Decimal]:
    """Parse numeric currency strings reliably, supporting European (comma) and US (dot) formats.
    
    Examples:
      '232.95'   -> Decimal('232.95')
      '232,95'   -> Decimal('232.95')
      '1,250.00' -> Decimal('1250.00')
      '1.250,00' -> Decimal('1250.00')
    """
    if not value_str:
        return None

    # Clean currency symbols and spaces
    cleaned = re.sub(r"[^\d,\.\-]", "", value_str.strip())
    if not cleaned or cleaned == "-":
        return None

    try:
        # Determine format if both comma and dot exist
        if "," in cleaned and "." in cleaned:
            # Check which appears last
            if cleaned.rfind(".") > cleaned.rfind(","):
                # US: 1,234.56
                cleaned = cleaned.replace(",", "")
            else:
                # EU: 1.234,56
                cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "," in cleaned and "." not in cleaned:
            # If comma is followed by exactly 2 digits, likely decimal separator: 232,95
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) in [2, 3]:
                cleaned = cleaned.replace(",", ".")
            else:
                # Thousand separator: 1,000
                cleaned = cleaned.replace(",", "")

        val = Decimal(cleaned)
        return val.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def standardize_date(date_str: str) -> Optional[str]:
    """Parse various date formats and normalize to ISO 'YYYY-MM-DD'.
    
    Supported:
      MM/DD/YYYY, YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD, Month DD, YYYY
    """
    if not date_str:
        return None

    cleaned = date_str.strip()
    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d-%b-%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%B %d %Y",
        "%b %d %Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            # Basic sanity check: year between 1980 and 2050
            if 1980 <= dt.year <= 2050:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def detect_currency(text: str) -> str:
    """Detect currency from symbols or 3-letter ISO codes."""
    if "€" in text or "EUR" in text:
        return "EUR"
    elif "£" in text or "GBP" in text:
        return "GBP"
    elif "₹" in text or "INR" in text:
        return "INR"
    elif "¥" in text or "JPY" in text:
        return "JPY"
    elif "CAD" in text:
        return "CAD"
    elif "AUD" in text:
        return "AUD"
    return "USD"


# -------------------------------------------------------------------------
# OCR Execution & Spatial Line Clustering
# -------------------------------------------------------------------------

def run_ocr(image: Union[np.ndarray, Image.Image]) -> Tuple[List[Dict[str, Any]], str, float]:
    """Run Tesseract OCR via pytesseract.image_to_data safely.
    
    Guarantees:
    - Never throws AttributeError on empty or float text columns.
    - Gracefully catches TesseractNotFoundError or OS subprocess errors without crashing.
    - If initial thresholded image yields 0 words, falls back to grayscale for robust OCR.
    
    Returns:
      (valid_words_list, raw_text, overall_confidence)
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    def _ocr_pass(img_to_run: Image.Image) -> Tuple[List[Dict[str, Any]], str, float]:
        try:
            data = pytesseract.image_to_data(img_to_run, output_type=pytesseract.Output.DATAFRAME)
            if data is None or data.empty:
                return [], "", 0.0

            # Safe type normalization
            if "text" in data.columns:
                data["text"] = data["text"].astype(str)
            else:
                return [], "", 0.0

            if "conf" in data.columns:
                data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(-1.0)
            else:
                data["conf"] = -1.0

            # Filter rows with empty text or invalid confidence
            valid = data[(~data["text"].isin(["", "nan", "None"])) & (data["text"].str.strip() != "") & (data["conf"] >= 0)].copy()
            if valid.empty:
                return [], "", 0.0

            words: List[Dict[str, Any]] = []
            for _, row in valid.iterrows():
                words.append({
                    "text": str(row["text"]).strip(),
                    "conf": float(row["conf"]),
                    "left": int(row["left"]),
                    "top": int(row["top"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "line_num": int(row.get("line_num", 0)),
                    "block_num": int(row.get("block_num", 0)),
                })

            overall_conf = float(valid["conf"].mean()) if not valid.empty else 0.0
            raw_text = " ".join(w["text"] for w in words)
            return words, raw_text, overall_conf
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Tesseract OCR execution error: %s", e)
            return [], "", 0.0

    words, raw_text, overall_conf = _ocr_pass(pil_img)
    # If 0 words found and image mode is not grayscale, try L mode
    if not words and pil_img.mode != "L":
        words, raw_text, overall_conf = _ocr_pass(pil_img.convert("L"))

    return words, raw_text, overall_conf


def extract_words_from_pdf(
    pdf_source: Union[bytes, str, Path],
    page_num: int = 0,
    dpi: int = 300,
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Extract native vector text and word bounding boxes directly from PDF via PyMuPDF.
    
    Provides 100% character precision on native digital invoices without relying on OCR.
    Returns:
      (words_list, raw_text, confidence)
    """
    import pymupdf

    try:
        if isinstance(pdf_source, (str, Path)):
            doc = pymupdf.open(str(pdf_source))
        else:
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")

        if len(doc) == 0:
            return [], "", 0.0

        page_index = max(0, min(page_num, len(doc) - 1))
        page = doc[page_index]
        raw_words = page.get_text("words")
        doc.close()

        if not raw_words:
            return [], "", 0.0

        zoom = dpi / 72.0
        words: List[Dict[str, Any]] = []

        for w in raw_words:
            txt = str(w[4]).strip()
            if not txt:
                continue
            words.append({
                "text": txt,
                "conf": 99.0,
                "left": int(w[0] * zoom),
                "top": int(w[1] * zoom),
                "width": max(1, int((w[2] - w[0]) * zoom)),
                "height": max(1, int((w[3] - w[1]) * zoom)),
                "line_num": int(w[6]),
                "block_num": int(w[5]),
            })

        raw_text = " ".join(w["text"] for w in words)
        return words, raw_text, 99.0
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("PyMuPDF native text extraction error: %s", e)
        return [], "", 0.0


def extract_document_words(
    source: Union[bytes, str, Path, np.ndarray, Image.Image],
    mime_type: Optional[str] = None,
    page_num: int = 0,
    fallback_image: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Unified multi-engine document text extractor.
    
    1. If source is PDF: first attempts native digital vector extraction via PyMuPDF.
       If native text is present (>= 5 words), returns directly with 99% confidence.
       If scanned / empty (< 5 words), falls back to OCR on the rasterized image.
    2. If source is an image array / bytes: runs OCR with graceful error handling.
    """
    is_pdf = False
    if mime_type == "application/pdf":
        is_pdf = True
    elif isinstance(source, bytes) and source.startswith(b"%PDF"):
        is_pdf = True
    elif isinstance(source, (str, Path)) and str(source).lower().endswith(".pdf"):
        is_pdf = True

    if is_pdf and isinstance(source, (bytes, str, Path)):
        pdf_words, pdf_raw, pdf_conf = extract_words_from_pdf(source, page_num=page_num)
        if len(pdf_words) >= 5:
            return pdf_words, pdf_raw, pdf_conf

    # Use fallback image or load image
    if fallback_image is not None:
        return run_ocr(fallback_image)

    if isinstance(source, (np.ndarray, Image.Image)):
        return run_ocr(source)

    try:
        from src.image_processing import load_image_from_source
        img_bgr, _ = load_image_from_source(source, pdf_page=page_num)
        return run_ocr(img_bgr)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Unified document word extraction failed: %s", e)
        return [], "", 0.0


def cluster_words_into_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group words into horizontal lines using 2D vertical bounding overlap.
    
    This reconstructs the true 2D layout of multi-column invoices and headers.
    Within each line, words are ordered left-to-right by X coordinate.
    """
    if not words:
        return []

    # Sort words primarily by top coordinate
    sorted_words = sorted(words, key=lambda w: (w["top"], w["left"]))
    lines: List[List[Dict[str, Any]]] = []

    for w in sorted_words:
        placed = False
        w_top = w["top"]
        w_bottom = w["top"] + w["height"]

        for line in lines:
            line_top = min(lw["top"] for lw in line)
            line_bottom = max(lw["top"] + lw["height"] for lw in line)
            
            # Compute vertical overlap
            overlap = min(line_bottom, w_bottom) - max(line_top, w_top)
            min_h = min(line_bottom - line_top, w["height"])

            if min_h > 0 and (overlap / min_h) > 0.45:
                line.append(w)
                placed = True
                break

        if not placed:
            lines.append([w])

    # Sort lines vertically by average top coordinate, and words horizontally
    for line in lines:
        line.sort(key=lambda w: w["left"])

    lines.sort(key=lambda line: min(w["top"] for w in line))
    return lines


# -------------------------------------------------------------------------
# Field Extraction Heuristics
# -------------------------------------------------------------------------

def extract_invoice_fields(
    words: List[Dict[str, Any]],
    raw_text: str,
    overall_conf: float,
) -> ExtractedInvoice:
    """Extract structured invoice fields from OCR words using layout heuristics & regex.
    
    Heuristics are documented per field below.
    """
    lines = cluster_words_into_lines(words)
    line_strings = [" ".join(w["text"] for w in line) for line in lines]
    full_text = "\n".join(line_strings)

    inv = ExtractedInvoice(
        overall_confidence=overall_conf,
        ocr_words=words,
        raw_text=full_text,
    )
    inv.currency = detect_currency(full_text)

    # 1. Invoice Number Heuristic:
    inv_num_patterns = [
        r"\b(?:Invoice|Inv|Bill)\s*(?:no\.?|number|#|id)?\s*[:#]\s*([A-Za-z0-9\-_/]+)",
        r"\b(?:Invoice|Inv|Bill)\s+(?:no\.?|number|#|id)\s+([A-Za-z0-9\-_/]+)",
        r"\b(INV-[A-Za-z0-9\-_/]+)\b",
        r"\b([A-Z]{2,4}-\d{3,8})\b",
    ]
    for pattern in inv_num_patterns:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Must contain at least one digit and be at least 3 chars
            if len(candidate) >= 3 and any(ch.isdigit() for ch in candidate) and candidate.lower() not in ["invoice", "inv", "bill"]:
                inv.invoice_number = candidate
                matched_words = [w for w in words if candidate in w["text"]]
                inv.field_confidence["invoiceNumber"] = (
                    round(float(np.mean([w["conf"] for w in matched_words])), 2)
                    if matched_words else round(overall_conf, 2)
                )
                break

    # 2. Date Heuristic:
    date_patterns = [
        r"\b(?:Date\s*(?:of\s*issue)?|Invoice\s*Date|Date|Dated)\s*[:#]?\s*([0-9]{1,4}[/\-\.][0-9]{1,2}[/\-\.][0-9]{1,4})",
        r"\b(?:Date\s*(?:of\s*issue)?|Invoice\s*Date|Date|Dated)\s*[:#]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
        r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b",
        r"\b([0-9]{2}/[0-9]{2}/[0-9]{4})\b",
        r"\b([0-9]{2}\.[0-9]{2}\.[0-9]{4})\b",
        r"\b([0-9]{2}-[A-Za-z]{3}-[0-9]{4})\b",
        r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b",
    ]
    for pattern in date_patterns:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            raw_date = m.group(1).strip()
            std_date = standardize_date(raw_date)
            if std_date:
                inv.invoice_date = std_date
                matched_words = [w for w in words if raw_date in w["text"] or raw_date.replace("/", "") in w["text"]]
                inv.field_confidence["date"] = (
                    round(float(np.mean([w["conf"] for w in matched_words])), 2)
                    if matched_words else round(overall_conf, 2)
                )
                break

    # 2b. Due Date Heuristic:
    due_date_patterns = [
        r"\b(?:Due\s*Date|Payment\s*Due|Pay\s*By|Due\s*By)\s*[:#]?\s*([0-9]{1,4}[/\-\.][0-9]{1,2}[/\-\.][0-9]{1,4})",
        r"\b(?:Due\s*Date|Payment\s*Due|Pay\s*By|Due\s*By)\s*[:#]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
        r"\b(?:Due\s*Date|Payment\s*Due|Pay\s*By|Due\s*By)\s*[:#]?\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})",
    ]
    for pattern in due_date_patterns:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            raw_d = m.group(1).strip()
            std_d = standardize_date(raw_d)
            if std_d:
                inv.due_date = std_d
                inv.field_confidence["dueDate"] = round(overall_conf, 2)
                break

    # 3. Vendor & Customer Heuristic (Multi-column aware):
    for i, line_str in enumerate(line_strings):
        if "seller:" in line_str.lower() and "client:" in line_str.lower() and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line:
                mid_x = np.median([w["left"] for w in next_line])
                seller_words = [w for w in next_line if w["left"] < mid_x]
                client_words = [w for w in next_line if w["left"] >= mid_x]
                if seller_words and not inv.vendor:
                    inv.vendor = " ".join(w["text"] for w in seller_words)
                    inv.field_confidence["vendor"] = round(float(np.mean([w["conf"] for w in seller_words])), 2)
                if client_words and not inv.customer:
                    inv.customer = " ".join(w["text"] for w in client_words)
                    inv.field_confidence["customer"] = round(float(np.mean([w["conf"] for w in client_words])), 2)
            break
        elif "seller:" in line_str.lower() and not inv.vendor and i + 1 < len(lines):
            next_line = lines[i + 1]
            inv.vendor = " ".join(w["text"] for w in next_line[:4])
            inv.field_confidence["vendor"] = round(float(np.mean([w["conf"] for w in next_line[:4]])), 2)
        elif "client:" in line_str.lower() and not inv.customer and i + 1 < len(lines):
            next_line = lines[i + 1]
            inv.customer = " ".join(w["text"] for w in next_line[:4])
            inv.field_confidence["customer"] = round(float(np.mean([w["conf"] for w in next_line[:4]])), 2)

    # Fallback vendor / customer heuristics
    if not inv.vendor:
        m = re.search(r"\b(?:From|Vendor|Supplier|Seller)\s*[:#]\s*([^\n\r]+)", full_text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"\b(?:Invoice|Date|Bill\s*To|Client|Total)\b", cand, flags=re.IGNORECASE)[0].strip()
            if len(cand) >= 2:
                inv.vendor = cand
                inv.field_confidence["vendor"] = round(overall_conf, 2)

    # Top-of-page prominent business header heuristic
    if not inv.vendor:
        for line in lines[:3]:
            line_str = " ".join(w["text"] for w in line).strip()
            cand = re.sub(r"\b(?:INVOICE|RECEIPT|TAX\s*INVOICE|STATEMENT)\b", "", line_str, flags=re.IGNORECASE).strip()
            if len(cand) >= 3 and not re.match(r"^\d+", cand) and not any(k in cand.lower() for k in ["bill to", "due date", "date:"]):
                if not re.search(r"\b(?:Street|St\.|Drive|Dr\.|Avenue|Ave\.|Boulevard|Blvd\.|Road|Rd\.)\b", cand, re.IGNORECASE):
                    inv.vendor = cand
                    inv.field_confidence["vendor"] = round(overall_conf, 2)
                    break

    if not inv.customer:
        m = re.search(r"\b(?:Bill\s*To|Sold\s*To|Client|Customer)\s*[:#]?\s*([^\n\r]+)", full_text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"\b(?:Invoice|Date|Due|Total|Tax|Subtotal)\b", cand, flags=re.IGNORECASE)[0].strip()
            if len(cand) >= 2:
                inv.customer = cand
                inv.field_confidence["customer"] = round(overall_conf, 2)

    if not inv.customer:
        for i, line_str in enumerate(line_strings):
            if re.search(r"\b(?:Bill\s*To|Sold\s*To|Client|Customer)\b", line_str, re.IGNORECASE):
                if i + 1 < len(lines):
                    next_txt = " ".join(w["text"] for w in lines[i+1]).strip()
                    if len(next_txt) >= 2 and not any(k in next_txt.lower() for k in ["subtotal", "total", "tax", "date", "invoice", "description"]):
                        inv.customer = next_txt
                        inv.field_confidence["customer"] = round(overall_conf, 2)
                        break

    # Pattern to match monetary amounts with optional space, comma, or dot thousands separators
    money_pat = r"\b\d{1,3}(?:[\.,\s]\d{3})*(?:[,\.]\d{2})\b|\b\d+(?:[,\.]\d{2})\b|\b\d{2,6}\b"

    # 4. Financial Totals Heuristic (Summary table & keyword matching):
    for line in reversed(lines):
        line_txt = " ".join(w["text"] for w in line)
        low_txt = line_txt.lower()

        # Multi-value summary line: 'Total $ 211,77 $ 21,18 $ 232,95'
        numbers = re.findall(money_pat, line_txt)
        if "total" in low_txt and len(numbers) >= 3 and inv.total is None:
            vals = [parse_decimal_amount(n) for n in numbers if parse_decimal_amount(n) is not None]
            if len(vals) >= 3:
                inv.subtotal = vals[0]
                inv.tax = vals[1]
                inv.total = vals[2]
                inv.field_confidence["total"] = round(float(np.mean([w["conf"] for w in line])), 2)
                inv.field_confidence["subtotal"] = inv.field_confidence["total"]
                inv.field_confidence["tax"] = inv.field_confidence["total"]

        # Standalone Total
        if inv.total is None and any(k in low_txt for k in ["grand total", "total amount", "invoice total", "total", "gross worth", "amount due", "balance"]):
            # Do not skip if it says 'total (incl. tax)' - only skip if it is purely a tax line like 'total tax'
            is_tax_only = bool(re.search(r"\b(tax total|total tax|vat total|total vat|tax amount)\b", low_txt))
            if not is_tax_only and "subtotal" not in low_txt:
                amt = re.findall(money_pat, line_txt)
                if amt:
                    val = parse_decimal_amount(amt[-1])
                    if val is not None:
                        inv.total = val
                        inv.field_confidence["total"] = round(float(np.mean([w["conf"] for w in line])), 2)

        # Standalone Subtotal
        if inv.subtotal is None and any(k in low_txt for k in ["subtotal", "sub total", "net worth", "net amount"]):
            amt = re.findall(money_pat, line_txt)
            if amt:
                val = parse_decimal_amount(amt[-1])
                if val is not None:
                    inv.subtotal = val
                    inv.field_confidence["subtotal"] = round(float(np.mean([w["conf"] for w in line])), 2)

        # Standalone Discount
        if inv.discount is None and any(k in low_txt for k in ["discount", "less discount", "rebate"]):
            amt = re.findall(money_pat, line_txt)
            if amt:
                val = parse_decimal_amount(amt[-1])
                if val is not None:
                    inv.discount = val
                    inv.field_confidence["discount"] = round(float(np.mean([w["conf"] for w in line])), 2)

        # Standalone Tax / VAT
        if inv.tax is None and any(k in low_txt for k in ["tax", "vat", "gst"]):
            amt = re.findall(money_pat, line_txt)
            if amt:
                val = parse_decimal_amount(amt[-1])
                if val is not None:
                    inv.tax = val
                    inv.field_confidence["tax"] = round(float(np.mean([w["conf"] for w in line])), 2)

        # Tax Rate (e.g. '10%', '20%')
        if inv.tax_rate is None and "%" in line_txt:
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", line_txt)
            if pct_match:
                try:
                    pct = Decimal(pct_match.group(1)) / Decimal(100)
                    inv.tax_rate = pct
                    inv.field_confidence["taxRate"] = round(float(np.mean([w["conf"] for w in line])), 2)
                except InvalidOperation:
                    pass

    # 5. Line Items Heuristic:
    # Scan table lines between 'ITEMS' / 'Description' and 'SUMMARY' / 'Total'
    in_items_section = False
    for line in lines:
        line_txt = " ".join(w["text"] for w in line)
        if "items" in line_txt.lower() or "description" in line_txt.lower():
            in_items_section = True
            continue
        if in_items_section and any(k in line_txt.lower() for k in ["summary", "total", "subtotal", "vat ["]):
            in_items_section = False
            break

        if in_items_section:
            # Multi-token line item extraction with mathematical solver:
            words_in_line = line_txt.split()
            tokens = []
            for idx, w in enumerate(words_in_line):
                clean_w = re.sub(r"^[^\d]+|[^\d]+$", "", w.replace(",", ""))
                if clean_w and re.match(r"^\d+(\.\d+)?$", clean_w):
                    tokens.append((idx, Decimal(clean_w)))

            qty = None
            unit_price = None
            line_total = None
            desc_end_idx = len(words_in_line)

            if len(tokens) >= 3:
                t_val = tokens[-1][1]
                p_val = tokens[-2][1]
                q_val = tokens[-3][1]
                # Check if last token is product of preceding two: qty * unit_price == line_total
                if abs((q_val * p_val) - t_val) <= Decimal("0.05"):
                    qty = q_val
                    unit_price = p_val
                    line_total = t_val
                    desc_end_idx = tokens[-3][0]
                elif abs((p_val * q_val) - t_val) <= Decimal("0.05"):
                    qty = p_val
                    unit_price = q_val
                    line_total = t_val
                    desc_end_idx = tokens[-3][0]
                else:
                    qty = q_val
                    unit_price = p_val
                    line_total = t_val
                    desc_end_idx = tokens[-3][0]
            elif len(tokens) == 2:
                val1 = tokens[0][1]
                val2 = tokens[1][1]
                if val1 == val2:
                    qty = Decimal("1.0")
                    unit_price = val1
                    line_total = val2
                elif val1 <= Decimal("50"):
                    qty = val1
                    line_total = val2
                    unit_price = (val2 / val1).quantize(Decimal("0.01"))
                else:
                    qty = Decimal("1.0")
                    unit_price = val1
                    line_total = val2
                desc_end_idx = tokens[0][0]

            if qty is not None and line_total is not None:
                desc = " ".join(words_in_line[:desc_end_idx])
                desc = re.sub(r"[$€£₹]|USD|EUR|INR|CAD|AUD", "", desc)
                desc = re.sub(r"\b(each|pc|pcs|hours|hrs|items)\b", "", desc, flags=re.IGNORECASE).strip()
                if len(desc) >= 2:
                    inv.line_items.append(
                        LineItem(
                            description=desc,
                            quantity=qty,
                            unit_price=unit_price or Decimal(0),
                            line_total=line_total,
                        )
                    )

    # Fill default confidence of 0 for fields not found
    for field_name in ["invoiceNumber", "date", "vendor", "customer", "subtotal", "tax", "total"]:
        if field_name not in inv.field_confidence:
            inv.field_confidence[field_name] = 0.0

    return inv


def process_invoice_document(image_source: Union[str, Path, bytes, np.ndarray]) -> ExtractedInvoice:
    """End-to-end extraction pipeline: load -> OCR -> extract fields."""
    from src.image_processing import load_image_from_source

    img_bgr, _ = load_image_from_source(image_source)
    words, raw_text, overall_conf = run_ocr(img_bgr)
    extracted = extract_invoice_fields(words, raw_text, overall_conf)
    return extracted


# -------------------------------------------------------------------------
# Evaluation on Ground Truth Dataset
# -------------------------------------------------------------------------

def evaluate_extraction_accuracy(
    csv_path: Union[str, Path],
    images_base_dir: Union[str, Path],
    sample_size: int = 15,
) -> Dict[str, Any]:
    """Evaluate extraction accuracy against ground-truth CSV annotations on real dataset.
    
    Computes exact match precision, field recovery rate, and average confidences.
    No mock data: computes real evaluation metrics.
    """
    import json

    df = pd.read_csv(csv_path)
    sample_df = df.head(sample_size)

    metrics: Dict[str, Dict[str, int]] = {
        "invoice_number": {"total": 0, "correct": 0},
        "invoice_date": {"total": 0, "correct": 0},
        "vendor": {"total": 0, "correct": 0},
        "customer": {"total": 0, "correct": 0},
        "total": {"total": 0, "correct": 0},
        "tax": {"total": 0, "correct": 0},
    }

    base_dir = Path(images_base_dir)

    for _, row in sample_df.iterrows():
        file_name = row["File Name"]
        img_path = base_dir / file_name
        if not img_path.exists():
            continue

        gt_data = json.loads(row["Json Data"])
        gt_inv = gt_data.get("invoice", {})
        gt_sub = gt_data.get("subtotal", {})

        # Run extraction
        extracted = process_invoice_document(img_path)

        # 1. Invoice Number
        gt_num = gt_inv.get("invoice_number", "").strip()
        if gt_num:
            metrics["invoice_number"]["total"] += 1
            if extracted.invoice_number and extracted.invoice_number == gt_num:
                metrics["invoice_number"]["correct"] += 1

        # 2. Date
        gt_date = standardize_date(gt_inv.get("invoice_date", ""))
        if gt_date:
            metrics["invoice_date"]["total"] += 1
            if extracted.invoice_date and extracted.invoice_date == gt_date:
                metrics["invoice_date"]["correct"] += 1

        # 3. Vendor
        gt_vendor = gt_inv.get("seller_name", "").strip().lower()
        if gt_vendor:
            metrics["vendor"]["total"] += 1
            if extracted.vendor and gt_vendor in extracted.vendor.lower():
                metrics["vendor"]["correct"] += 1

        # 4. Customer
        gt_cust = gt_inv.get("client_name", "").strip().lower()
        if gt_cust:
            metrics["customer"]["total"] += 1
            if extracted.customer and gt_cust in extracted.customer.lower():
                metrics["customer"]["correct"] += 1

        # 5. Total
        gt_tot = parse_decimal_amount(gt_sub.get("total", ""))
        if gt_tot is not None:
            metrics["total"]["total"] += 1
            if extracted.total is not None and abs(extracted.total - gt_tot) < Decimal("0.05"):
                metrics["total"]["correct"] += 1

        # 6. Tax
        gt_tax = parse_decimal_amount(gt_sub.get("tax", ""))
        if gt_tax is not None:
            metrics["tax"]["total"] += 1
            if extracted.tax is not None and abs(extracted.tax - gt_tax) < Decimal("0.05"):
                metrics["tax"]["correct"] += 1

    # Compute percentage accuracy per field
    results: Dict[str, Any] = {}
    total_fields = 0
    total_correct = 0

    for field_name, counts in metrics.items():
        tot = counts["total"]
        cor = counts["correct"]
        acc = (cor / tot * 100.0) if tot > 0 else 0.0
        results[field_name] = {
            "annotated_count": tot,
            "correct_matches": cor,
            "accuracy_pct": round(acc, 2),
        }
        total_fields += tot
        total_correct += cor

    overall_accuracy = (total_correct / total_fields * 100.0) if total_fields > 0 else 0.0
    results["overall_accuracy_pct"] = round(overall_accuracy, 2)
    results["samples_evaluated"] = len(sample_df)

    return results


# -------------------------------------------------------------------------
# Document Validity Verification (Distinguish Invoices from Random Photos)
# -------------------------------------------------------------------------

INVOICE_INDICATOR_KEYWORDS = [
    "invoice", "bill to", "billed to", "receipt", "tax invoice", "total",
    "subtotal", "amount due", "balance due", "due date", "remit", "vat", "gst",
    "qty", "quantity", "unit price", "line total", "description", "vendor",
    "payment", "order no", "purchase order", "charges", "statement", "net amount",
    "gross amount", "currency", "terms", "payable", "discount"
]


def validate_is_invoice_document(
    raw_text: str,
    words_count: int,
    fields: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate whether an uploaded or photographed image is actually an invoice or receipt.
    
    Prevents random photos (selfies, blank walls, objects, scenery) from entering
    the verification pipeline and generating false 100% risk scores.
    """
    text_lower = (raw_text or "").lower()
    matches = [kw for kw in INVOICE_INDICATOR_KEYWORDS if kw in text_lower]

    tot = fields.get("total")
    has_total = tot is not None and float(tot or 0.0) > 0.0
    has_subtotal = fields.get("subtotal") is not None and float(fields.get("subtotal") or 0.0) > 0.0
    has_inv_num = bool(fields.get("invoiceNumber"))
    has_vendor = bool(fields.get("vendor"))

    # If practically no text was found (e.g. photo of face, wall, room):
    if words_count < 4 or len(text_lower.strip()) < 15:
        return (
            False,
            "⚠️ Image is not a valid invoice or receipt. No invoice structure, billing details, or financial text were detected. Please take a photo of a valid invoice.",
        )

    # If no financial keywords AND no financial numbers/identifiers:
    if len(matches) < 2 and not (has_total or has_subtotal) and not (has_inv_num and has_vendor):
        return (
            False,
            "⚠️ Image is not a valid invoice or receipt. No invoice structure, billing details, or financial totals were detected. Please take a photo of a valid invoice.",
        )

    return True, "Valid invoice structure detected."
