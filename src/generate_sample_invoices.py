"""Generate realistic synthetic sample invoices in PDF and PNG formats for testing FinVision."""

from pathlib import Path
import io
import zipfile
import pymupdf


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_invoices"


def create_invoice_doc(
    title: str,
    address: str,
    inv_num: str,
    date_str: str,
    due_date_str: str,
    bill_to: str,
    bill_address: str,
    items: list,
    subtotal_str: str,
    tax_str: str,
    discount_str: str = "",
    total_str: str = "",
    currency_symbol: str = "$",
    footer_note: str = "Thank you for your business. Please remit payment by the due date.",
) -> pymupdf.Document:
    """Create a clean, professional vector PDF invoice."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # 8.5 x 11 in (US Letter)

    # 1. Vendor Header
    if title:
        page.insert_text((50, 60), title, fontsize=16, fontname="helv", color=(0.08, 0.18, 0.36))
    if address:
        page.insert_text((50, 78), address, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    # 2. Document Identifier
    page.insert_text((420, 60), "INVOICE", fontsize=20, fontname="helv", color=(0.08, 0.18, 0.36))
    if inv_num:
        page.insert_text((420, 85), f"Invoice #: {inv_num}", fontsize=10, fontname="helv")
    if date_str:
        page.insert_text((420, 100), f"Date: {date_str}", fontsize=10, fontname="helv")
    if due_date_str:
        page.insert_text((420, 115), f"Due Date: {due_date_str}", fontsize=10, fontname="helv")

    # Divider
    page.draw_line((50, 135), (562, 135), color=(0.85, 0.88, 0.92), width=1)

    # 3. Bill To
    page.insert_text((50, 155), "BILL TO:", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    if bill_to:
        page.insert_text((50, 170), bill_to, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    if bill_address:
        page.insert_text((50, 184), bill_address, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

    # 4. Line Items Table Header
    table_top = 215
    page.draw_rect(pymupdf.Rect(50, table_top, 562, table_top + 22), color=None, fill=(0.94, 0.96, 0.99))
    page.insert_text((58, table_top + 15), "Description", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text((340, table_top + 15), "Qty", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text((410, table_top + 15), "Unit Price", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
    page.insert_text((490, table_top + 15), "Line Total", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))

    # Items
    curr_y = table_top + 40
    for desc, qty, price, ltotal in items:
        page.insert_text((58, curr_y), desc, fontsize=9, fontname="helv")
        page.insert_text((345, curr_y), str(qty), fontsize=9, fontname="helv")
        page.insert_text((410, curr_y), f"{currency_symbol} {price}", fontsize=9, fontname="helv")
        page.insert_text((490, curr_y), f"{currency_symbol} {ltotal}", fontsize=9, fontname="helv")
        curr_y += 24

    page.draw_line((50, curr_y + 10), (562, curr_y + 10), color=(0.85, 0.88, 0.92), width=1)

    # 5. Financial Summary Totals
    summary_y = curr_y + 35
    if subtotal_str:
        page.insert_text((380, summary_y), "Subtotal:", fontsize=10, fontname="helv")
        page.insert_text((480, summary_y), f"{currency_symbol} {subtotal_str}", fontsize=10, fontname="helv")
        summary_y += 20

    if discount_str:
        page.insert_text((380, summary_y), "Discount:", fontsize=10, fontname="helv")
        page.insert_text((480, summary_y), f"{currency_symbol} {discount_str}", fontsize=10, fontname="helv")
        summary_y += 20

    if tax_str:
        page.insert_text((380, summary_y), "Tax / VAT:", fontsize=10, fontname="helv")
        page.insert_text((480, summary_y), f"{currency_symbol} {tax_str}", fontsize=10, fontname="helv")
        summary_y += 20

    if total_str:
        page.draw_rect(pymupdf.Rect(370, summary_y - 5, 562, summary_y + 22), color=None, fill=(0.94, 0.96, 0.99))
        page.insert_text((380, summary_y + 13), "Total Amount:", fontsize=11, fontname="helv", color=(0.08, 0.18, 0.36))
        page.insert_text((480, summary_y + 13), f"{currency_symbol} {total_str}", fontsize=11, fontname="helv", color=(0.08, 0.18, 0.36))

    # Footer
    page.insert_text((50, 740), footer_note, fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    page.insert_text((50, 755), "FinVision Synthetic Verification Test Asset — Validated Deterministic Specimen", fontsize=8, fontname="helv", color=(0.6, 0.6, 0.6))

    return doc


def generate_all_samples():
    """Build all 6 test invoice PDFs and PNGs."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    samples_meta = [
        {
            "filename_stem": "sample_01_valid_invoice",
            "title": "ACME GLOBAL SOLUTIONS INC.",
            "address": "123 Innovation Drive, Suite 400, Tech City, CA 94016",
            "inv_num": "INV-2026-001",
            "date_str": "2026-10-15",
            "due_date_str": "2026-11-15",
            "bill_to": "FinVision Enterprise LLC",
            "bill_address": "789 Financial Square, Floor 12, New York, NY 10005",
            "items": [
                ("Enterprise Platform License - Tier 1", "1", "600.00", "600.00"),
                ("Technical Advisory & Architecture Support", "4", "100.00", "400.00"),
            ],
            "subtotal": "1,000.00",
            "tax": "100.00",
            "discount": "",
            "total": "1,100.00",
            "currency": "$",
            "expected": "PASS (Low Risk, ~5/100, Verified)",
        },
        {
            "filename_stem": "sample_02_math_mismatch",
            "title": "APEX LOGISTICS CORP",
            "address": "456 Freight Way, Dallas, TX 75201",
            "inv_num": "INV-2026-002",
            "date_str": "2026-10-20",
            "due_date_str": "2026-11-20",
            "bill_to": "FinVision Enterprise LLC",
            "bill_address": "789 Financial Square, Floor 12, New York, NY 10005",
            "items": [
                ("Interstate Freight Transport Services", "2", "200.00", "400.00"),
                ("Warehouse Handling & Storage", "1", "100.00", "100.00"),
            ],
            "subtotal": "500.00",
            "tax": "50.00",
            "discount": "",
            "total": "850.00",  # Intentional error: 500 + 50 = 550, not 850
            "currency": "$",
            "expected": "FLAGGED (High Risk, ~75/100, Math Discrepancy)",
        },
        {
            "filename_stem": "sample_03_duplicate_invoice",
            "title": "ACME GLOBAL SOLUTIONS INC.",
            "address": "123 Innovation Drive, Suite 400, Tech City, CA 94016",
            "inv_num": "INV-2026-001",  # Same invoice number as sample 01
            "date_str": "2026-10-15",
            "due_date_str": "2026-11-15",
            "bill_to": "FinVision Enterprise LLC",
            "bill_address": "789 Financial Square, Floor 12, New York, NY 10005",
            "items": [
                ("Enterprise Platform License - Tier 1", "1", "600.00", "600.00"),
                ("Technical Advisory & Architecture Support", "4", "100.00", "400.00"),
            ],
            "subtotal": "1,000.00",
            "tax": "100.00",
            "discount": "",
            "total": "1,100.00",
            "currency": "$",
            "expected": "FLAGGED (High Risk, ~85/100, Duplicate Collision with Sample 01)",
        },
        {
            "filename_stem": "sample_04_missing_fields",
            "title": "",  # Missing Vendor
            "address": "",
            "inv_num": "",  # Missing Invoice #
            "date_str": "",  # Missing Date
            "due_date_str": "",
            "bill_to": "Customer Account #99",
            "bill_address": "",
            "items": [
                ("Unspecified General Equipment", "1", "400.00", "400.00"),
            ],
            "subtotal": "400.00",
            "tax": "50.00",
            "discount": "",
            "total": "450.00",
            "currency": "$",
            "expected": "WARNING (Medium Risk, ~60/100, Missing Mandatory Fields)",
        },
        {
            "filename_stem": "sample_05_with_discount_tax",
            "title": "VERTEX DATA SYSTEMS",
            "address": "100 Cloud Park, Seattle, WA 98101",
            "inv_num": "INV-2026-005",
            "date_str": "2026-11-01",
            "due_date_str": "2026-12-01",
            "bill_to": "FinVision Enterprise LLC",
            "bill_address": "789 Financial Square, Floor 12, New York, NY 10005",
            "items": [
                ("High-Performance GPU Compute Instances", "5", "200.00", "1,000.00"),
            ],
            "subtotal": "1,000.00",
            "tax": "76.00",
            "discount": "50.00",
            "total": "1,026.00",  # Math: 1000 - 50 + 76 = 1026
            "currency": "$",
            "expected": "PASS (Low Risk, ~5/100, Verified Discount & Tax)",
        },
        {
            "filename_stem": "sample_06_european_format",
            "title": "EUROTECH CONSULTING GMBH",
            "address": "Friedrichstraße 42, 10117 Berlin, Germany",
            "inv_num": "INV-DE-8841",
            "date_str": "15.10.2026",
            "due_date_str": "15.11.2026",
            "bill_to": "FinVision European Branch",
            "bill_address": "Königsallee 10, Düsseldorf",
            "items": [
                ("Softwarearchitektur Beratung", "1", "1.000,00", "1.000,00"),
            ],
            "subtotal": "1.000,00",
            "tax": "190,00",
            "discount": "",
            "total": "1.190,00",
            "currency": "EUR",
            "expected": "PASS (Low Risk, ~5/100, Verified European Decimal Format)",
        },
    ]

    for s in samples_meta:
        doc = create_invoice_doc(
            title=s["title"],
            address=s["address"],
            inv_num=s["inv_num"],
            date_str=s["date_str"],
            due_date_str=s["due_date_str"],
            bill_to=s["bill_to"],
            bill_address=s["bill_address"],
            items=s["items"],
            subtotal_str=s["subtotal"],
            tax_str=s["tax"],
            discount_str=s["discount"],
            total_str=s["total"],
            currency_symbol=s["currency"],
        )

        pdf_path = SAMPLE_DIR / f"{s['filename_stem']}.pdf"
        doc.save(str(pdf_path))

        # Also render high-res PNG
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        png_path = SAMPLE_DIR / f"{s['filename_stem']}.png"
        pix.save(str(png_path))
        doc.close()
        print(f"Generated: {pdf_path.name} and {png_path.name}")

    # Create README in sample_invoices/
    readme_path = SAMPLE_DIR / "README.md"
    readme_content = """# FinVision Verification Test Pack

This directory contains 6 representative test invoices in both PDF and PNG formats, designed to verify all validation rules, duplicate collision mechanisms, and risk calculations of the FinVision platform.

## Test Invoices Overview

| File | Scenario | Key Traits | Expected Outcome |
| :--- | :--- | :--- | :--- |
| `sample_01_valid_invoice` | **Clean Valid Invoice** | Subtotal: $1,000.00, Tax: $100.00, Total: $1,100.00 | **PASS (Low Risk, ~5/100)** |
| `sample_02_math_mismatch` | **Arithmetic Discrepancy** | Subtotal: $500.00, Tax: $50.00, Stated Total: $850.00 ($300 discrepancy) | **FLAGGED (High Risk, ~75/100)** |
| `sample_03_duplicate_invoice` | **Duplicate Collision** | Identical invoice number (`INV-2026-001`) and vendor as sample 01 | **FLAGGED (High Risk, ~85/100)** |
| `sample_04_missing_fields` | **Missing Mandatory Fields** | Omitted vendor name, missing date, missing invoice number | **WARNING (Medium Risk, ~60/100)** |
| `sample_05_with_discount_tax` | **Discount & Tax Formula** | Subtotal $1,000.00 - Discount $50.00 + Tax $76.00 = Total $1,026.00 | **PASS (Low Risk, ~5/100)** |
| `sample_06_european_format` | **European Currency & Decimals** | EUR currency, dot/comma separators (`1.000,00 €` + `190,00 €` = `1.190,00 €`) | **PASS (Low Risk, ~5/100)** |

You can test these files either directly in the web UI using the **⚡ Test Invoices Pack** selector, or upload them individually via drag-and-drop.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Generated: {readme_path.name}")


def create_sample_zip() -> bytes:
    """Generate in-memory zip archive containing all sample invoices."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in SAMPLE_DIR.iterdir():
            if p.is_file():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    generate_all_samples()
