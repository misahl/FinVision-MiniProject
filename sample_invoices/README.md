# FinVision Verification Test Pack

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
