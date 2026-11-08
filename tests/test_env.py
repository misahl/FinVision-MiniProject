import sys
import os
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def test_python_version():
    """Verify that Python 3.12 is used."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12, f"Expected Python 3.12, got {sys.version}"


def test_tesseract_installed():
    """Verify that Tesseract OCR binary is reachable and returns version."""
    tess_cmd = os.getenv("TESSERACT_CMD", "tesseract")
    res = subprocess.run([tess_cmd, "--version"], capture_output=True, text=True)
    assert res.returncode == 0, f"Tesseract failed: {res.stderr}"
    assert "tesseract" in res.stdout.lower()


def test_configs_valid_json():
    """Verify configuration files are syntactically valid JSON."""
    root = Path(__file__).parent.parent
    for cfg in ["rules_config.json", "risk_weights.json"]:
        p = root / "config" / cfg
        assert p.exists(), f"Missing config file {cfg}"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)


def test_firebase_files_exist():
    """Verify Firebase config files exist."""
    root = Path(__file__).parent.parent
    for fname in ["firebase.json", ".firebaserc", "storage.rules", "firestore.rules", "firestore.indexes.json"]:
        p = root / fname
        assert p.exists(), f"Missing {fname}"
        assert p.stat().st_size > 0, f"Empty {fname}"


