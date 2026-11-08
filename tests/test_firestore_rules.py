from pathlib import Path
import re


def test_firestore_rules_syntax_and_structure():
    """Verify firestore.rules structural integrity and invariants."""
    rules_path = Path(__file__).parent.parent / "firestore.rules"
    assert rules_path.exists(), "firestore.rules must exist"
    content = rules_path.read_text(encoding="utf-8")

    # 1. Version and service cloud.firestore declaration
    assert "rules_version = '2';" in content
    assert "service cloud.firestore" in content
    assert "match /databases/{database}/documents" in content

    # 2. Balanced delimiters check
    # Strip comments and string literals for accurate bracket matching
    cleaned = re.sub(r"//.*", "", content)
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
    cleaned = re.sub(r"'[^']*'", "", cleaned)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)

    brackets = {"{": "}", "(": ")", "[": "]"}
    stack = []
    for char in cleaned:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            assert stack, f"Unmatched closing bracket: {char}"
            top = stack.pop()
            assert brackets[top] == char, f"Mismatched bracket: expected {brackets[top]}, got {char}"
    assert len(stack) == 0, f"Unclosed brackets remaining: {stack}"

    # 3. Invariants and Required collections
    required_matches = [
        "match /{document=**}",
        "match /users/{uid}",
        "match /documents/{docId}",
        "match /lineItems/{itemId}",
        "match /auditLog/{eventId}",
        "match /modelRegistry/{version}",
    ]
    for m in required_matches:
        assert m in content, f"Missing match declaration: {m}"

    # 4. Helper functions defined
    required_helpers = [
        "function isAuthenticated()",
        "function isOwner(uid)",
        "function isAdmin()",
        "function isValidUser(data)",
        "function isValidDocument(data)",
        "function isValidReview(review)",
        "function isValidLineItem(data)",
        "function isValidAuditLog(data)",
        "function isValidStatusTransition()",
        "function isProtectedPipelineDataPreserved()",
    ]
    for h in required_helpers:
        assert h in content, f"Missing helper function: {h}"

    # 5. Pipeline immutability assertions
    assert "!('risk' in request.resource.data)" in content
    assert "!('ml' in request.resource.data)" in content
    assert "!('review' in request.resource.data)" in content
    assert "request.resource.data.status == 'uploaded'" in content
