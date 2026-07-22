"""Tests for Auth and Audit Services (F18)."""

import tempfile
from pathlib import Path

import pytest

from services.auth import AuthService, _hash_password, _verify_password, _generate_token, _verify_token
from services.audit import AuditLog


# =============================================================================
# Auth Tests
# =============================================================================

@pytest.fixture
def temp_auth():
    """AuthService with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_auth.db"
        auth = AuthService(db_path=db_path)
        yield auth


class TestPasswordHashing:
    """Test password hashing."""

    def test_hash_returns_tuple(self):
        result = _hash_password("test123")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hash_deterministic(self):
        h1, salt1 = _hash_password("test123")
        h2, _ = _hash_password("test123", salt1)
        assert h1 == h2

    def test_verify_correct(self):
        h, salt = _hash_password("mypassword")
        assert _verify_password("mypassword", h, salt) is True

    def test_verify_wrong(self):
        h, salt = _hash_password("mypassword")
        assert _verify_password("wrongpassword", h, salt) is False


class TestTokenOperations:
    """Test token generation and verification."""

    def test_generate_and_verify(self):
        token = _generate_token("user-1", "drsmith", "doctor")
        payload = _verify_token(token)
        assert payload is not None
        assert payload["user_id"] == "user-1"
        assert payload["username"] == "drsmith"
        assert payload["role"] == "doctor"

    def test_invalid_token(self):
        assert _verify_token("invalid.token.here") is None

    def test_tampered_token(self):
        token = _generate_token("user-1", "drsmith", "doctor")
        # Tamper with the token
        parts = token.split(".")
        tampered = parts[0] + ".0000" + parts[1][4:]
        assert _verify_token(tampered) is None


class TestUserManagement:
    """Test user CRUD."""

    def test_create_user(self, temp_auth):
        result = temp_auth.create_user("drsmith", "password123", role="doctor")
        assert result["username"] == "drsmith"
        assert result["role"] == "doctor"

    def test_create_duplicate_user(self, temp_auth):
        temp_auth.create_user("drsmith", "password123")
        with pytest.raises(ValueError, match="already exists"):
            temp_auth.create_user("drsmith", "another_password")

    def test_authenticate_success(self, temp_auth):
        temp_auth.create_user("drsmith", "password123")
        result = temp_auth.authenticate("drsmith", "password123")
        assert result is not None
        assert "token" in result
        assert result["user"]["username"] == "drsmith"

    def test_authenticate_wrong_password(self, temp_auth):
        temp_auth.create_user("drsmith", "password123")
        result = temp_auth.authenticate("drsmith", "wrongpassword")
        assert result is None

    def test_authenticate_nonexistent_user(self, temp_auth):
        result = temp_auth.authenticate("nobody", "password")
        assert result is None

    def test_list_users(self, temp_auth):
        temp_auth.create_user("dr1", "pass1")
        temp_auth.create_user("dr2", "pass2")
        users = temp_auth.list_users()
        assert len(users) == 2
        # Should not contain password hashes
        for u in users:
            assert "password_hash" not in u
            assert "salt" not in u

    def test_deactivate_user(self, temp_auth):
        user = temp_auth.create_user("drsmith", "password123")
        assert temp_auth.deactivate_user(user["id"]) is True
        # Should not authenticate after deactivation
        result = temp_auth.authenticate("drsmith", "password123")
        assert result is None

    def test_deactivate_nonexistent(self, temp_auth):
        assert temp_auth.deactivate_user("nonexistent") is False


class TestRBAC:
    """Test role-based access control."""

    def test_doctor_role(self, temp_auth):
        assert temp_auth.has_role({"role": "doctor"}, "doctor") is True
        assert temp_auth.has_role({"role": "doctor"}, "admin") is False

    def test_admin_role(self, temp_auth):
        assert temp_auth.has_role({"role": "admin"}, "doctor") is True
        assert temp_auth.has_role({"role": "admin"}, "admin") is True


# =============================================================================
# Audit Tests
# =============================================================================

@pytest.fixture
def temp_audit():
    """AuditLog with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_audit.db"
        audit = AuditLog(db_path=db_path)
        yield audit


class TestAuditLogging:
    """Test audit log entries."""

    def test_log_entry(self, temp_audit):
        result = temp_audit.log(action="login", user_id="u1", username="drsmith")
        assert "id" in result
        assert result["action"] == "login"
        assert result["entry_hash"]

    def test_log_with_details(self, temp_audit):
        result = temp_audit.log(
            action="save_record",
            user_id="u1",
            resource_type="record",
            resource_id="rec-123",
            details="Saved consultation note",
        )
        assert result["resource_type"] == "record"
        assert result["resource_id"] == "rec-123"

    def test_count(self, temp_audit):
        temp_audit.log(action="a1")
        temp_audit.log(action="a2")
        assert temp_audit.count() == 2


class TestAuditChain:
    """Test hash chain integrity."""

    def test_empty_chain_valid(self, temp_audit):
        is_valid, count = temp_audit.verify_chain()
        assert is_valid is True
        assert count == 0

    def test_single_entry_valid(self, temp_audit):
        temp_audit.log(action="test")
        is_valid, count = temp_audit.verify_chain()
        assert is_valid is True
        assert count == 1

    def test_multiple_entries_valid(self, temp_audit):
        for i in range(5):
            temp_audit.log(action=f"action_{i}")
        is_valid, count = temp_audit.verify_chain()
        assert is_valid is True
        assert count == 5

    def test_chain_hash_chained(self, temp_audit):
        """Each entry's previous_hash should match the prior entry's hash."""
        temp_audit.log(action="first")
        temp_audit.log(action="second")
        temp_audit.log(action="third")

        # Query entries in chronological order (ASC)
        import sqlite3
        conn = sqlite3.connect(str(temp_audit._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM audit_log ORDER BY rowid ASC").fetchall()
        conn.close()
        entries = [dict(r) for r in rows]

        assert entries[1]["previous_hash"] == entries[0]["entry_hash"]
        assert entries[2]["previous_hash"] == entries[1]["entry_hash"]


class TestAuditQuery:
    """Test audit log querying."""

    def test_query_all(self, temp_audit):
        temp_audit.log(action="login", user_id="u1")
        temp_audit.log(action="save", user_id="u2")
        results = temp_audit.query()
        assert len(results) == 2

    def test_query_by_user(self, temp_audit):
        temp_audit.log(action="login", user_id="u1")
        temp_audit.log(action="login", user_id="u2")
        results = temp_audit.query(user_id="u1")
        assert len(results) == 1

    def test_query_by_action(self, temp_audit):
        temp_audit.log(action="login", user_id="u1")
        temp_audit.log(action="save", user_id="u1")
        results = temp_audit.query(action="login")
        assert len(results) == 1

    def test_query_limit(self, temp_audit):
        for i in range(10):
            temp_audit.log(action=f"action_{i}")
        results = temp_audit.query(limit=3)
        assert len(results) == 3


class TestAuditEdgeCases:
    """Test edge cases."""

    def test_unicode_details(self, temp_audit):
        result = temp_audit.log(action="test", details="Patient notes in Hindi: नमस्ते")
        assert result is not None
        # Verify details stored by querying
        entries = temp_audit.query(action="test")
        assert entries[0]["details"] == "Patient notes in Hindi: नमस्ते"

    def test_long_details(self, temp_audit):
        long_details = "details " * 500
        result = temp_audit.log(action="test", details=long_details)
        assert result is not None

    def test_genesis_hash(self, temp_audit):
        """First entry should chain from 'genesis'."""
        temp_audit.log(action="first")
        entries = temp_audit.query()
        assert entries[0]["previous_hash"] == "genesis"
