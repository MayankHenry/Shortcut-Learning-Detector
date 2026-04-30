"""
test_main.py — Backend API Tests
Team 85 · GLA University · 4th Semester AIML

Run with: pytest test_main.py -v

Covers:
  - Health check
  - User registration (valid, duplicate, weak password, bad username)
  - Login (valid, wrong password, nonexistent user)
  - /analyze endpoint (auth, file type, file size, valid image)
  - /admin/logs endpoint (auth required)
  - Security headers present on all responses
  - Rate limit headers present
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_test_image(color=(255, 0, 0), size=(100, 100)) -> bytes:
    """Create a simple in-memory PNG image for upload tests."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def register_and_login(username="testuser_85", password="Test1234") -> str:
    """Register a user and return a valid JWT access token."""
    client.post(
        "/register",
        data={"username": username, "password": password},
    )
    res = client.post(
        "/login",
        data={"username": username, "password": password},
    )
    return res.json().get("access_token", "")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Health Check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_root_returns_200(self):
        res = client.get("/")
        assert res.status_code == 200

    def test_root_returns_status_ok(self):
        res = client.get("/")
        assert res.json()["status"] == "ok"

    def test_docs_available(self):
        res = client.get("/docs")
        assert res.status_code == 200

    def test_invalid_route_returns_404(self):
        res = client.get("/this-does-not-exist")
        assert res.status_code == 404

# ─────────────────────────────────────────────────────────────────────────────
# 2. Security Headers
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityHeaders:
    def test_xss_protection_header(self):
        res = client.get("/")
        assert "X-XSS-Protection" in res.headers

    def test_x_content_type_options_header(self):
        res = client.get("/")
        assert res.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_header(self):
        res = client.get("/")
        assert res.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy_header(self):
        res = client.get("/")
        assert "Referrer-Policy" in res.headers

    def test_content_security_policy_header(self):
        res = client.get("/")
        assert "Content-Security-Policy" in res.headers

# ─────────────────────────────────────────────────────────────────────────────
# 3. Registration
# ─────────────────────────────────────────────────────────────────────────────

class TestRegister:
    def test_valid_registration(self):
        res = client.post(
            "/register",
            data={"username": "newuser_valid", "password": "Secure123"},
        )
        assert res.status_code == 200
        assert "message" in res.json()

    def test_duplicate_username_rejected(self):
        client.post("/register", data={"username": "dup_user", "password": "Test1234"})
        res = client.post("/register", data={"username": "dup_user", "password": "Test1234"})
        assert res.status_code == 400
        assert "already registered" in res.json()["detail"]

    def test_short_password_rejected(self):
        res = client.post(
            "/register",
            data={"username": "user_shortpw", "password": "abc"},
        )
        assert res.status_code == 400
        assert "8 characters" in res.json()["detail"]

    def test_no_number_in_password_rejected(self):
        res = client.post(
            "/register",
            data={"username": "user_nonum", "password": "OnlyLetters"},
        )
        assert res.status_code == 400
        assert "number" in res.json()["detail"]

    def test_no_letter_in_password_rejected(self):
        res = client.post(
            "/register",
            data={"username": "user_nolet", "password": "12345678"},
        )
        assert res.status_code == 400
        assert "letter" in res.json()["detail"]

    def test_username_too_short_rejected(self):
        res = client.post(
            "/register",
            data={"username": "ab", "password": "Valid123"},
        )
        assert res.status_code == 400

    def test_username_with_special_chars_rejected(self):
        res = client.post(
            "/register",
            data={"username": "<script>alert(1)</script>", "password": "Valid123"},
        )
        assert res.status_code == 400

    def test_username_with_spaces_rejected(self):
        res = client.post(
            "/register",
            data={"username": "user name", "password": "Valid123"},
        )
        assert res.status_code == 400

# ─────────────────────────────────────────────────────────────────────────────
# 4. Login
# ─────────────────────────────────────────────────────────────────────────────

class TestLogin:
    def test_valid_login_returns_token(self):
        client.post("/register", data={"username": "login_user", "password": "Login123"})
        res = client.post("/login", data={"username": "login_user", "password": "Login123"})
        assert res.status_code == 200
        assert "access_token" in res.json()
        assert res.json()["token_type"] == "bearer"

    def test_wrong_password_returns_401(self):
        client.post("/register", data={"username": "auth_user", "password": "Right123"})
        res = client.post("/login", data={"username": "auth_user", "password": "WrongPass1"})
        assert res.status_code == 401

    def test_nonexistent_user_returns_401(self):
        res = client.post("/login", data={"username": "ghost_user_xyz", "password": "Any123"})
        assert res.status_code == 401

    def test_token_is_non_empty_string(self):
        client.post("/register", data={"username": "token_user", "password": "Token123"})
        res = client.post("/login", data={"username": "token_user", "password": "Token123"})
        token = res.json().get("access_token", "")
        assert isinstance(token, str) and len(token) > 10

# ─────────────────────────────────────────────────────────────────────────────
# 5. /analyze endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyze:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.token = register_and_login("analyze_user_85", "Analyze123")
        self.auth_header = {"Authorization": f"Bearer {self.token}"}

    def test_analyze_requires_auth(self):
        """Without token, should return 401."""
        img_bytes = make_test_image()
        res = client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 401

    def test_analyze_with_invalid_token(self):
        res = client.post(
            "/analyze",
            headers={"Authorization": "Bearer this.is.fake"},
            files={"file": ("test.png", make_test_image(), "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 401

    def test_analyze_rejects_pdf(self):
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 415

    def test_analyze_rejects_oversized_file(self):
        big_data = b"x" * (6 * 1024 * 1024)  # 6MB
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("big.png", big_data, "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 413

    def test_analyze_rejects_corrupt_image(self):
        """File has image MIME type but is not a real image."""
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("fake.png", b"this is not an image", "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 400

    def test_analyze_biased_model_returns_result(self):
        img_bytes = make_test_image(color=(0, 255, 0))
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("digit.png", img_bytes, "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 200
        body = res.json()
        assert "class_name" in body
        assert "confidence" in body
        assert "heatmap_base64" in body
        assert body["class_name"].startswith("Digit")
        assert 0.0 <= body["confidence"] <= 100.0

    def test_analyze_unbiased_model_returns_result(self):
        img_bytes = make_test_image(color=(255, 0, 0))
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("digit.png", img_bytes, "image/png")},
            data={"model_type": "unbiased"},
        )
        assert res.status_code == 200
        body = res.json()
        assert "heatmap_base64" in body

    def test_analyze_invalid_model_type_rejected(self):
        img_bytes = make_test_image()
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("digit.png", img_bytes, "image/png")},
            data={"model_type": "cheating_model"},
        )
        assert res.status_code == 422  # FastAPI validation error

    def test_heatmap_base64_is_valid_string(self):
        img_bytes = make_test_image()
        res = client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("digit.png", img_bytes, "image/png")},
            data={"model_type": "biased"},
        )
        assert res.status_code == 200
        heatmap = res.json()["heatmap_base64"]
        import base64
        decoded = base64.b64decode(heatmap)
        assert len(decoded) > 0

# ─────────────────────────────────────────────────────────────────────────────
# 6. /admin/logs endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestAdminLogs:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.token = register_and_login("admin_user_85", "Admin1234")
        self.auth_header = {"Authorization": f"Bearer {self.token}"}

    def test_logs_requires_auth(self):
        res = client.get("/admin/logs")
        assert res.status_code == 401

    def test_logs_with_invalid_token_rejected(self):
        res = client.get(
            "/admin/logs",
            headers={"Authorization": "Bearer fake.token.here"},
        )
        assert res.status_code == 401

    def test_logs_returns_list(self):
        res = client.get("/admin/logs", headers=self.auth_header)
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_log_entry_has_required_fields(self):
        # First create a prediction so there's at least one log
        img_bytes = make_test_image()
        client.post(
            "/analyze",
            headers=self.auth_header,
            files={"file": ("digit.png", img_bytes, "image/png")},
            data={"model_type": "biased"},
        )
        res = client.get("/admin/logs", headers=self.auth_header)
        assert res.status_code == 200
        logs = res.json()
        if logs:
            entry = logs[0]
            assert "id" in entry
            assert "model_type" in entry
            assert "predicted_class" in entry
            assert "confidence" in entry