"""Tests for insult.core.errors — in-character error responses."""

from insult.core.errors import classify_error, get_error_response


class TestGetErrorResponse:
    def test_returns_string(self):
        assert isinstance(get_error_response("generic"), str)

    def test_timeout_response(self):
        resp = get_error_response("timeout")
        assert len(resp) > 10

    def test_rate_limit_response(self):
        resp = get_error_response("rate_limit")
        assert len(resp) > 10

    def test_auth_response(self):
        resp = get_error_response("auth")
        assert len(resp) > 10

    def test_too_long_response(self):
        resp = get_error_response("too_long")
        assert len(resp) > 10

    def test_context_failed_response(self):
        resp = get_error_response("context_failed")
        assert len(resp) > 10

    def test_unknown_type_falls_back_to_generic(self):
        resp = get_error_response("nonexistent_type")
        assert isinstance(resp, str)
        assert len(resp) > 10

    def test_no_response_mentions_claude(self):
        """Insult NEVER reveals the underlying model in errors."""
        for error_type in ["timeout", "rate_limit", "auth", "generic", "too_long", "context_failed"]:
            for _ in range(10):  # sample random responses
                resp = get_error_response(error_type)
                assert "Claude" not in resp
                assert "claude" not in resp
                assert "Anthropic" not in resp
                assert "API" not in resp


class TestClassifyError:
    def test_timeout_error(self):
        class FakeTimeoutError(Exception):
            pass

        FakeTimeoutError.__name__ = "APITimeoutError"
        assert classify_error(FakeTimeoutError()) == "timeout"

    def test_rate_limit_error(self):
        class FakeRateLimitError(Exception):
            pass

        FakeRateLimitError.__name__ = "RateLimitError"
        assert classify_error(FakeRateLimitError()) == "rate_limit"

    def test_auth_error(self):
        class FakeAuthError(Exception):
            pass

        FakeAuthError.__name__ = "AuthenticationError"
        assert classify_error(FakeAuthError()) == "auth"

    def test_generic_error(self):
        assert classify_error(ValueError("something")) == "generic"

    def test_unknown_error(self):
        assert classify_error(RuntimeError("boom")) == "generic"
