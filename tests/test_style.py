"""Tests for insult.core.style — pure Python, zero mocks needed."""

import pytest

from insult.core.style import (
    UserStyleProfile,
    _compute_formality,
    _compute_technical_level,
    _detect_language,
)

# --- Language Detection ---


class TestDetectLanguage:
    def test_spanish_text(self):
        assert _detect_language("hola que tal como estas") == "es"

    def test_english_text(self):
        assert _detect_language("hello how are you doing today") == "en"

    def test_mixed_defaults_to_higher_score(self):
        result = _detect_language("the server no funciona en el deploy")
        assert result in ("es", "en")  # both are valid, depends on stopword count

    def test_empty_defaults_to_es(self):
        assert _detect_language("xyz abc") == "es"

    def test_spanish_slang(self):
        assert _detect_language("pues no se que onda con eso") == "es"


# --- Formality ---


class TestFormality:
    def test_casual_spanish(self):
        score = _compute_formality("wey neta nmms que pedo")
        assert score < 0.3

    def test_casual_english(self):
        score = _compute_formality("lol bruh idk gonna wanna")
        assert score < 0.3

    def test_formal_english(self):
        score = _compute_formality("Therefore, I would furthermore like to address this matter.")
        assert score > 0.5

    def test_neutral_text(self):
        score = _compute_formality("hello world")
        assert score == 0.5  # no markers = neutral

    def test_proper_punctuation_adds_formality(self):
        score = _compute_formality("This is a proper sentence.")
        assert score > 0.5


# --- Technical Level ---


class TestTechnicalLevel:
    def test_non_technical(self):
        score = _compute_technical_level("me gusta el helado de chocolate")
        assert score < 0.2

    def test_highly_technical(self):
        score = _compute_technical_level("deploy the docker api server to aws endpoint")
        assert score > 0.3

    def test_code_references(self):
        score = _compute_technical_level("the function foo() uses async await import")
        assert score > 0.3

    def test_code_blocks(self):
        score = _compute_technical_level("here is the code ```python\nprint('hello')```")
        assert score > 0.0


# --- UserStyleProfile ---


class TestUserStyleProfile:
    def test_default_values(self):
        p = UserStyleProfile()
        assert p.avg_word_count == 15.0
        assert p.formality == 0.5
        assert p.message_count == 0
        assert p.is_confident is False

    def test_confidence_after_5_messages(self):
        p = UserStyleProfile()
        for i in range(5):
            p.update(f"test message number {i}")
        assert p.is_confident is True

    def test_not_confident_at_4_messages(self):
        p = UserStyleProfile()
        for i in range(4):
            p.update(f"test message {i}")
        assert p.is_confident is False

    def test_update_changes_word_count(self):
        p = UserStyleProfile()
        p.update("one two three")  # 3 words
        # EMA: 0.3 * 3 + 0.7 * 15 = 11.4
        assert p.avg_word_count == pytest.approx(11.4, abs=0.1)

    def test_ema_smoothing_converges(self):
        """Send many short messages — avg should converge toward short."""
        p = UserStyleProfile()
        for _ in range(20):
            p.update("short msg")
        assert p.avg_word_count < 5.0

    def test_language_detection_in_update(self):
        p = UserStyleProfile()
        p.update("hello how are you doing today my friend")
        assert p.detected_language == "en"

    def test_serialization_roundtrip(self):
        p = UserStyleProfile(avg_word_count=25.3, formality=0.2, message_count=10)
        json_str = p.to_json()
        restored = UserStyleProfile.from_json(json_str)
        assert restored.avg_word_count == 25.3
        assert restored.formality == 0.2
        assert restored.message_count == 10

    def test_to_dict(self):
        p = UserStyleProfile(detected_language="en", technical_level=0.8)
        d = p.to_dict()
        assert d["detected_language"] == "en"
        assert d["technical_level"] == 0.8

    def test_from_dict_with_defaults(self):
        p = UserStyleProfile.from_dict({})
        assert p.avg_word_count == 15.0
        assert p.detected_language == "es"
