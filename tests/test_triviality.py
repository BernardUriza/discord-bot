"""Tests for trivial-message detection (cost guard in chat.py)."""

from insult.core.triviality import is_trivial


class TestIsTrivialTrue:
    def test_empty_string(self):
        assert is_trivial("")

    def test_only_whitespace(self):
        assert is_trivial("   \n\t  ")

    def test_single_ok(self):
        assert is_trivial("ok")

    def test_single_si(self):
        assert is_trivial("si")
        assert is_trivial("sí")

    def test_single_no(self):
        assert is_trivial("no")

    def test_gracias(self):
        assert is_trivial("gracias")

    def test_jaja_variants(self):
        assert is_trivial("jaja")
        assert is_trivial("jajaja")
        assert is_trivial("jeje")

    def test_xd(self):
        assert is_trivial("xd")
        assert is_trivial("xdd")
        assert is_trivial("XDDD")

    def test_only_emoji(self):
        assert is_trivial("😂")
        assert is_trivial("🔥🔥🔥")
        assert is_trivial("💀")

    def test_short_no_digits(self):
        assert is_trivial("uff")
        assert is_trivial("mm")

    def test_multi_word_all_trivial(self):
        assert is_trivial("ok gracias")
        assert is_trivial("si ya")
        assert is_trivial("no jaja")

    def test_case_insensitive(self):
        assert is_trivial("OK")
        assert is_trivial("Gracias")
        assert is_trivial("SI")

    def test_punctuation_stripped(self):
        assert is_trivial("ok.")
        assert is_trivial("gracias!")
        assert is_trivial("si?")


class TestIsTrivialFalse:
    def test_real_question(self):
        assert not is_trivial("¿qué opinas?")

    def test_statement_with_digits(self):
        # "2+2=4" is arguably substantive — digits bypass short-word trivial
        assert not is_trivial("2+2")

    def test_long_message(self):
        assert not is_trivial("hola, quiero hablar sobre algo importante")

    def test_mixed_trivial_and_content(self):
        # "ok, pero por qué" contains non-trivial token "pero"
        assert not is_trivial("ok pero por qué")

    def test_four_char_word(self):
        # Exactly at threshold — substantive
        assert not is_trivial("hola")

    def test_emoji_plus_text(self):
        assert not is_trivial("💀 qué onda")

    def test_single_substantive_word(self):
        assert not is_trivial("insultame")
        assert not is_trivial("explica")
