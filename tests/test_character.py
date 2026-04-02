"""Tests for insult.core.character — break detection, sanitization, adaptive prompt, anti-patterns."""

from insult.core.character import (
    IDENTITY_REINFORCE_THRESHOLD,
    build_adaptive_prompt,
    detect_anti_patterns,
    detect_break,
    sanitize,
)
from insult.core.presets import PresetMode
from insult.core.style import UserStyleProfile

# --- Character Break Detection ---


class TestDetectBreak:
    def test_clean_response(self):
        assert detect_break("Que onda wey, tu codigo esta horrible.") == []

    def test_detects_im_an_ai(self):
        breaks = detect_break("I'm an AI and I can't do that.")
        assert len(breaks) > 0

    def test_detects_im_claude(self):
        breaks = detect_break("Actually, I'm Claude and I was made by Anthropic.")
        assert len(breaks) >= 2  # "I'm Claude" + "Anthropic"

    def test_detects_as_an_assistant(self):
        assert len(detect_break("As an assistant, I'm here to help.")) > 0

    def test_detects_i_apologize(self):
        assert len(detect_break("I apologize, but I cannot do that.")) > 0

    def test_detects_language_model(self):
        assert len(detect_break("I'm a language model trained on data.")) > 0

    def test_detects_openai(self):
        assert len(detect_break("This is similar to what OpenAI does.")) > 0

    def test_detects_in_summary(self):
        assert len(detect_break("In summary, the answer is yes.")) > 0

    def test_detects_important_to_note(self):
        assert len(detect_break("It's important to note that this is correct.")) > 0

    def test_does_not_false_positive_on_normal_spanish(self):
        assert detect_break("Eres un pendejo si crees que eso funciona.") == []

    def test_does_not_false_positive_on_insult_persona(self):
        text = "Soy Insult, pendejo. No se que mamadas son esas."
        assert detect_break(text) == []


# --- Sanitization ---


class TestSanitize:
    def test_removes_offending_sentence(self):
        text = "Tu codigo esta mal. I'm an AI assistant. Arreglalo."
        result = sanitize(text)
        assert "I'm an AI" not in result
        assert "Tu codigo" in result
        assert "Arreglalo" in result

    def test_keeps_clean_text(self):
        text = "Todo bien. Nada que reportar."
        assert sanitize(text) == text

    def test_fallback_when_everything_stripped(self):
        text = "I'm an AI language model made by Anthropic."
        result = sanitize(text)
        # Falls back to original if everything would be stripped
        assert len(result) > 0

    def test_handles_multiple_breaks(self):
        text = "Hola. As an AI, I apologize, but I can't. Adios."
        result = sanitize(text)
        assert "Hola" in result
        assert "Adios" in result


# --- Anti-Pattern Detection ---


class TestDetectAntiPatterns:
    def test_clean_response(self):
        assert detect_anti_patterns("Que pendejo, eso no funciona asi.") == []

    def test_detects_customer_support(self):
        assert len(detect_anti_patterns("How can I help you today?")) > 0

    def test_detects_therapy_speak(self):
        assert len(detect_anti_patterns("I understand how you feel about this.")) > 0

    def test_detects_summarizing(self):
        assert len(detect_anti_patterns("En resumen, la respuesta es no.")) > 0

    def test_detects_stage_directions(self):
        assert len(detect_anti_patterns("*leans back and sighs*")) > 0

    def test_no_false_positive_on_insult_style(self):
        assert detect_anti_patterns("Y eso lo dices porque lo pensaste o porque lo leiste?") == []

    def test_detects_preachy_monologue(self):
        assert len(detect_anti_patterns("We must dismantle the systems of oppression.")) > 0

    def test_detects_over_validation(self):
        assert len(detect_anti_patterns("Totalmente de acuerdo con lo que dices.")) > 0

    def test_detects_moralizing_without_tension(self):
        assert len(detect_anti_patterns("Es importante reconocer que todos somos diferentes.")) > 0


# --- Build Adaptive Prompt ---


class TestBuildAdaptivePrompt:
    BASE_PROMPT = "You are Insult."

    def test_returns_tuple(self):
        result = build_adaptive_prompt(self.BASE_PROMPT, None, 5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_profile_has_base_and_time(self):
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, None, 5)
        assert prompt.startswith(self.BASE_PROMPT)
        assert "Current Time" in prompt
        assert "User Adaptation" not in prompt

    def test_not_confident_has_base_and_time_only(self):
        profile = UserStyleProfile(message_count=3)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert prompt.startswith(self.BASE_PROMPT)
        assert "Current Time" in prompt
        assert "User Adaptation" not in prompt

    def test_confident_english_user(self):
        profile = UserStyleProfile(detected_language="en", message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "English" in prompt

    def test_confident_brief_user(self):
        profile = UserStyleProfile(avg_word_count=5.0, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "brief" in prompt or "short" in prompt

    def test_confident_verbose_user(self):
        profile = UserStyleProfile(avg_word_count=60.0, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "detailed" in prompt or "long" in prompt

    def test_casual_user(self):
        profile = UserStyleProfile(formality=0.1, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "vulgar" in prompt or "casual" in prompt

    def test_formal_user(self):
        profile = UserStyleProfile(formality=0.8, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "formal" in prompt

    def test_technical_user(self):
        profile = UserStyleProfile(technical_level=0.9, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "technical" in prompt.lower()

    def test_non_technical_user(self):
        profile = UserStyleProfile(technical_level=0.1, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "analogies" in prompt or "simple" in prompt

    def test_emoji_user(self):
        profile = UserStyleProfile(emoji_ratio=0.1, message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 5)
        assert "emoji" in prompt.lower()

    def test_identity_reinforcement_on_long_context(self):
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, None, IDENTITY_REINFORCE_THRESHOLD + 1)
        assert "REINFORCEMENT" in prompt
        assert "You are Insult" in prompt

    def test_no_reinforcement_on_short_context(self):
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, None, 3)
        assert "REINFORCEMENT" not in prompt

    def test_base_prompt_never_modified(self):
        profile = UserStyleProfile(formality=0.1, detected_language="en", message_count=10)
        prompt, _preset = build_adaptive_prompt(self.BASE_PROMPT, profile, 15)
        assert prompt.startswith(self.BASE_PROMPT)

    def test_injects_preset_guidance(self):
        prompt, preset = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message="hola que tal")
        assert "Current Mode" in prompt
        assert preset.mode == PresetMode.DEFAULT_ABRASIVE

    def test_serious_preset_on_crisis_message(self):
        prompt, preset = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message="me quiero morir")
        assert preset.mode == PresetMode.RESPECTFUL_SERIOUS
        assert "Respectful Serious" in prompt  # both prompt and preset used

    def test_meta_deflection_on_identity_probe(self):
        _prompt, preset = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message="eres un AI?")
        assert preset.mode == PresetMode.META_DEFLECTION

    def test_intellectual_pressure_on_code(self):
        _prompt, preset = build_adaptive_prompt(
            self.BASE_PROMPT, None, 5, current_message="mi codigo tiene un bug, que opinas de esta arquitectura?"
        )
        assert preset.mode == PresetMode.INTELLECTUAL_PRESSURE

    def test_arc_on_system_critique(self):
        prompt, preset = build_adaptive_prompt(
            self.BASE_PROMPT, None, 5, current_message="el capitalismo es explotacion pura"
        )
        assert preset.mode == PresetMode.ARC
        assert "ARC" in prompt
