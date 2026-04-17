"""Tests for insult.core.character — break detection, sanitization, adaptive prompt, anti-patterns."""

from insult.core.character import (
    IDENTITY_REINFORCE_THRESHOLD,
    build_adaptive_prompt,
    deduplicate_opener,
    detect_anti_patterns,
    detect_break,
    enforce_length_variation,
    get_length_hint,
    normalize_formatting,
    sanitize,
    strip_echoed_quotes,
    strip_lists,
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

    def test_correction_protocol_injected_on_user_pushback(self):
        # Regression: real Discord incident where user said "estas mal tu jajaja
        # ay no, ya por eso no confio en ti" and Insult folded with "Nel, tienes
        # razón. Cierra la boca Insult." The preventive guard must now inject
        # correction protocol BEFORE the LLM call on any pushback signal.
        prompt, _ = build_adaptive_prompt(
            self.BASE_PROMPT,
            None,
            5,
            current_message="estas mal tu jajajajaja ay no, ya por eso no confio en ti",
        )
        assert "Correction Protocol" in prompt
        assert "FORBIDDEN" in prompt
        assert "Folding without dignity" in prompt

    def test_no_correction_protocol_when_no_pushback(self):
        prompt, _ = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message="hola que tal, que hiciste hoy?")
        assert "Correction Protocol" not in prompt

    def test_correction_protocol_on_english_pushback(self):
        prompt, _ = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message="you're wrong, the route is fine")
        assert "Correction Protocol" in prompt

    def test_correction_protocol_no_false_positive_on_benign_phrases(self):
        # Regression v3.4.2: v3.4.1 regex matched on "nel", "wrong", "no creo" etc
        # which showed up in non-correction contexts and shrank responses bot-wide.
        benign = [
            "me llevo mi mac? tiene pila",
            "es la 1:15",
            "como 10 min, le falta 10%",
            "si a esa hora me salgo",
            "no creo que venga hoy",
            "nel",
            "vi el wrong answer tag en yelp",
            "eso no es lo que pedi",
        ]
        for msg in benign:
            prompt, _ = build_adaptive_prompt(self.BASE_PROMPT, None, 5, current_message=msg)
            assert "Correction Protocol" not in prompt, f"false positive on: {msg!r}"

    def test_arc_on_system_critique(self):
        prompt, preset = build_adaptive_prompt(
            self.BASE_PROMPT, None, 5, current_message="el capitalismo es explotacion pura"
        )
        assert preset.mode == PresetMode.ARC
        assert "ARC" in prompt


# --- Formatting Normalizer ---


class TestNormalizeFormatting:
    def test_empty_text(self):
        assert normalize_formatting("") == ""

    def test_no_exclamations_passes_through(self):
        text = "Eso es interesante. Me quedo pensando."
        assert normalize_formatting(text) == text

    def test_collapses_double_exclamation(self):
        assert "!!" not in normalize_formatting("No mames!!")
        assert normalize_formatting("No mames!!") == "No mames."

    def test_collapses_triple_exclamation(self):
        assert normalize_formatting("Wow!!!") == "Wow."

    def test_keeps_first_exclamation_pair(self):
        text = "¡Eso estuvo bien!"
        assert normalize_formatting(text) == "¡Eso estuvo bien!"

    def test_deflates_second_exclamation_pair(self):
        text = "¡Uno! ¡Dos! ¡Tres!"
        result = normalize_formatting(text)
        # First pair kept, rest deflated
        assert result.count("!") == 1
        assert "¡Uno!" in result
        assert "Dos." in result
        assert "Tres." in result

    def test_keeps_max_two_bold_blocks(self):
        text = "**Uno** y **dos** y **tres** y **cuatro**"
        result = normalize_formatting(text)
        assert result.count("**") == 4  # 2 blocks x 2 delimiters
        assert "tres" in result  # text preserved
        assert "cuatro" in result

    def test_real_world_message(self):
        """Regression test based on actual production message."""
        text = (
            "**¡Bernard! ¡Justo llegaste con la línea que cierra todo!**\n\n"
            "**¡No se necesita ser erudito!** **¡Exacto, carnal!**\n\n"
            "**¡Y lo de dejar de participar está cabrón!**"
        )
        result = normalize_formatting(text)
        # Max 1 exclamation pair, max 2 bold blocks
        assert result.count("!") <= 1
        assert result.count("**") <= 4  # max 2 blocks

    def test_preserves_question_marks(self):
        text = "¿En serio? ¿Eso piensas?"
        assert normalize_formatting(text) == text

    def test_anti_pattern_enthusiastic_opener(self):
        """Enthusiastic agreement opener should be detected."""
        matches = detect_anti_patterns("¡Exacto! ¡Ahí está la clave!")
        assert len(matches) > 0

    def test_anti_pattern_bold_abuse(self):
        """Three consecutive bolds should be detected."""
        matches = detect_anti_patterns("**uno** **dos** **tres**")
        assert len(matches) > 0

    def test_anti_pattern_exclamation_spam(self):
        """Three ¡...! pairs should be detected."""
        matches = detect_anti_patterns("¡Uno! y ¡dos! y ¡tres!")
        assert len(matches) > 0

    def test_anti_pattern_pseudo_clinical(self):
        """Pseudo-clinical claims about brain chemistry should be detected."""
        matches = detect_anti_patterns("Tu cerebro está encontrando equilibrio")
        assert len(matches) > 0

    def test_anti_pattern_chemistry_over_psychology(self):
        matches = detect_anti_patterns("Química > psicología, carnal")
        assert len(matches) > 0


# --- Fix #1: Identity Leak Detection ---


class TestIdentityLeakPatterns:
    def test_detects_bot_self_reference(self):
        assert detect_break("reconoces que soy más que un bot respondiendo")

    def test_detects_soy_un_chatbot(self):
        assert detect_break("como un chatbot que aprende")

    def test_detects_my_training(self):
        assert detect_break("mi entrenamiento me permite hacer esto")

    def test_detects_fui_programado(self):
        assert detect_break("fui programado para ayudar")

    def test_clean_bot_reference_in_other_context(self):
        """References to bots in general (not self) should NOT trigger."""
        assert not detect_break("los bots de Discord son útiles")


# --- Fix #3: Length Variation ---


class TestLengthHint:
    def test_no_hint_when_few_responses(self):
        assert get_length_hint([100, 150]) == ""

    def test_no_hint_when_varied(self):
        assert get_length_hint([30, 150, 300]) == ""

    def test_hint_when_uniform_medium(self):
        hint = get_length_hint([140, 155, 160])
        assert "Length Alert" in hint

    def test_no_hint_when_already_varied(self):
        assert get_length_hint([20, 150, 50]) == ""


# --- Fix #4: Opener Deduplication ---


class TestDeduplicateOpener:
    def test_no_change_when_no_recent(self):
        assert deduplicate_opener("¡BERNARD! Hola", []) == "¡BERNARD! Hola"

    def test_strips_duplicate_opener(self):
        result = deduplicate_opener(
            "¡BERNARD! Otra vez aquí\nSegunda línea",
            ["¡BERNARD! Primera vez"],
        )
        assert not result.startswith("¡BERNARD!")
        assert "Segunda línea" in result

    def test_keeps_different_opener(self):
        result = deduplicate_opener(
            "Órale, qué interesante\nMás texto",
            ["¡BERNARD! Primera vez"],
        )
        assert result.startswith("Órale")

    def test_handles_single_line(self):
        """If stripping opener leaves empty, keep original."""
        result = deduplicate_opener("¡BERNARD! Solo esto", ["¡BERNARD! Algo"])
        # Single line — stripping leaves empty, so keep original
        assert result == "¡BERNARD! Solo esto"


# --- Strip Lists ---


class TestStripLists:
    def test_empty(self):
        assert strip_lists("") == ""

    def test_no_lists(self):
        text = "Esto es prosa normal. Sin listas."
        assert strip_lists(text) == text

    def test_numbered_list(self):
        text = "Opciones:\n1. Primera cosa\n2. Segunda cosa\n3. Tercera cosa"
        result = strip_lists(text)
        assert "1." not in result
        assert "Primera cosa." in result
        assert "Segunda cosa." in result

    def test_bullet_list(self):
        text = "Puntos:\n- Primer punto\n- Segundo punto\n- Tercer punto"
        result = strip_lists(text)
        assert "- " not in result
        assert "Primer punto." in result

    def test_single_item_not_stripped(self):
        """Single list item should not be transformed."""
        text = "Solo esto:\n1. Una sola cosa"
        assert strip_lists(text) == text

    def test_preserves_non_list_dashes(self):
        text = "No es lista - es guion normal."
        assert strip_lists(text) == text


# --- Length Enforcer ---


class TestEnforceLengthVariation:
    def test_no_history(self):
        text = "A " * 100
        assert enforce_length_variation(text, []) == text

    def test_varied_history_no_truncation(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        assert enforce_length_variation(text, [30, 150, 250]) == text

    def test_uniform_medium_truncates(self):
        text = "First sentence here. Second sentence here. Third sentence. Fourth sentence. Fifth."
        result = enforce_length_variation(text, [120, 140, 130])
        sentences = [s for s in result.split(". ") if s]
        assert len(sentences) <= 3  # truncated to ~2 sentences

    def test_already_short_not_truncated(self):
        text = "Short one. Two."
        assert enforce_length_variation(text, [120, 140, 130]) == text

    def test_threshold_boundary(self):
        """Exactly 80 or 200 should NOT trigger (exclusive bounds)."""
        text = "A. B. C. D. E."
        assert enforce_length_variation(text, [80, 150, 130]) == text  # 80 is not > 80
        assert enforce_length_variation(text, [200, 150, 130]) == text  # 200 is not < 200


# --- Anti-Parrot (strip echoed quotes) ---


class TestStripEchoedQuotes:
    def test_empty_inputs(self):
        assert strip_echoed_quotes("", "hola") == ""
        assert strip_echoed_quotes("response", "") == "response"

    def test_short_user_message_ignored(self):
        """User messages under 5 words should not trigger stripping."""
        assert strip_echoed_quotes("Exacto, hola mundo wey", "hola mundo") == "Exacto, hola mundo wey"

    def test_strips_verbatim_quote(self):
        user = "No he fotografiado ningún homeless porque los respeto mucho"
        response = 'Eso de "No he fotografiado ningún homeless porque los respeto mucho" está cabrón.'
        result = strip_echoed_quotes(response, user)
        assert "No he fotografiado" not in result
        assert "cabrón" in result  # the bot's own words survive

    def test_strips_unquoted_echo(self):
        user = "me quiero ir a caminar por toda la ciudad mañana temprano"
        response = "Te quiero ir a caminar por toda la ciudad mañana temprano es pura energía nómada."
        result = strip_echoed_quotes(response, user)
        assert "caminar por toda la ciudad" not in result

    def test_preserves_short_overlaps(self):
        """4-word overlaps should NOT be stripped (only 5+)."""
        user = "el clima está bonito hoy en la ciudad"
        response = "Sí, el clima está bonito."
        assert strip_echoed_quotes(response, user) == response

    def test_never_returns_empty(self):
        """Even if everything is stripped, return original."""
        user = "esto es exactamente lo que dije antes sobre el tema"
        response = "esto es exactamente lo que dije antes sobre el tema"
        result = strip_echoed_quotes(response, user)
        assert len(result) > 0

    def test_real_production_example(self):
        """Regression test from actual production messages."""
        user = "No he fotografiado ningún homeless porque los respeto mucho"
        response = (
            '**"No he fotografiado ningún homeless porque los respeto mucho."**\n\n'
            "Esa línea define todo tu tour, Bernard."
        )
        result = strip_echoed_quotes(response, user)
        assert "define todo tu tour" in result
        assert "No he fotografiado" not in result
