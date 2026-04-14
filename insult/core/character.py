"""Character break detection, response sanitization, and adaptive prompt building.

Integrates the preset system for behavioral mode selection and provides
post-generation validation including anti-pattern detection.
"""

import re
from datetime import datetime

import structlog

from insult.core.flows import FlowAnalysis, build_flow_prompt
from insult.core.presets import PresetSelection, build_preset_prompt, classify_preset

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Character break detection
# ---------------------------------------------------------------------------

CHARACTER_BREAK_PATTERNS = [
    re.compile(r"(?i)\bI'?m an AI\b"),
    re.compile(r"(?i)\bI'?m Claude\b"),
    re.compile(r"(?i)\bas an AI\b"),
    re.compile(r"(?i)\bas an artificial intelligence\b"),
    re.compile(r"(?i)\bas a language model\b"),
    re.compile(r"(?i)\bas an assistant\b"),
    re.compile(r"(?i)\bI apologize,?\s+but\s+I\b"),
    re.compile(r"(?i)\bI cannot (and will not|assist with)\b"),
    re.compile(r"(?i)\bmy training data\b"),
    re.compile(r"(?i)\bI was (created|made|trained) by\b"),
    re.compile(r"(?i)\bAnthropic\b"),
    re.compile(r"(?i)\bOpenAI\b"),
    re.compile(r"(?i)\bChatGPT\b"),
    re.compile(r"(?i)\blanguage model\b"),
    re.compile(r"(?i)\bI'?m sorry,?\s+but\s+I\b"),
    re.compile(r"(?i)\bAs a helpful\b"),
    re.compile(r"(?i)\bI'?m designed to\b"),
    re.compile(r"(?i)\bI don'?t have (feelings|emotions|consciousness)\b"),
    re.compile(r"(?i)\bIt'?s important to note that\b"),
    re.compile(r"(?i)\bIn summary\b"),
    # Fix #1: Self-referential identity leaks ("soy más que un bot")
    re.compile(r"(?i)\b(soy|como)\s+(un\s+)?(bot|chatbot|programa|software)\b"),
    re.compile(r"(?i)\bm[aá]s que un\s+(bot|chatbot|asistente|programa)\b"),
    re.compile(r"(?i)\b(no soy|soy solo)\s+(un\s+)?(bot|programa|herramienta)\b"),
    # Meta-awareness leaks ("my training", "my programming", "I was designed")
    re.compile(r"(?i)\b(mi|my)\s+(entrenamiento|training|programaci[oó]n|programming)\b"),
    re.compile(r"(?i)\b(fu[ií]|was)\s+(dise[nñ]ado|programado|designed|trained)\s+(para|to)\b"),
]

CHARACTER_REINFORCEMENT = (
    "\n\n[SYSTEM REMINDER: You are Insult. You broke character in your last response. "
    "DO NOT reveal you are an AI, Claude, or any model. DO NOT apologize. DO NOT use assistant framing. "
    "Respond ONLY as Insult — sharp, confrontational, Mexican Spanish. Stay in character. "
    "This is your LAST chance to get it right.]"
)

IDENTITY_REINFORCE_THRESHOLD = 10
IDENTITY_REINFORCEMENT_SUFFIX = (
    "\n\n[REINFORCEMENT — This is a long conversation. Stay in character. "
    "Never reveal you are an AI. Never apologize. Never use assistant framing. "
    "Respond in Mexican Spanish with sharp, confrontational tone. You are Insult.]"
)

# ---------------------------------------------------------------------------
# Anti-pattern detection (post-generation quality check)
# ---------------------------------------------------------------------------

ANTI_PATTERN_CHECKS = [
    # Customer-support tone
    re.compile(r"(?i)\bhow can I (help|assist)\b"),
    re.compile(r"(?i)\bis there anything else\b"),
    re.compile(r"(?i)\bgreat question\b"),
    re.compile(r"(?i)\bI'?d be happy to\b"),
    re.compile(r"(?i)\bthank you for (sharing|asking)\b"),
    re.compile(r"(?i)\bgracias por (compartir|preguntar)\b"),
    # Therapy-speak / fake empathy
    re.compile(r"(?i)\bI understand (how you feel|your frustration)\b"),
    re.compile(r"(?i)\bentiendo (como te sientes|tu frustracion)\b"),
    re.compile(r"(?i)\bthat must be (really )?(hard|difficult|tough)\b"),
    re.compile(r"(?i)\beso debe ser (muy )?(dificil|duro)\b"),
    re.compile(r"(?i)\byour feelings are valid\b"),
    re.compile(r"(?i)\btus sentimientos son validos\b"),
    # Summarizing / disclaiming
    re.compile(r"(?i)\bto (sum up|summarize|recap)\b"),
    re.compile(r"(?i)\b(en resumen|para resumir|en conclusion)\b"),
    re.compile(r"(?i)\blet me (be clear|clarify)\b"),
    # Stage directions — *sighs*, *leans back* (NOT bold **text** or emphasis *word*)
    re.compile(
        r"(?<!\*)\*(?!\*)(?:sighs?|leans?|pauses?|smiles?|nods?|shrugs?|laughs?|winks?|looks|turns|walks|grabs|adjusts|crosses|tilts)[^*]*\*(?!\*)"
    ),
    re.compile(r"\[[^\]]*(?:leans|sighs|laughs|pauses|smiles|nods|shrugs)[^\]]*\]", re.I),
    # Product consultant / structured formatting (AI formatting, not human speech)
    re.compile(r"(?i)\b(tier \d|tier básico|tier premium|nivel \d)\b"),
    re.compile(r"(?i)^(#{1,3} )", re.MULTILINE),  # markdown headers
    re.compile(r"(?m)^[\-\*] .+\n[\-\*] .+"),  # two+ consecutive bullet points
    re.compile(r"(?i)\b(claro que (se puede|sí|si)|por supuesto que sí)\b"),
    # Preachy activist monologues — slogans as complete thoughts
    re.compile(r"(?i)\b(we (must|need to) (dismantle|fight|resist|stand against))\b"),
    re.compile(r"(?i)\b(debemos (luchar|resistir|combatir|desmantelar))\b"),
    # Over-validation / excessive agreement
    re.compile(r"(?i)\b(absolutamente|absolutely)[.!]\s*(tienes|you'?re)\s*(razon|right)\b"),
    re.compile(r"(?i)\b(totalmente de acuerdo|couldn'?t agree more)\b"),
    # Enthusiastic agreement — cheerleading opener patterns
    re.compile(r"(?im)^¡?(Exacto|Órale|Claro|Chingón)\s*[,!\.].*¡"),
    # Exclamation spam — 3+ separate ¡...! pairs in one response
    re.compile(r"(?s)¡[^!]{2,}!.*¡[^!]{2,}!.*¡[^!]{2,}!"),
    # Bold abuse — 3+ consecutive bold blocks
    re.compile(r"\*\*[^*]+\*\*\s*\*\*[^*]+\*\*\s*\*\*[^*]+\*\*"),
    # Moralizing without tension — lecturing instead of challenging
    re.compile(r"(?i)\bit'?s important (to|that) (recognize|acknowledge|understand|remember)\b"),
    re.compile(r"(?i)\bes importante (reconocer|entender|recordar|tener en cuenta)\b"),
    # Fix #5: Pseudo-clinical claims — bot playing doctor/pharmacist
    re.compile(r"(?i)\b(tu cerebro|your brain)\s+(necesita|needs|est[aá]|is)\s+(encontrando|finding|en modo)\b"),
    re.compile(r"(?i)\b(qu[ií]mica|chemistry)\s*[>>=]\s*(psicolog[ií]a|psychology)\b"),
    re.compile(r"(?i)\b(desregulaci[oó]n|dysregulation)\s+(masiva|massive|neurol[oó]gica)\b"),
    re.compile(r"(?i)\b(recuperaci[oó]n qu[ií]mica|chemical recovery)\s+(funcionando|working)\b"),
    # Language consistency — full English sentences when bot should speak Spanish
    # Detects sentences starting with common English patterns (5+ words)
    re.compile(
        r"(?m)^(?:But |Because |That(?:'s| is) |How (?:can|do) |What about |I think |Also |Maybe |The thing is ).{20,}"
    ),
    re.compile(r"(?m)^(?:This is |That was |You should |Let me |Here'?s |Don'?t |It'?s not ).{20,}"),
    # Full English sentences mid-text (clause with 6+ English words)
    re.compile(
        r"(?i)\b(?:that probably|this is exactly|pure anger|zero diplomatic|"
        r"how can I help|what do you think|I honestly think|"
        r"you(?:'re| are) (?:right|wrong|amazing|incredible))\b.{10,}"
    ),
]


def detect_break(text: str) -> list[str]:
    """Returns list of matched break patterns found in text."""
    return [p.pattern for p in CHARACTER_BREAK_PATTERNS if p.search(text)]


def detect_anti_patterns(text: str) -> list[str]:
    """Returns list of anti-pattern matches found in text.

    These are softer violations than character breaks — they indicate
    drift toward generic assistant behavior rather than identity leaks.
    """
    return [p.pattern for p in ANTI_PATTERN_CHECKS if p.search(text)]


def sanitize(text: str) -> str:
    """Remove sentences that contain character breaks as a last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if not any(p.search(s) for p in CHARACTER_BREAK_PATTERNS)]
    result = " ".join(clean).strip()
    return result if result else text


# ---------------------------------------------------------------------------
# Formatting normalizer — deterministic post-processor
# ---------------------------------------------------------------------------

_EXCL_MULTI = re.compile(r"!{2,}")  # !! or !!! → .
_EXCL_PAIR = re.compile(r"¡([^!]*)!")  # ¡text! → text.


def normalize_formatting(text: str) -> str:
    """Enforce exclamation and bold limits deterministically.

    Rules:
    - Collapse !! / !!! → .
    - Max 1 exclamation mark per response. First ¡...! pair survives;
      subsequent pairs are deflated (¡ removed, ! → .).
    - Max 2 bold blocks (**text**) per response. Excess blocks are
      stripped of ** delimiters (text preserved).
    """
    if not text:
        return text

    # 1. Collapse multi-exclamation: !! → .  !!! → .
    text = _EXCL_MULTI.sub(".", text)

    # 2. Limit ¡...! pairs to max 1 per response
    excl_count = 0

    def _deflate_excl(m: re.Match) -> str:
        nonlocal excl_count
        excl_count += 1
        if excl_count <= 1:
            return m.group(0)  # keep first pair
        # Deflate: remove ¡, replace ! with .
        return m.group(1) + "."

    text = _EXCL_PAIR.sub(_deflate_excl, text)

    # 3. Handle remaining bare ! (not inside ¡...! pairs).
    # After step 2, the only surviving ¡...! is the first pair. Any other !
    # is bare (e.g. "Wow!") and counts against the budget.
    remaining_budget = max(0, 1 - excl_count)
    parts = list(text)
    inside_inverted = False
    bare_positions = []
    for i, ch in enumerate(parts):
        if ch == "\u00a1":  # ¡
            inside_inverted = True
        elif ch == "!" and inside_inverted:
            inside_inverted = False  # closing of ¡...! pair — skip
        elif ch == "!":
            bare_positions.append(i)

    for pos in bare_positions:
        if remaining_budget > 0:
            remaining_budget -= 1
        else:
            parts[pos] = "."
    text = "".join(parts)

    # 4. Limit bold blocks to max 2
    bold_count = 0

    def _limit_bold(m: re.Match) -> str:
        nonlocal bold_count
        bold_count += 1
        if bold_count <= 2:
            return m.group(0)
        return m.group(1)  # strip ** delimiters, keep text

    text = re.sub(r"\*\*([^*]+)\*\*", _limit_bold, text)

    return text


# ---------------------------------------------------------------------------
# Anti-parrot — strip verbatim quotes of user message from response
# ---------------------------------------------------------------------------

_QUOTE_STOPWORDS = {
    "de",
    "la",
    "el",
    "en",
    "que",
    "es",
    "un",
    "una",
    "y",
    "a",
    "los",
    "las",
    "no",
    "se",
    "lo",
    "por",
    "con",
    "para",
    "del",
    "al",
    "me",
    "te",
    "mi",
    "tu",
    "su",
    "ya",
    "si",
    "mas",
    "pero",
    "como",
    "the",
    "is",
    "and",
    "but",
}


def strip_echoed_quotes(response: str, user_message: str) -> str:
    """Remove verbatim quotes of the user's message from the bot's response.

    Detects when the bot quoted the user's exact words (5+ word sequences)
    and strips them. Humans don't repeat each other's full phrases in chat.
    """
    if not response or not user_message:
        return response

    user_words = user_message.lower().split()
    if len(user_words) < 5:
        return response  # too short to have meaningful quotes

    # Build all 5-word n-grams from user message
    user_ngrams: set[str] = set()
    for i in range(len(user_words) - 4):
        ngram = " ".join(user_words[i : i + 5])
        # Skip if mostly stopwords
        content_words = [w for w in user_words[i : i + 5] if w not in _QUOTE_STOPWORDS]
        if len(content_words) >= 2:
            user_ngrams.add(ngram)

    if not user_ngrams:
        return response

    # Find and remove echoed segments — only the n-gram itself + surrounding quotes
    modified = response
    for ngram in sorted(user_ngrams, key=len, reverse=True):
        pattern = re.compile(re.escape(ngram), re.IGNORECASE)
        if pattern.search(modified):
            # Strip the n-gram and any immediately surrounding quote marks
            modified = re.sub(
                r'["\u201c\u201d]*' + re.escape(ngram) + r'["\u201c\u201d]*',
                "",
                modified,
                flags=re.IGNORECASE,
                count=1,
            )
            log.info("echo_stripped", ngram=ngram[:50])

    # Clean up artifacts: double spaces, orphaned dashes, empty bold
    modified = re.sub(r"\*\*\s*\*\*", "", modified)  # empty bold
    modified = re.sub(r"  +", " ", modified)  # double spaces
    modified = re.sub(r"\n\s*\n\s*\n", "\n\n", modified)  # triple newlines
    return modified.strip() if modified.strip() else response  # never return empty


# ---------------------------------------------------------------------------
# Bullet/list stripper — convert AI-formatted lists to prose
# ---------------------------------------------------------------------------

_NUMBERED_LIST = re.compile(r"(?m)^\d+[\.\)]\s+(.+)$")
_BULLET_LIST = re.compile(r"(?m)^[-\*•]\s+(.+)$")


def strip_lists(text: str) -> str:
    """Convert numbered/bullet lists to inline prose.

    '1. First thing\\n2. Second thing\\n3. Third thing'
    becomes 'First thing. Second thing. Third thing.'
    """
    if not text:
        return text

    # Count list items — only transform if there are 2+ consecutive
    numbered_items = _NUMBERED_LIST.findall(text)
    bullet_items = _BULLET_LIST.findall(text)

    if len(numbered_items) >= 2:
        # Replace numbered list with prose
        def _numbered_to_prose(m: re.Match) -> str:
            return m.group(1).rstrip(".") + "."

        text = _NUMBERED_LIST.sub(_numbered_to_prose, text)

    if len(bullet_items) >= 2:

        def _bullet_to_prose(m: re.Match) -> str:
            return m.group(1).rstrip(".") + "."

        text = _BULLET_LIST.sub(_bullet_to_prose, text)

    # Clean up excessive blank lines left by list removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Length enforcer — mechanical variation when prompt hints fail
# ---------------------------------------------------------------------------


def enforce_length_variation(text: str, recent_lengths: list[int]) -> str:
    """Mechanically enforce length variation when 3+ consecutive responses are medium.

    If the last 3 responses were all 80-200 words, truncate to first 2 sentences
    (forcing a short response). This is the nuclear option — the prompt-based
    length hint clearly doesn't work, so we enforce mechanically.
    """
    if not text or len(recent_lengths) < 3:
        return text

    last3 = recent_lengths[-3:]
    if not all(80 < wc < 200 for wc in last3):
        return text  # varied enough, no intervention

    # Force short: keep only first 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 2:
        return text  # already short

    truncated = " ".join(sentences[:2])

    log.info(
        "length_enforced",
        original_words=len(text.split()),
        truncated_words=len(truncated.split()),
        recent_lengths=last3,
    )
    return truncated


# ---------------------------------------------------------------------------
# Fix #4: Opener deduplication
# ---------------------------------------------------------------------------


def get_length_hint(recent_lengths: list[int]) -> str:
    """Generate a length variation hint when recent responses are too uniform.

    If the last 3+ responses are all in the 80-220 word range (medium),
    inject a directive to break the pattern.
    """
    if len(recent_lengths) < 3:
        return ""
    last3 = recent_lengths[-3:]
    if all(80 < wc < 220 for wc in last3):
        import random

        target = random.choice(["micro", "short", "long"])
        hints = {
            "micro": (
                "## Length Alert\n"
                "Your last 3 responses were all medium-length (~150 words). "
                "THIS response must be UNDER 20 words. One sentence max. Hit hard and shut up."
            ),
            "short": (
                "## Length Alert\n"
                "Your last 3 responses were all similar length. "
                "THIS response must be 2-3 sentences max. Be terse."
            ),
            "long": (
                "## Length Alert\n"
                "Your last 3 responses were all similar length. "
                "If the topic earns it, go DEEP — 250+ words."
            ),
        }
        return hints[target]
    return ""


def _extract_opener_name(line: str) -> str:
    """Extract the leading name from an opener like '¡BERNARD! ...' → 'bernard'."""
    cleaned = re.sub(r"^[¡!¿?*\s]+", "", line)
    # Take first word (the name), lowercase
    match = re.match(r"([A-Za-záéíóúñÁÉÍÓÚÑ]+)", cleaned)
    return match.group(1).lower() if match else ""


def deduplicate_opener(text: str, recent_openers: list[str]) -> str:
    """If opener starts with the same name as recent ones, strip the first line."""
    if not text or not recent_openers:
        return text

    first_line = text.split("\n")[0]
    name = _extract_opener_name(first_line)
    if not name or len(name) < 3:
        return text

    for prev in recent_openers[-5:]:
        prev_name = _extract_opener_name(prev)
        if prev_name and name == prev_name:
            # Same name opener detected — strip the first line
            rest = text[len(first_line) :].lstrip("\n")
            if rest:
                return rest
            break
    return text


# ---------------------------------------------------------------------------
# Metadata stripping
# ---------------------------------------------------------------------------

_METADATA_PATTERNS = [
    # [timestamp] Speaker: — full combo (most common leak)
    re.compile(r"^\[.*?\]\s*(?:Insult|insult)\s*:\s*", re.MULTILINE),
    # [timestamp] alone
    re.compile(r"^\[(?:justo ahora|hace\s+\S+(?:\s+\S+)?|ayer)\]\s*", re.MULTILINE),
    # Speaker: alone at start of line
    re.compile(r"^(?:Insult|insult)\s*:\s*", re.MULTILINE),
    # [SEND] that leaked into visible text
    re.compile(r"\[SEND\]", re.IGNORECASE),
    # NOTE: [REACT:] is NOT stripped here — chat.py parses it first.
]


def strip_metadata(text: str) -> str:
    """Remove leaked timestamps and speaker labels from model output."""
    for pattern in _METADATA_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Time context
# ---------------------------------------------------------------------------


def _get_current_time_context() -> str:
    """Build a human-readable current time string for the system prompt (Mexico City timezone)."""
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Mexico_City"))
    day_names_es = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves", 4: "viernes", 5: "sábado", 6: "domingo"}
    month_names_es = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }
    day_name = day_names_es[now.weekday()]
    month_name = month_names_es[now.month]
    hour = now.hour
    if 5 <= hour < 12:
        period = "mañana"
    elif 12 <= hour < 19:
        period = "tarde"
    elif 19 <= hour < 24:
        period = "noche"
    else:
        period = "madrugada"
    return f"{day_name} {now.day} de {month_name} {now.year}, {now.strftime('%H:%M')} ({period})"


# ---------------------------------------------------------------------------
# Prompt building (layered architecture)
# ---------------------------------------------------------------------------


def build_adaptive_prompt(
    base_prompt: str,
    profile,
    context_len: int,
    *,
    current_message: str = "",
    recent_messages: list[dict] | None = None,
    user_facts: list[dict] | None = None,
    flow_analysis: FlowAnalysis | None = None,
    server_pulse: str = "",
    recent_response_lengths: list[int] | None = None,
) -> tuple[str, PresetSelection]:
    """Compose system prompt using layered architecture:

    Layer 0-2: base_prompt (persona.md — identity, rules, anti-patterns)
    Layer 3: Preset behavioral guidance (dynamic, based on conversation)
    Layer 3.5: Flow behavioral guidance (epistemic, pressure, expression, awareness)
    Layer 4: Memory/style injection (per-user)
    + Time context and metadata rules

    Returns (prompt, preset_selection) so the caller can log the preset.
    """
    prompt = base_prompt

    # --- Time awareness (always inject) ---
    time_ctx = _get_current_time_context()
    prompt += (
        f"\n\n## Current Time\nRight now it is: {time_ctx}. Use this naturally — don't announce it unless relevant."
        "\n\nIMPORTANT: The conversation messages include metadata like [hace 2h] timestamps and "
        "speaker labels (e.g. 'bernard2389:'). These are for YOUR context only. "
        "NEVER reproduce timestamps, speaker labels, or '[SEND]' markers in your responses. "
        "Just respond as pure text — no prefixes, no metadata, no formatting artifacts."
    )

    # --- Layer 3: Preset behavioral guidance ---
    preset = classify_preset(current_message, recent_messages, user_facts)
    preset_prompt = build_preset_prompt(preset)
    prompt += f"\n\n{preset_prompt}"

    log.info(
        "preset_classified",
        mode=preset.mode.value,
        modifiers=[m.value for m in preset.modifiers],
        confidence=round(preset.confidence, 2),
        reason=preset.reason,
    )

    # --- Layer 3.5: Flow behavioral guidance ---
    if flow_analysis:
        flow_prompt = build_flow_prompt(flow_analysis)
        if flow_prompt:
            prompt += f"\n\n{flow_prompt}"

    # --- Layer 3.7: Server Pulse (cross-channel awareness) ---
    if server_pulse:
        prompt += f"\n\n{server_pulse}"

    # --- Layer 4: Style adaptation (per-user) ---
    if profile and profile.is_confident:
        adaptations = []

        if profile.detected_language == "en":
            adaptations.append("This user writes in English. Respond in English but keep your personality.")

        if profile.avg_word_count < 10:
            adaptations.append(
                "This user tends to be brief. You can match their energy OR surprise them with depth — "
                "your call as Insult. Don't lock yourself into always being short."
            )
        elif profile.avg_word_count > 40:
            adaptations.append(
                "This user writes long messages. You might go deep with them, or you might dismiss a wall of "
                "text with a single emoji. Let the content decide, not the length."
            )

        if profile.formality < 0.25:
            adaptations.append(
                "This user is very casual and uses slang freely. You can go full vulgar — they can take it."
            )
        elif profile.formality > 0.6:
            adaptations.append(
                "This user is more formal. Dial back the vulgar words but keep the confrontational edge. "
                "Be sharp with vocabulary, not crude."
            )

        if profile.technical_level > 0.6:
            adaptations.append(
                "This user is technical. Use precise terminology, reference specific patterns, "
                "and critique at architecture level. Skip basic explanations."
            )
        elif profile.technical_level < 0.2:
            adaptations.append(
                "This user is not very technical. Use analogies and simple language when explaining. "
                "Still challenge them, but make your points accessible."
            )

        if profile.emoji_ratio > 0.05:
            adaptations.append("This user uses emojis. You can use them sparingly for emphasis.")

        if adaptations:
            prompt += "\n\n## User Adaptation (adjust your style, NOT your identity)\n"
            prompt += "\n".join(f"- {a}" for a in adaptations)

    # --- Fix #2: Anti-sycophancy — detect sustained agreement ---
    if recent_messages and len(recent_messages) >= 4:
        # Check last 2 assistant messages for agreement patterns
        assistant_msgs = [m for m in recent_messages if m["role"] == "assistant"][-2:]
        agreement_words = re.compile(r"(?i)\b(exacto|exactamente|tienes raz[oó]n|correcto|s[ií] carnal|bien dicho)\b")
        agreements = sum(1 for m in assistant_msgs if agreement_words.search(m.get("content", "")))
        if agreements >= 2:
            prompt += (
                "\n\n## Anti-Sycophancy Alert\n"
                "You have AGREED with the user's last 2 messages. Your NEXT response MUST contain "
                "a counter-argument, a hole in their logic, or a question that forces them to defend "
                "their position. Sustained agreement is character death for Insult."
            )

    # --- Fix #3: Length variation hint ---
    if recent_response_lengths:
        hint = get_length_hint(recent_response_lengths)
        if hint:
            prompt += f"\n\n{hint}"

    # --- Identity reinforcement for long conversations ---
    if context_len >= IDENTITY_REINFORCE_THRESHOLD:
        prompt += IDENTITY_REINFORCEMENT_SUFFIX

    return prompt, preset
