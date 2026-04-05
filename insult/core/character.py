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
    re.compile(r"(?<!\*)\*(?!\*)(?:sighs?|leans?|pauses?|smiles?|nods?|shrugs?|laughs?|winks?|looks|turns|walks|grabs|adjusts|crosses|tilts)[^*]*\*(?!\*)"),
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
    # Moralizing without tension — lecturing instead of challenging
    re.compile(r"(?i)\bit'?s important (to|that) (recognize|acknowledge|understand|remember)\b"),
    re.compile(r"(?i)\bes importante (reconocer|entender|recordar|tener en cuenta)\b"),
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

    # --- Identity reinforcement for long conversations ---
    if context_len >= IDENTITY_REINFORCE_THRESHOLD:
        prompt += IDENTITY_REINFORCEMENT_SUFFIX

    return prompt, preset
