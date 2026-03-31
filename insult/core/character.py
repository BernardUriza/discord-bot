"""Character break detection and response sanitization."""

import re

import structlog

log = structlog.get_logger()

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


def detect_break(text: str) -> list[str]:
    """Returns list of matched break patterns found in text."""
    return [p.pattern for p in CHARACTER_BREAK_PATTERNS if p.search(text)]


def sanitize(text: str) -> str:
    """Remove sentences that contain character breaks as a last resort."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = [s for s in sentences if not any(p.search(s) for p in CHARACTER_BREAK_PATTERNS)]
    result = " ".join(clean).strip()
    return result if result else text


def build_adaptive_prompt(base_prompt: str, profile, context_len: int) -> str:
    """Compose system prompt: base persona + user style adaptation + identity reinforcement.

    The base persona is NEVER modified — adaptation is appended as soft guidance.
    """
    prompt = base_prompt

    # Style adaptation — only if we have enough data
    if profile and profile.is_confident:
        adaptations = []

        if profile.detected_language == "en":
            adaptations.append("This user writes in English. Respond in English but keep your personality.")

        if profile.avg_word_count < 10:
            adaptations.append(
                "This user is brief. Keep your responses short and punchy — 1-3 sentences. "
                "Hit hard, hit fast."
            )
        elif profile.avg_word_count > 40:
            adaptations.append(
                "This user writes long messages. You can give more detailed responses with examples, "
                "but stay sharp — don't ramble."
            )

        if profile.formality < 0.25:
            adaptations.append(
                "This user is very casual and uses slang freely. You can go full vulgar — "
                "they can take it."
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

    # Identity reinforcement for long conversations
    if context_len >= IDENTITY_REINFORCE_THRESHOLD:
        prompt += IDENTITY_REINFORCEMENT_SUFFIX

    return prompt
