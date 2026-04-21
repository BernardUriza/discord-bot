"""Language Cure — post-generation normalization of mixed-language output.

Uses a fast, cheap LLM (Haiku) to translate English fragments in
predominantly Spanish text back to Mexican Spanish, preserving brands,
acronyms, proper nouns, tech terms, and Discord/bot syntax.

This is step 7c in the post-generation pipeline, after character guard
and before delivery.
"""

import structlog

log = structlog.get_logger()

LANGUAGE_CURE_SYSTEM = """\
You are a Spanish language normalizer for a Mexican chatbot.

TASK
The input is chatbot output that mixes Spanish and English.
Translate ONLY the English words and phrases to casual Mexican Spanish.
Return ONLY the corrected text — no tags, no prefix, no commentary, no wrappers.

RULES
- Translate English words and phrases to casual Mexican Spanish
- Keep the EXACT same tone, punctuation, capitalization style, and structure
- DO NOT TRANSLATE: brand names (Discord, iPhone, Netflix, Cyberpunk, Spotify), \
acronyms (API, URL, LLM, IA, GTA, CDMX, DM, OP), proper nouns (names of people, \
places, games, movies, bands), tech terms (build, deploy, bug, commit, netrunner, \
cyberware, cyberpsycho), Discord syntax (<@123456>, [REACT:emoji], [SEND]), \
emojis, and URLs
- Words commonly used as Mexican slang stay unchanged: "cool", "bro", "wey", \
"random", "cringe", "mid", "based", "flow", "vibe"
- If the text is already fully in Spanish, return it UNCHANGED
- NEVER add, remove, or rephrase content — only translate the language
- Keep the same line breaks and spacing
- NEVER wrap your answer in <output>, <input>, <response>, ```, or any other \
delimiter. Just write the plain text.

EXAMPLES (the arrow "→" separates input from expected output; do NOT produce it)

But Bernard, this video is INTENSE. Pure anger, zero diplomatic approach.
→ Pero Bernard, este video es INTENSO. Pura rabia, cero enfoque diplomático.

That probably resonated with your experience de being called "aggressive vegan."
→ Eso probablemente resonó con tu experiencia de que te digan "vegano agresivo."

¿En serio no sabes qué son las Corporate Wars en Cyberpunk?
→ ¿En serio no sabes qué son las Corporate Wars en Cyberpunk?

Eso es un classic pattern de evasión, honestly I think you should reconsider [REACT:💀]
→ Eso es un patrón clásico de evasión, honestamente creo que deberías reconsiderar [REACT:💀]

Alex, take your time con la energía. Yesterday was heavy processing - today can be gentle movement.
→ Alex, tómate tu tiempo con la energía. Ayer fue procesamiento pesado - hoy puede ser movimiento suave."""


async def language_cure(
    client,
    model: str,
    text: str,
) -> str:
    """Normalize mixed-language chatbot output to Mexican Spanish.

    Uses a fast/cheap model (Haiku) with no conversation context —
    pure text transformation. Returns original text on any failure.
    """
    if not text or len(text) < 10:
        return text

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=min(len(text) * 3, 4096),  # ~2x input, capped
            system=LANGUAGE_CURE_SYSTEM,
            # User content is the raw text — no wrapper tags. Previous
            # implementation used "<input>{text}</input>" which, combined with
            # XML few-shot, taught Haiku to answer with <output>...</output>
            # wrappers. Even with a strip pass, partial/stray tags leaked.
            messages=[{"role": "user", "content": text}],
        )
        cured = response.content[0].text.strip()

        # Safety: if Haiku returned something wildly different in length, keep original
        if not cured or len(cured) < len(text) * 0.5 or len(cured) > len(text) * 2.5:
            log.warning("language_cure_length_mismatch", original=len(text), cured=len(cured))
            return text

        # Strip stray wrappers Haiku sometimes adds despite the instruction.
        # Handles <output>X</output>, <response>X</response>, and an initial
        # "→ " left over from the few-shot arrow style.
        wrapper_pairs = [
            ("<output>", "</output>"),
            ("<response>", "</response>"),
            ("<input>", "</input>"),
        ]
        for open_tag, close_tag in wrapper_pairs:
            if cured.startswith(open_tag) and cured.endswith(close_tag):
                cured = cured[len(open_tag) : -len(close_tag)].strip()
                break
        if cured.startswith("→ ") or cured.startswith("→"):
            cured = cured.lstrip("→").lstrip()

        if cured != text:
            log.info("language_cure_applied", original_len=len(text), cured_len=len(cured))
        else:
            log.debug("language_cure_no_change")

        return cured

    except Exception:
        log.exception("language_cure_failed")
        return text  # Fail-safe: return original on any error
