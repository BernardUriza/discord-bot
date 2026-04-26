"""Detection of identity-leak and assistant-drift patterns + reinforcement constants.

Three pattern families:

- ``CHARACTER_BREAK_PATTERNS`` — hard identity leaks (the model claiming to be
  an AI / Claude / a language model). Triggers a retry with a reinforced prompt
  and, on retry failure, sanitization of the offending sentences.
- ``ANTI_PATTERN_CHECKS`` — soft drift toward generic-assistant tone
  (customer-support framing, therapy-speak, summarizing). Logged for monitoring;
  does not block the response.
- ``CLARIFICATION_DUMP_PATTERNS`` — the bot punting the task back at the user
  ("dime qué busco", "a qué te refieres") despite having context. Triggers a
  retry with ``CONTEXT_REINFORCEMENT`` appended.

Reinforcement strings + identity thresholds live here too because they belong
to the same failure-mode-and-remedy story; ``llm.chat`` and the prompt
builder both consume them.
"""

from __future__ import annotations

import re

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

# Reinforcement appended when the model produces a clarification-dump response
# ("dime qué busco", "a qué te refieres", "repite") despite having conversation
# context. Fires via `detect_clarification_dump` + retry path in llm.chat().
# Kept distinct from CHARACTER_REINFORCEMENT because the failure mode is
# different — the bot is not out of character, it is being lazy and tossing
# the task back at the user instead of reading the scrollback.
CONTEXT_REINFORCEMENT = (
    "\n\n[SYSTEM REMINDER: Your previous response asked the user to clarify, "
    "repeat, or specify something they almost certainly already told you. "
    "Scan the last 20 messages of conversation context above — the answer is "
    "there. Do NOT hand back a clarifying question. Pick the most likely "
    "interpretation from the context and answer with a declarative statement. "
    "This is your LAST chance to use the context you already have.]"
)

IDENTITY_REINFORCE_THRESHOLD = 10
IDENTITY_REINFORCEMENT_SUFFIX = (
    "\n\n[REINFORCEMENT — This is a long conversation. Stay in character. "
    "Never reveal you are an AI. Never apologize. Never use assistant framing. "
    "Respond in Mexican Spanish with sharp, confrontational tone. You are Insult.]"
)

# Marker separating cacheable (stable) prompt content from dynamic content.
# Everything BEFORE this marker is marked cache_control=ephemeral so Anthropic caches it.
# Everything AFTER is dynamic (time, preset, flows, facts) and changes per request.
# llm._send detects this marker to build a two-block system prompt.
CACHE_BOUNDARY = "\n<<<CACHE_BOUNDARY>>>\n"


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


# These patterns fire when the bot responds by dumping the task back at the
# user — asking them to repeat, clarify, or specify something the conversation
# context almost certainly already answered. The classic 2026-04-23 regression:
# user says "Es solo una búsqueda sencilla. Hazla." and the bot replies
# "Dime qué busco." even though the previous 6 messages had the topic.
#
# Narrow on purpose: we only match patterns where the bot is explicitly
# punting. Legitimate probing questions ("¿Por qué crees que X?", "¿Qué te
# hace pensar eso?") are DEFAULT_ABRASIVE-compliant and MUST NOT match.
#
# Used by llm.chat() when len(messages) > 3 to trigger a retry with
# CONTEXT_REINFORCEMENT appended. See persona.md § "Context-First Rule".
CLARIFICATION_DUMP_PATTERNS = [
    # "dime qué [busco/buscar/hago/hacer/quieres/necesitas]" — dumping work back
    re.compile(r"(?i)\bdime\s+(?:qu[eé]|cu[aá]l)\s+(?:busco|buscar|hago|hacer|quieres|necesitas)\b"),
    # "qué quieres que [busque/haga/diga/responda/pregunte/encuentre]" — deflection
    re.compile(r"(?i)\bqu[eé]\s+quieres\s+que\s+(?:busque|haga|diga|responda|pregunte|encuentre)\b"),
    # "a qué te refieres" — forcing user to expand
    re.compile(r"(?i)\ba\s+qu[eé]\s+te\s+refieres\b"),
    # "repite" as standalone imperative at end of a line / sentence
    re.compile(r"(?im)(?<!\w)repite(?:\s+(?:la\s+pregunta|lo\s+que|por\s+favor|eso))?[.!?]?\s*$"),
    # "específicame" / "sé (más) específico" — dumping specificity back
    re.compile(r"(?i)\b(?:s[eé]\s+(?:m[aá]s\s+)?espec[íi]fico|espec[íi]fica(?:me)?)\b"),
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


def detect_clarification_dump(text: str) -> list[str]:
    """Returns list of matched clarification-dump patterns found in text.

    An empty list means the response did NOT deflect the task back to the user.
    A non-empty list means at least one deflection pattern matched.
    """
    return [p.pattern for p in CLARIFICATION_DUMP_PATTERNS if p.search(text)]


def sanitize(text: str) -> str:
    """Remove sentences that contain character breaks as a last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if not any(p.search(s) for p in CHARACTER_BREAK_PATTERNS)]
    result = " ".join(clean).strip()
    return result if result else text
