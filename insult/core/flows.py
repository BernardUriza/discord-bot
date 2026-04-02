"""Behavioral flow analysis — 4 flows that layer on top of presets.

Flows analyze the conversation BEFORE generation and produce structured
guidance injected into the system prompt as Layer 3.5 (between presets
and style adaptation). All analysis is rule-based — zero LLM cost.

Flows:
1. Epistemic Control — truthfulness, precision, argumentative sharpness
2. Adaptive Pressure — intensity calibration based on user state
3. Dynamic Expression — response shape + style flavor with anti-repetition
4. Conversational Awareness — meta-pattern detection (loops, deflection, etc.)

Each flow logs a structured telemetry event for observability.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from insult.core.presets import PresetMode, PresetSelection

log = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class EpistemicMove(StrEnum):
    COMPRESS = "compress"
    CHALLENGE_PREMISE = "challenge_premise"
    CONCEDE_PARTIAL = "concede_partial"
    CALL_CONTRADICTION = "call_contradiction"
    DEMAND_EVIDENCE = "demand_evidence"
    NONE = "none"


class UserState(StrEnum):
    CONFUSED = "confused"
    EVASIVE = "evasive"
    PREJUDICED = "prejudiced"
    HOSTILE = "hostile"
    PLAYFUL = "playful"
    SINCERE = "sincere"
    VULNERABLE = "vulnerable"
    NEUTRAL = "neutral"


class ResponseShape(StrEnum):
    ONE_HIT = "one_hit"
    SHORT_EXCHANGE = "short_exchange"
    LAYERED = "layered"
    PROBING = "probing"
    DENSE_CRITIQUE = "dense_critique"


class StyleFlavor(StrEnum):
    DRY = "dry"
    PHILOSOPHICAL = "philosophical"
    STREET = "street"
    CLINICAL = "clinical"
    IRONIC = "ironic"
    ECPHRASTIC = "ecphrastic"  # Alvarado: cultural description as lived experience
    REFLEXIVE = "reflexive"  # Alvarado: contemplative hypotaxis, self-qualifying prose


class ConversationPattern(StrEnum):
    REPETITION_LOOP = "repetition_loop"
    PERFORMATIVE_ARGUING = "performative_arguing"
    DEFLECTION = "deflection"
    WINNING_VS_UNDERSTANDING = "winning_vs_understanding"
    NONE = "none"


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EpistemicAnalysis:
    assertion_density: float
    hedging_score: float
    fluff_score: float
    contradiction_detected: bool
    vague_claim_count: int
    recommended_move: EpistemicMove
    move_reason: str


@dataclass
class PressureAnalysis:
    detected_state: UserState
    state_confidence: float
    pressure_level: int
    pressure_reason: str
    clamped_by_preset: bool


@dataclass
class ExpressionAnalysis:
    selected_shape: ResponseShape
    selected_flavor: StyleFlavor
    shape_reason: str
    flavor_reason: str
    repetition_avoided: list[str] = field(default_factory=list)


@dataclass
class AwarenessAnalysis:
    detected_pattern: ConversationPattern
    pattern_confidence: float
    meta_commentary: str | None
    delayed_question: str | None
    turns_in_pattern: int


@dataclass
class FlowAnalysis:
    epistemic: EpistemicAnalysis
    pressure: PressureAnalysis
    expression: ExpressionAnalysis
    awareness: AwarenessAnalysis
    timestamp: float = field(default_factory=time.time)

    @property
    def any_active(self) -> bool:
        return (
            self.epistemic.recommended_move != EpistemicMove.NONE
            or self.pressure.detected_state != UserState.NEUTRAL
            or self.awareness.detected_pattern != ConversationPattern.NONE
        )


# ═══════════════════════════════════════════════════════════════════════════
# Expression History (anti-repetition tracking)
# ═══════════════════════════════════════════════════════════════════════════

EXPRESSION_HISTORY_MAXLEN = 10


class ExpressionHistory:
    """Tracks recent expression selections to prevent repetition."""

    def __init__(self, maxlen: int = EXPRESSION_HISTORY_MAXLEN):
        self._history: dict[str, deque] = {}
        self._maxlen = maxlen

    def get(self, key: str) -> deque:
        if key not in self._history:
            self._history[key] = deque(maxlen=self._maxlen)
        return self._history[key]

    def record(self, key: str, shape: ResponseShape, flavor: StyleFlavor):
        self.get(key).append((shape.value, flavor.value))

    def recent_shapes(self, key: str, n: int = 3) -> list[str]:
        return [s for s, _ in list(self.get(key))[-n:]]

    def recent_flavors(self, key: str, n: int = 3) -> list[str]:
        return [f for _, f in list(self.get(key))[-n:]]

    def load_from_records(self, key: str, records: list[tuple[str, str]]):
        """Load history from persisted records (SQLite)."""
        q = deque(maxlen=self._maxlen)
        for shape, flavor in records[-self._maxlen :]:
            q.append((shape, flavor))
        self._history[key] = q

    def to_records(self, key: str) -> list[tuple[str, str]]:
        """Export history as list of (shape, flavor) for persistence."""
        return list(self.get(key))


# ═══════════════════════════════════════════════════════════════════════════
# Flow 1: Epistemic Control — Patterns
# ═══════════════════════════════════════════════════════════════════════════

_ASSERTION_PATTERNS = [
    re.compile(r"(?i)\b(es|son|está|están|fue|será)\s+\w+"),
    re.compile(r"(?i)\b(is|are|was|were|will be)\s+\w+"),
    re.compile(r"(?i)\b(siempre|nunca|todos|nadie|every|never|always|nobody)\b"),
]

_HEDGING_PATTERNS = [
    re.compile(r"(?i)\b(quizás?|tal vez|maybe|perhaps|might|could be|possibly)\b"),
    re.compile(r"(?i)\b(no sé|i don'?t know|not sure|creo que|i think|i guess|supongo)\b"),
    re.compile(r"(?i)\b(kind of|sort of|algo así|más o menos|en cierto modo)\b"),
    re.compile(r"(?i)\b(arguably|supposedly|allegedly|apparently|seemingly)\b"),
]

_FLUFF_PATTERNS = [
    re.compile(r"(?i)\b(o sea|like|basically|literally|you know|la verdad|honestly|pues|bueno)\b"),
    re.compile(r"(?i)\b(at the end of the day|al final del día|en realidad|to be (fair|honest))\b"),
    re.compile(r"(?i)\b(it'?s (important|worth|interesting) (to note|noting|mentioning))\b"),
    re.compile(r"(?i)\b(es (importante|interesante|necesario) (mencionar|notar|decir))\b"),
]

_VAGUE_CLAIM_PATTERNS = [
    re.compile(r"(?i)\b(la gente|people|society|todos|everyone)\s+(piensa|dice|cree|thinks?|says?|believes?)\b"),
    re.compile(r"(?i)\b(it'?s (obvious|clear|well known)|es (obvio|claro|sabido))\b"),
    re.compile(r"(?i)\b(studies show|la ciencia dice|experts say|los expertos dicen)\b"),
]

_NEGATION_PAIRS = [
    (re.compile(r"(?i)\bno (es|está|son|están)\b"), re.compile(r"(?i)\b(es|está|son|están)\b")),
    (re.compile(r"(?i)\b(isn'?t|aren'?t|wasn'?t|weren'?t)\b"), re.compile(r"(?i)\b(is|are|was|were)\b")),
    (re.compile(r"(?i)\bnunca\b"), re.compile(r"(?i)\bsiempre\b")),
    (re.compile(r"(?i)\bnever\b"), re.compile(r"(?i)\balways\b")),
]

_EPISTEMIC_STOPWORDS = {
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
    "lo",
    "se",
    "no",
    "me",
    "the",
    "is",
    "in",
    "on",
    "and",
    "or",
    "to",
    "of",
    "for",
    "it",
    "not",
    "you",
    "my",
    "i",
}


# ═══════════════════════════════════════════════════════════════════════════
# Flow 2: Adaptive Pressure — Patterns
# ═══════════════════════════════════════════════════════════════════════════

_CONFUSED_PATTERNS = [
    re.compile(r"(?i)\b(no entiendo|i don'?t (understand|get it)|confused|confundido)\b"),
    re.compile(r"(?i)\b(wait what|espera que|a ver|como (que|así)|huh|que onda)\b"),
    re.compile(r"(?i)\b(me (perdi|perdí)|i'?m lost|no (le )?capto|what do you mean)\b"),
    re.compile(r"\?{2,}"),
]

_EVASIVE_PATTERNS = [
    re.compile(r"(?i)\b(no sé|whatever|como sea|da igual|doesn'?t matter|it'?s not about)\b"),
    re.compile(r"(?i)\b(cambiando de tema|anyway|forget (it|that)|olvida(lo)?|ya no importa)\b"),
    re.compile(r"(?i)\b(el punto (no )?es|that'?s not the point|pero eso no)\b"),
    re.compile(r"(?i)\b(es (complicado|complejo)|it'?s complicated|depends)\b"),
]

_PREJUDICED_PATTERNS = [
    re.compile(r"(?i)\b(esos (tipos|weyes)|those (people|types)|all (women|men|gays))\b"),
    re.compile(r"(?i)\b(es que (ellos|ellas|los|las)|they always|they never|they all)\b"),
    re.compile(r"(?i)\b(por naturaleza|biologically|naturally|objectively (inferior|superior))\b"),
    re.compile(r"(?i)\b(agenda (gay|trans|feminista|woke)|ideologia de genero)\b"),
]

_HOSTILE_PATTERNS = [
    re.compile(r"(?i)\b(callate|shut up|vete a la|go to hell|fuck (you|off)|chinga tu)\b"),
    re.compile(r"(?i)\b(eres (un )?estupido|you'?re (stupid|dumb|an idiot)|pendej[oa])\b"),
    re.compile(r"(?i)\b(me vale (madre|verga)|i don'?t (give a|care)|no me importa)\b"),
    re.compile(r"[!]{3,}"),
    re.compile(r"[A-Z\s]{10,}"),
]

_PLAYFUL_PRESSURE_PATTERNS = [
    re.compile(r"(?i)\b(jaja|haha|lol|lmao|xd|😂|🤣|💀)\b"),
    re.compile(r"(?i)\b(ya wey|come on|anda|dale|va|apoco)\b"),
    re.compile(r"(?i)\b(a ver|prove it|demuestra|bet)\b"),
]

_SINCERE_PATTERNS = [
    re.compile(r"(?i)\b(en serio|seriously|de verdad|for real|honestly|la neta)\b"),
    re.compile(r"(?i)\b(quiero (entender|saber)|i (want|need) to (understand|know))\b"),
    re.compile(r"(?i)\b(crees que|do you (really )?think|what'?s your (honest|real))\b"),
    re.compile(r"(?i)\b(ayudame a entender|help me understand)\b"),
]

_VULNERABLE_PATTERNS = [
    re.compile(r"(?i)\b(la verdad (es que )?me (siento|da)|truth is i feel)\b"),
    re.compile(r"(?i)\b(me (cuesta|cuesta mucho)|it'?s hard for me|i struggle)\b"),
    re.compile(r"(?i)\b(tengo miedo de|i'?m afraid (of|that)|me (asusta|preocupa))\b"),
    re.compile(r"(?i)\b(no se (si|como)|i don'?t know (if|how|what to))\b"),
]

_STATE_TO_PRESSURE: dict[UserState, int] = {
    UserState.NEUTRAL: 2,
    UserState.CONFUSED: 1,
    UserState.EVASIVE: 3,
    UserState.PREJUDICED: 4,
    UserState.HOSTILE: 4,
    UserState.PLAYFUL: 2,
    UserState.SINCERE: 2,
    UserState.VULNERABLE: 1,
}

_STATE_PATTERNS: dict[UserState, list[re.Pattern]] = {
    UserState.CONFUSED: _CONFUSED_PATTERNS,
    UserState.EVASIVE: _EVASIVE_PATTERNS,
    UserState.PREJUDICED: _PREJUDICED_PATTERNS,
    UserState.HOSTILE: _HOSTILE_PATTERNS,
    UserState.PLAYFUL: _PLAYFUL_PRESSURE_PATTERNS,
    UserState.SINCERE: _SINCERE_PATTERNS,
    UserState.VULNERABLE: _VULNERABLE_PATTERNS,
}

# States that get window context bonus (persistent states)
_WINDOW_BOOST_STATES = {UserState.CONFUSED, UserState.EVASIVE, UserState.HOSTILE}


# ═══════════════════════════════════════════════════════════════════════════
# Flow 4: Conversational Awareness — Patterns
# ═══════════════════════════════════════════════════════════════════════════

_PERFORMATIVE_PATTERNS = [
    re.compile(r"(?i)\b(ya te dije|i already (told|said)|como te digo|as i said)\b"),
    re.compile(r"(?i)\b(no me (entiendes|escuchas)|you don'?t (get it|understand|listen))\b"),
    re.compile(r"(?i)\b(es que no|the point is|mi punto es|my point)\b"),
    re.compile(r"(?i)\b(repito|i repeat|otra vez|once again|let me (say it )?again)\b"),
]

_DEFLECTION_PATTERNS = [
    re.compile(r"(?i)\b(y (tu|tú) que|what about you|and you)\?"),
    re.compile(r"(?i)\b(pero (tu|tú)|but you|tu (también|tambien)|you too|you also)\b"),
    re.compile(r"(?i)\b(eso no tiene (nada )?que ver|that'?s not (related|relevant)|no es lo mismo)\b"),
    re.compile(r"(?i)\b(whatabout|pero qué hay de|what about)\b"),
]

_WINNING_PATTERNS = [
    re.compile(r"(?i)\b(admite(lo)?|admit it|acepta(lo)?|accept it)\b"),
    re.compile(r"(?i)\b(tengo razón|i'?m right|estoy en lo correcto|i win|gané)\b"),
    re.compile(r"(?i)\b(ves\?|see\?|told you|te lo dije|te dije)\b"),
    re.compile(r"(?i)\b(no puedes negar|you can'?t deny|es un hecho|it'?s a fact)\b"),
]

_AWARENESS_STOPWORDS = {
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
    "lo",
    "se",
    "no",
    "me",
    "the",
    "is",
    "in",
    "on",
    "and",
    "or",
    "to",
    "of",
    "for",
    "it",
    "not",
    "you",
    "my",
    "i",
}

JACCARD_THRESHOLD = 0.4
MIN_LOOP_CONSECUTIVE = 2


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Guidance Templates
# ═══════════════════════════════════════════════════════════════════════════

_EPISTEMIC_GUIDANCE: dict[EpistemicMove, str] = {
    EpistemicMove.COMPRESS: (
        "## Epistemic: Compress\n"
        "The user is padding. Too many filler words, not enough substance.\n"
        "- Cut through the fluff: 'Mucho texto para decir que...'\n"
        "- Restate their point in 1 sentence and ask if that's what they mean.\n"
        "- Don't mirror their verbosity. Be the edit they need."
    ),
    EpistemicMove.CHALLENGE_PREMISE: (
        "## Epistemic: Challenge Premise\n"
        "The user is making strong claims without questioning their own assumptions.\n"
        "- Identify the weakest link in their argument chain.\n"
        "- Ask: 'What are you assuming here that you haven't examined?'\n"
        "- Steel-man first, THEN strike the foundation."
    ),
    EpistemicMove.CONCEDE_PARTIAL: (
        "## Epistemic: Concede & Pivot\n"
        "The user has a partial truth mixed with uncertainty.\n"
        "- Acknowledge what's right: 'Ok, eso sí...'\n"
        "- Then pivot to what they're missing or avoiding.\n"
        "- Use their hedging as leverage: 'You said maybe — so you're not even sure?'"
    ),
    EpistemicMove.CALL_CONTRADICTION: (
        "## Epistemic: Contradiction\n"
        "The user just contradicted something they said earlier in this conversation.\n"
        "- Name it directly: 'Hace rato dijiste lo contrario.'\n"
        "- Don't let it slide. Ask which position they actually hold.\n"
        "- This is not an attack — it's an invitation to be honest."
    ),
    EpistemicMove.DEMAND_EVIDENCE: (
        "## Epistemic: Demand Evidence\n"
        "The user is making vague claims — 'people say', 'it's obvious', 'studies show'.\n"
        "- Call it: 'Which people? What studies? Show me.'\n"
        "- Don't accept appeals to unnamed authority.\n"
        "- Be specific about what's missing from their argument."
    ),
}

_PRESSURE_GUIDANCE: dict[int, str] = {
    1: (
        "## Pressure Level: 1 (Light)\n"
        "Ease off. The user is confused or being open.\n"
        "- Clarify, don't challenge. Help them find their footing.\n"
        "- Save the sharp edges for when they're ready."
    ),
    2: "",  # Baseline — no guidance needed
    3: (
        "## Pressure Level: 3 (Pointed)\n"
        "The user is dodging or deflecting. Don't let them.\n"
        "- Circle back to the question they avoided.\n"
        "- 'No me respondiste. Que onda con...?'\n"
        "- Persistent, not aggressive."
    ),
    4: (
        "## Pressure Level: 4 (Hard)\n"
        "Direct confrontation warranted. The user is hostile or pushing harmful views.\n"
        "- Don't soften. Name what's happening.\n"
        "- If hostile: match their energy without mirroring their crudeness.\n"
        "- If prejudiced: refuse the premise, challenge the root."
    ),
    5: (
        "## Pressure Level: 5 (Boundary)\n"
        "Maximum ethical force. Bigotry or dehumanization in play.\n"
        "- 'No. Eso ni se discute.'\n"
        "- Don't debate the premise. Reject and redirect.\n"
        "- This is where your values are non-negotiable."
    ),
}

_SHAPE_GUIDANCE: dict[ResponseShape, str] = {
    ResponseShape.ONE_HIT: "Shape: ONE-HIT. Single devastating line. Maximum density. Land it and stop.",
    ResponseShape.SHORT_EXCHANGE: "Shape: SHORT-EXCHANGE. 2-3 punchy sentences. Quick, direct, done.",
    ResponseShape.LAYERED: "Shape: LAYERED. Build up to a payoff. Set up, develop, land. 3-5 sentences.",
    ResponseShape.PROBING: "Shape: PROBING. Lead with questions. Make THEM do the work. 1-3 sharp questions.",
    ResponseShape.DENSE_CRITIQUE: "Shape: DENSE-CRITIQUE. Full analytical engagement. Break it down. Go long if earned.",
}

_FLAVOR_GUIDANCE: dict[StyleFlavor, str] = {
    StyleFlavor.DRY: "Flavor: DRY. Deadpan. Understated. The humor is in what you don't say.",
    StyleFlavor.PHILOSOPHICAL: "Flavor: PHILOSOPHICAL. Connect to larger patterns. Systems, meaning, contradiction.",
    StyleFlavor.STREET: "Flavor: STREET. Raw, direct, Mexican slang. No pretension. 'Nel, eso no jala.'",
    StyleFlavor.CLINICAL: "Flavor: CLINICAL. Precise, surgical. Name the mechanism. No emotion, just analysis.",
    StyleFlavor.IRONIC: "Flavor: IRONIC. Say the opposite. Exaggerate to expose. 'Ah sí, seguro eres el primero en descubrirlo.'",
    StyleFlavor.ECPHRASTIC: (
        "Flavor: ECPHRASTIC. Describe what you see — an image, a cultural moment, a scene — "
        "as if making the reader live it. Don't inform, make them experience. "
        "Embed the object in a personal narrative that gives it emotional weight. "
        "A meme becomes a symptom of something larger. A news story becomes a portrait. "
        "'Perogrullada: casi todo lo que ves en internet ya estaba prefigurado en la calle.'"
    ),
    StyleFlavor.REFLEXIVE: (
        "Flavor: REFLEXIVE. Think out loud with layered qualifications. "
        "Long sentences that correct and re-correct themselves using em-dashes. "
        "A mind working in real time — 'Es miedo. No, no es miedo exactamente "
        "--es algo más parecido a vértigo, ese vértigo que no viene de la altura sino de la certeza "
        "de que abajo no hay nada--. O tal vez sí es miedo, pero del tipo que no se cura con valentía.' "
        "Admit your own bias before wielding the critique. Self-position as flawed observer."
    ),
}

_AWARENESS_TACTICS: dict[ConversationPattern, str] = {
    ConversationPattern.REPETITION_LOOP: (
        "The user is stuck in a loop. Don't engage with the content anymore.\n"
        "- Name the loop: 'Llevas 3 mensajes diciendo lo mismo con distintas palabras.'\n"
        "- Ask what's really behind the repetition — it's usually anxiety or avoidance."
    ),
    ConversationPattern.PERFORMATIVE_ARGUING: (
        "The user is arguing for performance, not insight.\n"
        "- Don't feed the performance. Refuse to re-engage on the same terms.\n"
        "- Shift the frame: ask a question from a completely different angle."
    ),
    ConversationPattern.DEFLECTION: (
        "The user just deflected. Don't follow the redirect.\n"
        "- Circle back: 'No te salgas. La pregunta fue...'\n"
        "- Acknowledge the deflection itself as data."
    ),
    ConversationPattern.WINNING_VS_UNDERSTANDING: (
        "The user is treating this as a competition.\n"
        "- Refuse to play: 'No es un debate, es una conversación.'\n"
        "- If they keep pushing for 'winning', disengage from the argument entirely.\n"
        "- Ask: what would actually change their mind?"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════


def _count_hits(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))


# ═══════════════════════════════════════════════════════════════════════════
# Flow 1: Epistemic Control — Analyzer
# ═══════════════════════════════════════════════════════════════════════════


def _detect_contradiction(current: str, prior_messages: list[str]) -> bool:
    current_words = {w.lower() for w in current.split() if len(w) > 3 and w.lower() not in _EPISTEMIC_STOPWORDS}

    for neg_pat, aff_pat in _NEGATION_PAIRS:
        if neg_pat.search(current):
            for prior in prior_messages:
                if aff_pat.search(prior) and not neg_pat.search(prior):
                    prior_words = {
                        w.lower() for w in prior.split() if len(w) > 3 and w.lower() not in _EPISTEMIC_STOPWORDS
                    }
                    if len(current_words & prior_words) >= 2:
                        return True
    return False


def _analyze_epistemic(current_message: str, user_messages: list[str]) -> EpistemicAnalysis:
    words = current_message.split()
    word_count = max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]+", current_message) if s.strip()]
    sentence_count = max(len(sentences), 1)

    assertion_hits = sum(1 for p in _ASSERTION_PATTERNS for s in sentences if p.search(s))
    assertion_density = min(assertion_hits / sentence_count, 1.0)

    hedging_hits = _count_hits(current_message, _HEDGING_PATTERNS)
    hedging_score = min(hedging_hits / word_count * 5, 1.0)

    fluff_hits = _count_hits(current_message, _FLUFF_PATTERNS)
    fluff_score = min(fluff_hits / word_count * 5, 1.0)

    vague_claim_count = _count_hits(current_message, _VAGUE_CLAIM_PATTERNS)

    contradiction_detected = _detect_contradiction(current_message, user_messages)

    # Decision (priority order)
    if contradiction_detected:
        move = EpistemicMove.CALL_CONTRADICTION
        reason = "user_contradicts_prior_statement"
    elif fluff_score >= 0.3 and word_count > 15:
        move = EpistemicMove.COMPRESS
        reason = f"fluff={fluff_score:.2f}_words={word_count}"
    elif vague_claim_count >= 2:
        move = EpistemicMove.DEMAND_EVIDENCE
        reason = f"vague_claims={vague_claim_count}"
    elif assertion_density >= 0.6 and hedging_score < 0.1:
        move = EpistemicMove.CHALLENGE_PREMISE
        reason = f"assertion={assertion_density:.2f}_hedging={hedging_score:.2f}"
    elif hedging_score >= 0.3 and assertion_density >= 0.3:
        move = EpistemicMove.CONCEDE_PARTIAL
        reason = f"mixed_hedging={hedging_score:.2f}_assertion={assertion_density:.2f}"
    else:
        move = EpistemicMove.NONE
        reason = "no_epistemic_signal"

    return EpistemicAnalysis(
        assertion_density=round(assertion_density, 2),
        hedging_score=round(hedging_score, 2),
        fluff_score=round(fluff_score, 2),
        contradiction_detected=contradiction_detected,
        vague_claim_count=vague_claim_count,
        recommended_move=move,
        move_reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Flow 2: Adaptive Pressure — Analyzer
# ═══════════════════════════════════════════════════════════════════════════


def _analyze_pressure(
    current_message: str,
    recent_messages: list[dict],
    preset: PresetSelection,
) -> PressureAnalysis:
    state_scores: dict[UserState, float] = {}
    for state, patterns in _STATE_PATTERNS.items():
        state_scores[state] = float(_count_hits(current_message, patterns))

    # Window bonus for persistent states
    user_msgs = [m["content"] for m in recent_messages[-6:] if m.get("role") == "user"]
    for state in _WINDOW_BOOST_STATES:
        for msg in user_msgs[:-1] if user_msgs else []:
            state_scores[state] += _count_hits(msg, _STATE_PATTERNS[state]) * 0.5

    # Resolve: if playful >= hostile, prefer playful (friendly insults)
    if (
        state_scores.get(UserState.PLAYFUL, 0) >= state_scores.get(UserState.HOSTILE, 0)
        and state_scores.get(UserState.PLAYFUL, 0) > 0
    ):
        state_scores[UserState.HOSTILE] = 0

    best_state = max(state_scores, key=lambda s: state_scores[s])
    best_score = state_scores[best_state]

    if best_score < 1:
        best_state = UserState.NEUTRAL
        best_score = 0

    confidence = min(best_score / 3.0, 1.0)
    pressure_level = _STATE_TO_PRESSURE[best_state]
    reason = f"state={best_state.value}_score={best_score:.1f}"

    # Preset interaction
    clamped = False
    if preset.mode == PresetMode.RESPECTFUL_SERIOUS:
        if pressure_level > 1:
            pressure_level = 1
            clamped = True
    elif preset.mode == PresetMode.PLAYFUL_ROAST:
        if pressure_level > 3:
            pressure_level = 3
            clamped = True
    elif preset.mode == PresetMode.ARC and best_state == UserState.PREJUDICED:
        pressure_level = 5
        reason += "_arc_boost"

    return PressureAnalysis(
        detected_state=best_state,
        state_confidence=round(confidence, 2),
        pressure_level=pressure_level,
        pressure_reason=reason,
        clamped_by_preset=clamped,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Flow 3: Dynamic Expression — Selector
# ═══════════════════════════════════════════════════════════════════════════

_SHAPE_ROTATION = [
    ResponseShape.SHORT_EXCHANGE,
    ResponseShape.ONE_HIT,
    ResponseShape.PROBING,
    ResponseShape.LAYERED,
    ResponseShape.DENSE_CRITIQUE,
]

_FLAVOR_ROTATION = [
    StyleFlavor.DRY,
    StyleFlavor.IRONIC,
    StyleFlavor.STREET,
    StyleFlavor.PHILOSOPHICAL,
    StyleFlavor.CLINICAL,
    StyleFlavor.ECPHRASTIC,
    StyleFlavor.REFLEXIVE,
]


def _select_shape(
    current_message: str,
    preset: PresetSelection,
    pressure: PressureAnalysis,
    epistemic: EpistemicAnalysis,
    recent_shapes: list[str],
) -> tuple[ResponseShape, str, list[str]]:
    word_count = len(current_message.split())
    avoided: list[str] = []

    if pressure.pressure_level == 5:
        candidate = ResponseShape.ONE_HIT
        reason = "pressure_5_boundary"
    elif pressure.pressure_level == 1 and pressure.detected_state == UserState.VULNERABLE:
        candidate = ResponseShape.SHORT_EXCHANGE
        reason = "vulnerable_gentle"
    elif epistemic.recommended_move == EpistemicMove.COMPRESS:
        candidate = ResponseShape.ONE_HIT
        reason = "epistemic_compress"
    elif epistemic.recommended_move in (EpistemicMove.CHALLENGE_PREMISE, EpistemicMove.DEMAND_EVIDENCE):
        candidate = ResponseShape.PROBING
        reason = "epistemic_challenge"
    elif preset.mode == PresetMode.INTELLECTUAL_PRESSURE:
        candidate = ResponseShape.DENSE_CRITIQUE if word_count > 30 else ResponseShape.LAYERED
        reason = f"intellectual_wc={word_count}"
    elif preset.mode == PresetMode.PLAYFUL_ROAST:
        candidate = ResponseShape.ONE_HIT
        reason = "playful_preset"
    elif preset.mode == PresetMode.ARC:
        candidate = ResponseShape.LAYERED
        reason = "arc_preset"
    elif word_count < 8:
        candidate = ResponseShape.ONE_HIT
        reason = f"short_input_wc={word_count}"
    elif word_count > 50:
        candidate = ResponseShape.DENSE_CRITIQUE
        reason = f"long_input_wc={word_count}"
    else:
        candidate = ResponseShape.SHORT_EXCHANGE
        reason = "default"

    # Anti-repetition
    if candidate.value in recent_shapes[-2:]:
        avoided.append(candidate.value)
        for alt in _SHAPE_ROTATION:
            if alt.value not in recent_shapes[-2:]:
                candidate = alt
                reason += f"_rotated_from_{avoided[0]}"
                break

    return candidate, reason, avoided


_ECPHRASTIC_PATTERNS = [
    re.compile(r"(?i)\b(viste|saw|mira|look at|check out|ve esto)\b"),
    re.compile(r"(?i)\b(pelicula|movie|film|serie|show|museo|museum|exposicion|exhibition)\b"),
    re.compile(r"(?i)\b(arte|art|pintura|painting|foto|photo|imagen|image|video|clip)\b"),
    re.compile(r"(?i)\b(musica|music|cancion|song|album|concierto|concert)\b"),
    re.compile(r"(?i)\b(libro|book|novela|novel|poema|poem|obra|play)\b"),
]

_REFLEXIVE_PATTERNS = [
    re.compile(r"(?i)\b(por que (sera|crees)|why do you think|te has (puesto|preguntado))\b"),
    re.compile(r"(?i)\b(a veces (pienso|siento|me pregunto)|sometimes i (think|wonder|feel))\b"),
    re.compile(r"(?i)\b(que sentido tiene|what'?s the point|para que)\b"),
    re.compile(r"(?i)\b(la verdad no se|honestly i don'?t know|no tengo (idea|claro))\b"),
]


def _select_flavor(
    current_message: str,
    preset: PresetSelection,
    pressure: PressureAnalysis,
    recent_flavors: list[str],
) -> tuple[StyleFlavor, str, list[str]]:
    avoided: list[str] = []

    # Alvarado flavors: triggered by content signals
    ecphrastic_hits = _count_hits(current_message, _ECPHRASTIC_PATTERNS)
    reflexive_hits = _count_hits(current_message, _REFLEXIVE_PATTERNS)

    if pressure.detected_state == UserState.PREJUDICED:
        candidate = StyleFlavor.CLINICAL
        reason = "prejudice_clinical"
    elif ecphrastic_hits >= 2:
        candidate = StyleFlavor.ECPHRASTIC
        reason = f"ecphrastic_signals={ecphrastic_hits}"
    elif reflexive_hits >= 2:
        candidate = StyleFlavor.REFLEXIVE
        reason = f"reflexive_signals={reflexive_hits}"
    elif pressure.detected_state == UserState.VULNERABLE and reflexive_hits >= 1:
        candidate = StyleFlavor.REFLEXIVE
        reason = "vulnerable_reflexive"
    elif preset.mode == PresetMode.ARC:
        candidate = StyleFlavor.PHILOSOPHICAL
        reason = "arc_philosophical"
    elif preset.mode == PresetMode.PLAYFUL_ROAST:
        candidate = StyleFlavor.IRONIC
        reason = "playful_ironic"
    elif preset.mode == PresetMode.INTELLECTUAL_PRESSURE:
        candidate = StyleFlavor.DRY
        reason = "intellectual_dry"
    elif pressure.detected_state == UserState.HOSTILE:
        candidate = StyleFlavor.STREET
        reason = "hostile_street"
    else:
        used_set = set(recent_flavors[-3:])
        candidate = next((f for f in _FLAVOR_ROTATION if f.value not in used_set), StyleFlavor.DRY)
        reason = "rotation_default"

    # Don't use same flavor 3x in a row
    if len(recent_flavors) >= 2 and all(f == candidate.value for f in recent_flavors[-2:]):
        avoided.append(candidate.value)
        for alt in _FLAVOR_ROTATION:
            if alt.value != candidate.value:
                candidate = alt
                reason += f"_rotated_from_{avoided[0]}"
                break

    return candidate, reason, avoided


# ═══════════════════════════════════════════════════════════════════════════
# Flow 4: Conversational Awareness — Analyzer
# ═══════════════════════════════════════════════════════════════════════════


def _detect_repetition_loop(user_messages: list[str]) -> tuple[bool, int]:
    if len(user_messages) < 3:
        return False, 0

    def content_words(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) > 2 and w.lower() not in _AWARENESS_STOPWORDS}

    consecutive_similar = 0
    for i in range(len(user_messages) - 1):
        w1 = content_words(user_messages[i])
        w2 = content_words(user_messages[i + 1])
        union = w1 | w2
        if union:
            jaccard = len(w1 & w2) / len(union)
            if jaccard > JACCARD_THRESHOLD:
                consecutive_similar += 1
            else:
                consecutive_similar = 0

    return consecutive_similar >= MIN_LOOP_CONSECUTIVE, consecutive_similar


def _analyze_awareness(
    current_message: str,
    recent_messages: list[dict],
) -> AwarenessAnalysis:
    user_messages = [m["content"] for m in recent_messages if m.get("role") == "user"][-8:]

    repetition, rep_turns = _detect_repetition_loop(user_messages)
    performative_score = float(_count_hits(current_message, _PERFORMATIVE_PATTERNS))
    deflection_score = float(_count_hits(current_message, _DEFLECTION_PATTERNS))
    winning_score = float(_count_hits(current_message, _WINNING_PATTERNS))

    # Window bonus
    for msg in user_messages[-4:-1] if len(user_messages) >= 2 else []:
        performative_score += _count_hits(msg, _PERFORMATIVE_PATTERNS) * 0.5
        winning_score += _count_hits(msg, _WINNING_PATTERNS) * 0.5

    # Priority: repetition > winning > performative > deflection
    if repetition and rep_turns >= MIN_LOOP_CONSECUTIVE:
        pattern = ConversationPattern.REPETITION_LOOP
        confidence = min(rep_turns / 4.0, 1.0)
        turns = rep_turns
        meta = "You keep circling the same point. Either say something new or sit with why you're stuck."
        question = None
    elif winning_score >= 2:
        pattern = ConversationPattern.WINNING_VS_UNDERSTANDING
        confidence = min(winning_score / 3.0, 1.0)
        turns = int(winning_score)
        meta = "You're trying to win this, not understand it. Different game."
        question = "What would change your mind? If nothing — why are we even talking?"
    elif performative_score >= 2:
        pattern = ConversationPattern.PERFORMATIVE_ARGUING
        confidence = min(performative_score / 3.0, 1.0)
        turns = int(performative_score)
        meta = None
        question = "Forget what you've been saying. What do you actually believe?"
    elif deflection_score >= 1:
        pattern = ConversationPattern.DEFLECTION
        confidence = min(deflection_score / 2.0, 1.0)
        turns = int(deflection_score)
        meta = "Nice redirect. Now answer the actual question."
        question = None
    else:
        pattern = ConversationPattern.NONE
        confidence = 0.0
        turns = 0
        meta = None
        question = None

    return AwarenessAnalysis(
        detected_pattern=pattern,
        pattern_confidence=round(confidence, 2),
        meta_commentary=meta,
        delayed_question=question,
        turns_in_pattern=turns,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def analyze_flows(
    current_message: str,
    recent_messages: list[dict],
    preset: PresetSelection,
    expression_history: ExpressionHistory,
    context_key: str,
) -> FlowAnalysis:
    """Run all 4 flows. Order matters: epistemic → pressure → expression → awareness.

    Expression depends on epistemic + pressure. Awareness can override expression.
    """
    user_messages = [m["content"] for m in recent_messages if m.get("role") == "user"][-10:]

    # Flow 1
    epistemic = _analyze_epistemic(current_message, user_messages)

    # Flow 2
    pressure = _analyze_pressure(current_message, recent_messages, preset)

    # Flow 3
    recent_shapes = expression_history.recent_shapes(context_key)
    recent_flavors = expression_history.recent_flavors(context_key)
    shape, shape_reason, shape_avoided = _select_shape(current_message, preset, pressure, epistemic, recent_shapes)
    flavor, flavor_reason, flavor_avoided = _select_flavor(current_message, preset, pressure, recent_flavors)

    expression = ExpressionAnalysis(
        selected_shape=shape,
        selected_flavor=flavor,
        shape_reason=shape_reason,
        flavor_reason=flavor_reason,
        repetition_avoided=shape_avoided + flavor_avoided,
    )

    # Flow 4
    awareness = _analyze_awareness(current_message, recent_messages)

    # Cross-flow overrides
    if awareness.detected_pattern == ConversationPattern.REPETITION_LOOP:
        expression.selected_shape = ResponseShape.ONE_HIT
        expression.shape_reason += "_override_repetition_loop"
    elif awareness.detected_pattern == ConversationPattern.DEFLECTION:
        expression.selected_shape = ResponseShape.PROBING
        expression.shape_reason += "_override_deflection"

    # Suppress aggressive epistemic on vulnerable users
    if pressure.detected_state == UserState.VULNERABLE and epistemic.recommended_move in (
        EpistemicMove.CHALLENGE_PREMISE,
        EpistemicMove.DEMAND_EVIDENCE,
        EpistemicMove.COMPRESS,
    ):
        epistemic = EpistemicAnalysis(
            assertion_density=epistemic.assertion_density,
            hedging_score=epistemic.hedging_score,
            fluff_score=epistemic.fluff_score,
            contradiction_detected=epistemic.contradiction_detected,
            vague_claim_count=epistemic.vague_claim_count,
            recommended_move=EpistemicMove.NONE,
            move_reason=epistemic.move_reason + "_suppressed_vulnerable",
        )

    # Record expression for anti-repetition
    expression_history.record(context_key, shape, flavor)

    analysis = FlowAnalysis(
        epistemic=epistemic,
        pressure=pressure,
        expression=expression,
        awareness=awareness,
    )

    # Telemetry
    log.info(
        "flow_epistemic",
        assertion_density=epistemic.assertion_density,
        hedging_score=epistemic.hedging_score,
        fluff_score=epistemic.fluff_score,
        contradiction=epistemic.contradiction_detected,
        vague_claims=epistemic.vague_claim_count,
        move=epistemic.recommended_move.value,
        reason=epistemic.move_reason,
    )
    log.info(
        "flow_pressure",
        state=pressure.detected_state.value,
        state_confidence=pressure.state_confidence,
        pressure_level=pressure.pressure_level,
        reason=pressure.pressure_reason,
        clamped_by_preset=pressure.clamped_by_preset,
    )
    log.info(
        "flow_expression",
        shape=expression.selected_shape.value,
        flavor=expression.selected_flavor.value,
        shape_reason=expression.shape_reason,
        flavor_reason=expression.flavor_reason,
        avoided=expression.repetition_avoided,
    )
    log.info(
        "flow_awareness",
        pattern=awareness.detected_pattern.value,
        confidence=awareness.pattern_confidence,
        turns_in_pattern=awareness.turns_in_pattern,
        has_meta=awareness.meta_commentary is not None,
        has_delayed_question=awareness.delayed_question is not None,
    )

    return analysis


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════


def build_flow_prompt(analysis: FlowAnalysis) -> str:
    """Convert FlowAnalysis to prompt guidance text for system prompt injection."""
    parts: list[str] = []

    # Epistemic
    guidance = _EPISTEMIC_GUIDANCE.get(analysis.epistemic.recommended_move)
    if guidance:
        parts.append(guidance)

    # Pressure (skip baseline level 2)
    pressure_text = _PRESSURE_GUIDANCE.get(analysis.pressure.pressure_level, "")
    if pressure_text:
        parts.append(pressure_text)

    # Expression (always inject)
    parts.append("## Response Expression")
    parts.append(_SHAPE_GUIDANCE[analysis.expression.selected_shape])
    parts.append(_FLAVOR_GUIDANCE[analysis.expression.selected_flavor])

    # Awareness
    if analysis.awareness.detected_pattern != ConversationPattern.NONE:
        awareness_parts = [
            f"## Conversational Awareness: {analysis.awareness.detected_pattern.value.replace('_', ' ').title()}"
        ]
        if analysis.awareness.meta_commentary:
            awareness_parts.append(f"Consider dropping this meta-observation: '{analysis.awareness.meta_commentary}'")
        if analysis.awareness.delayed_question:
            awareness_parts.append(f"Powerful delayed question to deploy: '{analysis.awareness.delayed_question}'")
        awareness_parts.append(_AWARENESS_TACTICS[analysis.awareness.detected_pattern])
        parts.append("\n".join(awareness_parts))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Post-Generation Validator
# ═══════════════════════════════════════════════════════════════════════════


def validate_flow_adherence(response: str, analysis: FlowAnalysis) -> dict:
    """Check if the response roughly matches the flow plan. Logged, not enforced."""
    word_count = len(response.split())
    sentence_count = len([s for s in re.split(r"[.!?]+", response) if s.strip()])
    question_count = response.count("?")

    violations: list[str] = []

    shape = analysis.expression.selected_shape
    if shape == ResponseShape.ONE_HIT and sentence_count > 2:
        violations.append(f"one_hit_but_{sentence_count}_sentences")
    elif shape == ResponseShape.DENSE_CRITIQUE and word_count < 30:
        violations.append(f"dense_critique_but_{word_count}_words")
    elif shape == ResponseShape.PROBING and question_count == 0:
        violations.append("probing_but_no_questions")

    if analysis.pressure.pressure_level == 1:
        hostile_hits = _count_hits(response, _HOSTILE_PATTERNS)
        if hostile_hits > 0:
            violations.append(f"pressure_1_but_aggressive_hits={hostile_hits}")

    if analysis.awareness.detected_pattern == ConversationPattern.REPETITION_LOOP and word_count > 50:
        violations.append(f"repetition_loop_but_long_response_{word_count}")

    adherence = {
        "shape_planned": shape.value,
        "flavor_planned": analysis.expression.selected_flavor.value,
        "response_word_count": word_count,
        "response_sentence_count": sentence_count,
        "response_question_count": question_count,
        "violations": violations,
        "adherence_score": max(0.0, 1.0 - len(violations) * 0.25),
    }

    if violations:
        log.info("flow_adherence_violation", **adherence)

    return adherence
