"""Post-generation validator — logs when the response diverged from the plan.

Runs AFTER the LLM produced output. Compares the realized response
against what the flow plan demanded (shape length bounds, probing
requires ?, pressure-1 forbids hostility, repetition-loop forbids
long replies). Records violations but does NOT block delivery — this
is a soft-monitoring signal for drift analysis, not a gate."""

from __future__ import annotations

import re

import structlog

from insult.core.flows.patterns import HOSTILE_PATTERNS, count_hits
from insult.core.flows.types import (
    ConversationPattern,
    FlowAnalysis,
    ResponseShape,
)

log = structlog.get_logger()


def validate_flow_adherence(response: str, analysis: FlowAnalysis) -> dict:
    """Compare realized response to the flow plan. Returns a dict of
    violations + adherence score. Logged, not enforced."""
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
        hostile_hits = count_hits(response, HOSTILE_PATTERNS)
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
