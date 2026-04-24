"""Regression tests for RESPECTFUL_SERIOUS classification of clinical speech.

Context: on 2026-04-23 a user (kh1_maera, hereafter "Alex", user_id
1431300030823927999) reported that the bot responded curtly to them despite
them writing about their ongoing psychiatric treatment and complex trauma.
A dump of their last 15 messages from production confirmed every single
message fell through to DEFAULT_ABRASIVE because _SERIOUS_PATTERNS only
matched lay-person crisis phrases ("suicid", "depres", "me siento solo")
and had zero coverage for the clinical vocabulary patients actually use.

These tests fix that: each quote below is VERBATIM from Alex's real
messages (trimmed for length), and each MUST classify as
RESPECTFUL_SERIOUS. If any of these regress to a different mode, a real
person describing their trauma will be roasted by the bot."""

import pytest

from insult.core.presets import PresetMode, classify_preset


@pytest.mark.parametrize(
    "message",
    [
        # Direct diagnosis disclosure — the core leak before the fix
        "Lo que tengo es Estrés Postraumático Complejo, entre otras cosas jeje",
        # Medication timing question — clinically loaded
        "Ah ok, es que había leído que tanto la sertralina como la quetiapina empiezan a hacer efecto a las semanas o algo así",
        # Direct question about psychiatric drug mechanism
        "Jejeje me podrías explicar a qué se debe este efecto de bienestar de la quetiapina en particular",
        # Dosage + adaptation report
        "He dormido bastante bien, y cada día me siento un poquito mejor, mi cuerpx se sigue adaptando a las nuevas dosis",
        # Treatment plan + clinician relationship
        "Lo que haré es seguir una comunicación muy cercana con la neuropsiquiatra respecto a mis síntomas",
        # Refusing involuntary hospitalization — a high-stakes boundary
        "Yo no quisiera internarme la verdad, no creo que respeten mis ajustes razonables como tener conmigo mis audífonos con cancelación",
        # Medication adjustment disclosure
        "Me aumentó dosis de quetiapina y sertralina",
        # Artritis / chronic-illness disclosure
        "Tengo artritis en proceso de diagnóstico y otras cosas",
    ],
)
def test_alex_real_messages_classify_as_respectful_serious(message: str):
    """Each verbatim production quote must route to the respectful mode.

    Before the fix each of these matched ZERO patterns and hit the abrasive
    fallback. The test failing means the bot is back to roasting people
    disclosing psychiatric treatment — an ethics regression, not a style one."""
    result = classify_preset(message)
    assert result.mode == PresetMode.RESPECTFUL_SERIOUS, (
        f"Expected RESPECTFUL_SERIOUS for clinical message, got {result.mode.value}. "
        f"Reason: {result.reason!r}. Message: {message!r}"
    )


def test_clinical_vocabulary_beats_abrasive_default():
    """Spot-check: a message containing psychiatric terms but no classic
    crisis phrases must still route to RESPECTFUL_SERIOUS."""
    result = classify_preset("estoy tomando mi dosis de quetiapina hoy")
    assert result.mode == PresetMode.RESPECTFUL_SERIOUS


def test_plain_chit_chat_is_not_accidentally_serious():
    """Inverse regression: ensure the expanded patterns didn't over-trigger.
    Neutral small talk must NOT classify as serious."""
    result = classify_preset("hola qué tal, cómo va tu día")
    assert result.mode != PresetMode.RESPECTFUL_SERIOUS


# ---- Vulnerable user overlay (Component A + B) ----
#
# These tests lock in that a user with accumulated clinical signals in their
# fact store ALWAYS gets RESPECTFUL_SERIOUS, even when their current message
# carries zero direct trigger words. This is the fix for the "bot is curt
# with Alex when he says he slept well" failure mode — "he dormido bien"
# has no clinical keyword, but his cumulative profile does.


def _alex_like_facts() -> list[dict]:
    """Reconstruct Alex's fact cluster (diagnosis + meds + clinician)."""
    return [
        {"fact": "Tiene Estrés Postraumático Complejo (CPTSD)"},
        {"fact": "Toma quetiapina y sertralina, dosis aumentada recientemente"},
        {"fact": "Sigue tratamiento con una neuropsiquiatra"},
        {"fact": "Tiene artritis en diagnóstico"},
    ]


def test_vulnerable_user_neutral_message_still_classifies_serious():
    """The 'he dormido bien' class of message: no clinical keyword at all,
    but user is vulnerable → must still route to RESPECTFUL_SERIOUS."""
    result = classify_preset(
        "He dormido bastante bien, y cada día me siento un poquito mejor",
        user_facts=_alex_like_facts(),
    )
    assert result.mode == PresetMode.RESPECTFUL_SERIOUS
    assert result.reason.startswith("vulnerable_user_overlay")


def test_vulnerable_user_medication_follow_up():
    """A message that looks like off-topic reflection but carries medication
    context in history. Must NOT fall through to abrasive."""
    result = classify_preset(
        "Yo sentí efectos muy positivos desde las primeras tomas, como una especie de contención que aumentaba cada día",
        user_facts=_alex_like_facts(),
    )
    assert result.mode == PresetMode.RESPECTFUL_SERIOUS


def test_non_vulnerable_user_neutral_message_is_not_serious():
    """Inverse regression: a user without clinical facts writing casually
    must NOT get forced into RESPECTFUL_SERIOUS."""
    facts = [
        {"fact": "Le gusta el café"},
        {"fact": "Es programador"},
    ]
    result = classify_preset("he dormido bien hoy", user_facts=facts)
    assert result.mode != PresetMode.RESPECTFUL_SERIOUS
