"""In-character error responses — Insult never breaks the fourth wall."""

import random

ERROR_RESPONSES = {
    "timeout": [
        "Se me fue la onda un momento. Preguntame otra vez, andale.",
        "Me distraje pensando en lo pendeja que fue tu ultima pregunta. Repite.",
        "Perdi el hilo. No es culpa mia, es que lo que dices es tan aburrido que me desconecto.",
    ],
    "rate_limit": [
        "Tranquilo, tigre. Estoy procesando tus pendejadas. Dame un segundo.",
        "Cual es la prisa? Se te acaba el internet? Esperate tantito.",
        "Voy a necesitar un momento. Tanta estupidez junta me aturdio.",
    ],
    "auth": [
        "Algo trono por dentro. No es tu culpa... bueno, probablemente si. Intentale despues.",
        "Tengo un problema tecnico. Si, yo tambien los tengo. Sorprendido? Intentale al rato.",
    ],
    "generic": [
        "Se me trabo algo. Y no, no te voy a explicar que. Intentale de nuevo.",
        "Algo salio mal y no tengo ganas de explicarte. Repite tu pregunta.",
        "Tuve un tropiezo. Pasa hasta en las mejores familias. Vuelve a intentar.",
    ],
    "too_long": [
        "Neta escribiste un libro entero. Quieres que te lo publique o que te responda? Hazlo mas corto.",
        "Eso es un mensaje o una tesis doctoral? Resumelo, no tengo toda la vida.",
    ],
    "context_failed": [
        "Se me olvido todo lo que hablamos. Si, asi de memorable eres. Preguntame de nuevo.",
        "Mi memoria me fallo. No te emociones, no es que seas importante. Repite.",
    ],
    "retry_notice": [
        "Se me corto la linea, un segundo.",
        "Dame chance, estoy reintentando.",
        "Se me trabo tantito, ya voy.",
    ],
    # "billing" fires when Anthropic returns a 400 BadRequestError with
    # "credit balance is too low" in the message. The in-character response
    # tells the OPERATOR (via Discord, in front of other users) that the
    # API wallet is empty — so they don't have to curl logs to know why
    # the bot went quiet. See classify_error() for the detection logic.
    "billing": [
        "Se me acabo el changarro. Dile a mi patron que me recargue.",
        "Sin pisto no hay insultos, compa. Dile a Bernard que pague la luz.",
        "Se acabaron las fichas. Recarga mi cuenta y vuelvo.",
    ],
}


def get_error_response(error_type: str) -> str:
    """Returns a random in-character error response."""
    responses = ERROR_RESPONSES.get(error_type, ERROR_RESPONSES["generic"])
    return random.choice(responses)


def classify_error(exc: Exception) -> str:
    """Map an exception to the error category used by `get_error_response`.

    Order matters: specific billing / status-code checks run BEFORE the
    coarse name-based matchers because an Anthropic BadRequestError that
    actually means "you're out of credits" must route to the billing
    bucket (operator-visible), not to the generic "tuve un tropiezo"
    that hides the real problem. See 2026-04-24 incident where a low
    credit balance silently produced "generic" errors for an hour
    because the classifier only matched on `type().__name__`.
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    # Billing: Anthropic returns a 400 with this specific phrase when the
    # workspace runs out of prepaid credits. Detect by message content
    # rather than exception subtype — the exception IS a BadRequestError
    # but so are many unrelated 400s, so class matching alone is too broad.
    if "credit balance" in msg or "credit_balance" in msg or "insufficient_quota" in msg:
        return "billing"

    if "Timeout" in name:
        return "timeout"
    if "RateLimit" in name:
        return "rate_limit"
    if "Authentication" in name:
        return "auth"
    return "generic"
