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
}


def get_error_response(error_type: str) -> str:
    """Returns a random in-character error response."""
    responses = ERROR_RESPONSES.get(error_type, ERROR_RESPONSES["generic"])
    return random.choice(responses)


def classify_error(exc: Exception) -> str:
    """Map exception type to error category."""
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout"
    if "RateLimit" in name:
        return "rate_limit"
    if "Authentication" in name:
        return "auth"
    return "generic"
