"""Lambda della skill Alexa "RAG Allenamento".

Fa da traduttore tra Alexa Skills Kit e il backend RAG:
  1. riceve la richiesta Alexa (intent + slot con la domanda dell'utente)
  2. chiama POST {RAG_API_URL}/alexa/ask con la chiave segreta condivisa
  3. traduce la risposta testuale in una risposta vocale Alexa (outputSpeech)

Variabili d'ambiente da configurare sulla funzione Lambda (console AWS):
  RAG_API_URL   es. https://tuo-dominio.example.com  (senza slash finale)
  RAG_API_KEY   la stessa chiave impostata come ALEXA_API_KEY nel backend

Dipendenze: il modulo `ask_sdk_core` / `ask_sdk_model` (pacchetto pip
"ask-sdk", va incluso nel deployment package della Lambda insieme a questo
file — vedi ask-skill.json per come è organizzata la skill).
"""

import os

import requests
from ask_sdk_core.dispatch_components import (
    AbstractExceptionHandler,
    AbstractRequestHandler,
)
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.utils import is_intent_name, is_request_type
from ask_sdk_model import Response

RAG_API_URL = os.environ["RAG_API_URL"].rstrip("/")
RAG_API_KEY = os.environ["RAG_API_KEY"]

# Frase di benvenuto e messaggio di errore, tenuti come costanti per non
# ripeterli identici nei vari handler sotto.
BENVENUTO = (
    "Ciao! Puoi chiedermi cosa mangiare, cosa comprare o come allenarti. "
    "Cosa vuoi sapere?"
)
ERRORE_GENERICO = (
    "Mi dispiace, non sono riuscito a recuperare la risposta. Riprova tra poco."
)


def _chiedi_al_rag(domanda: str) -> str:
    """Chiama il backend RAG e ritorna il testo della risposta.

    Timeout esplicito: senza limite, una richiesta lenta lascerebbe Alexa
    (e la Lambda, che viene fatturata a tempo) in attesa troppo a lungo.
    """
    risposta = requests.post(
        f"{RAG_API_URL}/alexa/ask",
        json={"domanda": domanda},
        headers={"X-Api-Key": RAG_API_KEY},
        timeout=20,
    )
    risposta.raise_for_status()
    return risposta.json()["risposta"]


class LaunchRequestHandler(AbstractRequestHandler):
    """Gestisce l'apertura della skill senza un comando specifico ("Alexa, apri...")."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return (
            handler_input.response_builder.speak(BENVENUTO)
            .ask("Cosa vuoi sapere?")
            .response
        )


class ChiediIntentHandler(AbstractRequestHandler):
    """Gestisce l'intent custom con lo slot contenente la domanda vera e propria.

    Nel modello di interazione della skill (interaction model, vedi
    ask-skill.json) va definito un intent "ChiediIntent" con uno slot
    "domanda" di tipo AMAZON.SearchQuery (o simile) e frasi di esempio come
    "chiedi {domanda}", "dimmi {domanda}", "{domanda}".
    """

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("ChiediIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        slots = handler_input.request_envelope.request.intent.slots
        domanda = slots["domanda"].value if slots and "domanda" in slots else None

        if not domanda:
            return (
                handler_input.response_builder.speak(
                    "Non ho capito la domanda, puoi ripeterla?"
                )
                .ask("Cosa vuoi sapere?")
                .response
            )

        try:
            testo_risposta = _chiedi_al_rag(domanda)
        except requests.RequestException:
            return handler_input.response_builder.speak(ERRORE_GENERICO).response

        return (
            handler_input.response_builder.speak(testo_risposta)
            .ask("Vuoi chiedere altro?")
            .response
        )


class AiutoIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return (
            handler_input.response_builder.speak(BENVENUTO)
            .ask("Cosa vuoi sapere?")
            .response
        )


class EsciIntentHandler(AbstractRequestHandler):
    """Gestisce insieme Stop/Cancel (chiusura esplicita) e SessionEnded (chiusura dal sistema)."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_intent_name("AMAZON.StopIntent")(handler_input)
            or is_intent_name("AMAZON.CancelIntent")(handler_input)
            or is_request_type("SessionEndedRequest")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.speak("A presto!").response


class GestoreErroriGenerico(AbstractExceptionHandler):
    """Rete di sicurezza: qualunque eccezione non gestita altrove finisce qui,
    così la skill risponde sempre qualcosa invece di andare in errore silenzioso."""

    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        return True

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        return (
            handler_input.response_builder.speak(ERRORE_GENERICO)
            .ask("Cosa vuoi sapere?")
            .response
        )


sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChiediIntentHandler())
sb.add_request_handler(AiutoIntentHandler())
sb.add_request_handler(EsciIntentHandler())
sb.add_exception_handler(GestoreErroriGenerico())

# `lambda_handler` è il nome che va indicato in AWS come "Handler" della
# funzione Lambda (formato file.funzione: "lambda_function.lambda_handler").
lambda_handler = sb.lambda_handler()
