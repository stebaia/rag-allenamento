"""Endpoint dedicato alla skill Alexa, chiamato da una Lambda intermedia.

Non riusiamo /ask (vedi ask.py) perché lì l'autenticazione è UtenteCorrente
(JWT da login): Alexa non fa login, quindi qui l'accesso è protetto da una
semplice chiave segreta condivisa (vedi AlexaUserId in ../deps.py), e
l'utente a cui rispondere è fisso (ALEXA_USER_ID), non ricavato dal token.

La Lambda della skill si occupa di tutta la parte "Alexa Skills Kit"
(intent, slot, outputSpeech): qui riceviamo/rispondiamo con un JSON semplice
{"domanda": "..."} / {"risposta": "..."}, esattamente come /ask.
"""

from fastapi import APIRouter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from rag.config import K
from rag.graph import costruisci_grafo
from rag.vectorstore import retriever_per_utente

from ..deps import AlexaUserId
from ..state import get_embeddings, get_llm

router = APIRouter(prefix="/alexa", tags=["alexa"])

# Stesso prompt di /ask, ma con un'indicazione in più: la risposta verrà
# LETTA AD ALTA VOCE da Alexa, quindi deve restare breve e senza formattazione
# (niente elenchi puntati, markdown, ecc. che non hanno senso parlati).
_PROMPT = ChatPromptTemplate.from_template(
    "Sei un assistente vocale su dieta, spesa e allenamento. Rispondi usando "
    "SOLO il contesto. Se l'informazione non c'è, dillo. Rispondi in "
    "italiano, in una o due frasi brevi, senza elenchi puntati o "
    "formattazione: la risposta verrà letta ad alta voce.\n\n"
    "CONTESTO:\n{context}\n\nDOMANDA: {question}"
)


class DomandaIn(BaseModel):
    domanda: str


class RispostaOut(BaseModel):
    risposta: str


@router.post("/ask", response_model=RispostaOut)
def alexa_ask(payload: DomandaIn, user_id: AlexaUserId):
    """Risponde a una domanda posta tramite la skill Alexa (utente fisso, no login)."""
    embeddings = get_embeddings()
    retriever = retriever_per_utente(embeddings, user_id, K)
    llm = get_llm()
    grafo = costruisci_grafo(retriever, llm, _PROMPT)

    risultato = grafo.invoke({"domanda": payload.domanda, "tentativi": 0})
    return RispostaOut(risposta=risultato["risposta"])
