"""Endpoint di interrogazione del RAG, isolato per utente.

Questo è l'endpoint "finale" che mette insieme tutti gli altri pezzi:
autenticazione (UtenteCorrente), ricerca vettoriale filtrata per utente
(retriever_per_utente) e il grafo di ragionamento LangGraph già esistente
in rag/graph.py (che qui NON abbiamo toccato: lo riusiamo così com'era
nella versione a riga di comando del progetto).
"""

from fastapi import APIRouter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from rag.config import K
from rag.graph import costruisci_grafo
from rag.vectorstore import retriever_per_utente

from ..deps import UtenteCorrente
from ..state import get_embeddings, get_llm

router = APIRouter(tags=["ask"])

# Un ChatPromptTemplate è un "modello" di prompt con dei segnaposto
# ({context} e {question}) che LangChain riempie con i valori veri al
# momento della chiamata. Tenerlo come costante di modulo evita di
# ricrearlo identico ad ogni richiesta.
_PROMPT = ChatPromptTemplate.from_template(
    "Sei un assistente su dieta, spesa e allenamento. Rispondi usando SOLO il "
    "contesto. Se l'informazione non c'è, dillo. Rispondi in italiano, conciso.\n\n"
    "CONTESTO:\n{context}\n\nDOMANDA: {question}"
)


class DomandaIn(BaseModel):
    domanda: str


class RispostaOut(BaseModel):
    risposta: str


@router.post("/ask", response_model=RispostaOut)
def ask(payload: DomandaIn, utente: UtenteCorrente):
    """Risponde a una domanda usando solo i documenti dell'utente autenticato."""
    embeddings = get_embeddings()
    # Il filtro per utente avviene qui: questo retriever, quando LangGraph
    # gli chiederà "trovami i chunk più pertinenti", cercherà SOLO tra i
    # vettori con metadato user_id uguale a quello dell'utente loggato.
    retriever = retriever_per_utente(embeddings, utente.id, K)
    llm = get_llm()
    # Il grafo viene ricostruito ad ogni richiesta perché il retriever
    # cambia a seconda di CHI sta chiedendo — costruirlo è un'operazione
    # leggera (assembla solo delle funzioni), quindi non è uno spreco.
    grafo = costruisci_grafo(retriever, llm, _PROMPT)

    # `.invoke(...)` esegue il grafo dall'inizio alla fine e ritorna lo
    # stato finale come dizionario Python; ne leggiamo solo la chiave
    # "risposta" (le altre, come "documenti" e "tentativi", sono dettagli
    # interni del grafo che non servono a chi chiama l'API).
    risultato = grafo.invoke({"domanda": payload.domanda, "tentativi": 0})
    return RispostaOut(risposta=risultato["risposta"])
