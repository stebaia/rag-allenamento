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

from rag.config import K, MAX_STORICO
from rag.graph import costruisci_grafo, riformula_con_storico
from rag.vectorstore import retriever_per_utente

from ..deps import UtenteCorrente
from ..state import get_embeddings, get_llm

router = APIRouter(tags=["ask"])

# Un ChatPromptTemplate è un "modello" di prompt con dei segnaposto
# ({context} e {question}) che LangChain riempie con i valori veri al
# momento della chiamata. Tenerlo come costante di modulo evita di
# ricrearlo identico ad ogni richiesta.
#
# {oggi}: il grafo (vedi rag/graph.py, risolvi_giorni_relativi) sostituisce
# già "oggi"/"domani"/"ieri" nella domanda col nome del giorno prima di
# cercare nei documenti; qui ripetiamo l'informazione anche nel prompt così
# l'LLM può rispondere in modo naturale (es. dire "oggi" invece del nome del
# giorno) e risolvere correttamente riferimenti relativi impliciti.
#
# {storico}: gli scambi precedenti della conversazione (vuoto alla prima
# domanda). Serve a dare continuità alle risposte (es. capire "e quella
# giornata?" riferito a un giorno nominato prima) — il RECUPERO dei
# documenti usa invece la domanda già riformulata in modo autonomo (vedi
# riformula_con_storico), lo storico qui serve solo per il tono/riferimenti
# della risposta finale.
_PROMPT = ChatPromptTemplate.from_template(
    "Sei un assistente su dieta, spesa e allenamento. Oggi è {oggi}. Rispondi "
    "usando SOLO il contesto. Se l'informazione non c'è, dillo. Rispondi in "
    "italiano, conciso.\n\n"
    # Il contesto può contenere chunk di DUE documenti diversi (la dieta di un
    # giorno e la lista della spesa, vedi il boost in rag/vectorstore.py).
    # Senza questa istruzione l'LLM tende a rispondere solo col primo e a
    # elencare gli alimenti del pasto invece delle quantità da comprare.
    "Se la domanda riguarda la spesa, riporta le quantità e le indicazioni "
    "della LISTA DELLA SPESA (es. '1 confezione'), non i grammi dei pasti "
    "del piano alimentare: sono due cose diverse. Se nel contesto trovi sia "
    "i pasti sia le voci della lista della spesa, incrociali e indica cosa "
    "comprare per gli alimenti di quei pasti.\n\n"
    "CONVERSAZIONE PRECEDENTE:\n{storico}\n\n"
    "CONTESTO:\n{context}\n\nDOMANDA: {question}"
)


class MessaggioIn(BaseModel):
    ruolo: str  # "user" oppure "assistant"
    contenuto: str


class DomandaIn(BaseModel):
    domanda: str
    # Cronologia della chat PRIMA di questa domanda (dal messaggio meno
    # recente al più recente). Il frontend la costruisce dai messaggi già
    # mostrati a schermo (vedi frontend/src/lib/chat-context.tsx) — il
    # backend non conserva nessuno stato di conversazione tra una richiesta
    # e l'altra, quindi è il client a doverla rimandare ogni volta.
    storico: list[MessaggioIn] = []


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

    # Teniamo solo gli ultimi MAX_STORICO messaggi: mandare tutta la chat
    # ad ogni domanda farebbe crescere senza limite il costo (e la latenza)
    # di ogni richiesta man mano che la conversazione si allunga.
    storico = [m.model_dump() for m in payload.storico[-MAX_STORICO:]]

    # PRIMA di cercare nei documenti, riscriviamo una domanda ellittica
    # ("e quella giornata?") in una domanda autonoma che nomina esplicitamente
    # a cosa si riferisce: il retriever (vedi rag/vectorstore.py) non ha
    # nessuna nozione di conversazione, cerca solo le parole/il significato
    # della stringa che riceve.
    domanda = riformula_con_storico(llm, payload.domanda, storico)

    # `.invoke(...)` esegue il grafo dall'inizio alla fine e ritorna lo
    # stato finale come dizionario Python; ne leggiamo solo la chiave
    # "risposta" (le altre, come "documenti" e "tentativi", sono dettagli
    # interni del grafo che non servono a chi chiama l'API).
    testo_storico = "\n".join(f"{m['ruolo']}: {m['contenuto']}" for m in storico)
    risultato = grafo.invoke({"domanda": domanda, "tentativi": 0, "storico": testo_storico})
    return RispostaOut(risposta=risultato["risposta"])
