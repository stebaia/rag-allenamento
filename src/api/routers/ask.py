"""Endpoint di interrogazione del RAG, isolato per utente.

Questo è l'endpoint "finale" che mette insieme tutti gli altri pezzi:
autenticazione (UtenteCorrente), ricerca vettoriale filtrata per utente
(retriever_per_utente) e il grafo di ragionamento LangGraph già esistente
in rag/graph.py (che qui NON abbiamo toccato: lo riusiamo così com'era
nella versione a riga di comando del progetto).
"""

import uuid

from fastapi import APIRouter, status
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from pydantic import BaseModel

from rag.config import K
from rag.graph import costruisci_grafo, costruisci_grafo_agente
from rag.vectorstore import retriever_per_utente

from ..deps import UtenteCorrente
from ..state import get_checkpointer, get_embeddings, get_llm, get_store

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



# System prompt del grafo agentico. Diverso da _PROMPT: qui non riempiamo noi
# {context}: è l'LLM che decide se chiamare cerca_documenti e con che query,
# quindi il prompt descrive un COMPORTAMENTO invece di incorniciare dei dati.
_SYSTEM_AGENTE = (
    "Sei un assistente su dieta, spesa e allenamento. Oggi è {oggi}. "
    "Per rispondere usa il tool cerca_documenti: non inventare nulla e non "
    "rispondere a memoria sui contenuti dei documenti.\n\n"
    "Rispondi ESATTAMENTE a ciò che è stato chiesto:\n"
    "- domanda sui PASTI ('cosa mangio', 'cosa c'è a cena') → una ricerca sui "
    "pasti, e riporta gli alimenti con i grammi del piano alimentare;\n"
    "- domanda sulla SPESA ('cosa devo comprare') → riporta le quantità della "
    "lista della spesa (es. '1 confezione'), non i grammi dei pasti;\n"
    "- domanda che richiede ENTRAMBI ('cosa devo comprare per i pasti di "
    "lunedì') → fai DUE ricerche separate e incrocia i risultati.\n\n"
    "Non trasformare una domanda sui pasti in una risposta sulla spesa. Se "
    "l'informazione non c'è nei documenti, dillo. Per domande generiche sul "
    "tuo funzionamento rispondi direttamente, senza tool. Rispondi in "
    "italiano, conciso."
)


class DomandaIn(BaseModel):
    domanda: str
    # Identifica la conversazione. Il client non manda più i messaggi: manda
    # solo QUALE conversazione, e il checkpointer ricarica il resto.
    conversation_id: str | None = None
    # True → grafo con tool (l'LLM decide se e quante volte cercare),
    # False → grafo deterministico della Fase 1. Tenerli affiancati permette
    # di confrontare le due modalità sulla stessa domanda.
    agente: bool = False
    # Solo in modalità agente: sospende con interrupt() prima di scrivere una
    # memoria, invece di salvarla senza chiedere.
    conferma_memorie: bool = False


class RispostaOut(BaseModel):
    risposta: str
    conversation_id: str
    # Valorizzato quando il grafo si è sospeso in attesa di conferma: contiene
    # il payload di interrupt(). Il client riprende con POST /ask/resume.
    in_attesa: dict | None = None


class MessaggioOut(BaseModel):
    ruolo: str  # "user" oppure "assistant"
    contenuto: str


class ConversazioneOut(BaseModel):
    conversation_id: str
    messaggi: list[MessaggioOut]


def _risposta_da_stato(risultato: dict) -> str:
    """Estrae il testo dell'ultimo messaggio dell'assistente.

    Il grafo deterministico espone un campo "risposta"; quello agentico no —
    lì la risposta è semplicemente l'ultimo messaggio della lista.
    """
    if "risposta" in risultato:
        return risultato["risposta"]
    for m in reversed(risultato.get("messaggi", [])):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


@router.post("/ask", response_model=RispostaOut)
def ask(payload: DomandaIn, utente: UtenteCorrente):
    embeddings = get_embeddings()
    retriever = retriever_per_utente(embeddings, utente.id, K)
    llm = get_llm()

    # Se il client non manda un id, ne creiamo uno: è l'inizio di una nuova
    # conversazione. Lo restituiamo così il client sa cosa rimandare dopo.
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    # Il thread_id è namespaced sull'utente: senza il prefisso, un utente che
    # indovina l'id di una conversazione altrui potrebbe leggerla.
    config = {"configurable": {"thread_id": f"{utente.id}:{conversation_id}"}}

    if payload.agente:
        grafo = costruisci_grafo_agente(
            retriever,
            llm,
            _SYSTEM_AGENTE,
            checkpointer=get_checkpointer(),
            store=get_store(),
            user_id=utente.id,
            conferma_memorie=payload.conferma_memorie,
        )
        ingresso = {"messaggi": [HumanMessage(content=payload.domanda)]}
    else:
        grafo = costruisci_grafo(
            retriever, llm, _PROMPT, checkpointer=get_checkpointer()
        )
        ingresso = {"messaggi": [HumanMessage(content=payload.domanda)], "tentativi": 0}

    risultato = grafo.invoke(ingresso, config)

    # __interrupt__ compare quando un nodo ha chiamato interrupt(): il grafo è
    # congelato nel checkpointer e aspetta POST /ask/resume.
    if "__interrupt__" in risultato:
        return RispostaOut(
            risposta="",
            conversation_id=conversation_id,
            in_attesa=dict(risultato["__interrupt__"][0].value),
        )

    return RispostaOut(
        risposta=_risposta_da_stato(risultato), conversation_id=conversation_id
    )


class MemoriaOut(BaseModel):
    id: str
    fatto: str


@router.get("/memorie", response_model=list[MemoriaOut])
def elenca_memorie(utente: UtenteCorrente):
    """Rende ispezionabili le memorie a lungo termine.

    Senza questo, un fatto sbagliato salvato dall'agente resta invisibile e
    continua a inquinare le risposte (§3.3 Test D): non compare in nessuna
    conversazione perché non è uno storico, è uno store.
    """
    memorie = get_store().search(("memorie", utente.id), limit=100)
    return [MemoriaOut(id=m.key, fatto=m.value["fatto"]) for m in memorie]


@router.delete("/memorie/{memoria_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancella_memoria(memoria_id: str, utente: UtenteCorrente):
    """Cancella una memoria. Il namespace è chiuso sull'utente, quindi non si
    può cancellare quella di un altro nemmeno indovinandone l'id."""
    get_store().delete(("memorie", utente.id), memoria_id)


class RipresaIn(BaseModel):
    conversation_id: str
    # Il valore che interrupt() restituisce al nodo sospeso: "sì" per salvare.
    risposta: str


@router.post("/ask/resume", response_model=RispostaOut)
def ask_resume(payload: RipresaIn, utente: UtenteCorrente):
    """Riprende un grafo sospeso da interrupt().

    Non ricostruiamo l'input: `Command(resume=...)` riparte dal punto esatto in
    cui il nodo si era fermato, con lo stato ricaricato dal checkpointer.
    """
    embeddings = get_embeddings()
    retriever = retriever_per_utente(embeddings, utente.id, K)
    grafo = costruisci_grafo_agente(
        retriever,
        get_llm(),
        _SYSTEM_AGENTE,
        checkpointer=get_checkpointer(),
        store=get_store(),
        user_id=utente.id,
        conferma_memorie=True,
    )
    config = {
        "configurable": {"thread_id": f"{utente.id}:{payload.conversation_id}"}
    }
    risultato = grafo.invoke(Command(resume=payload.risposta), config)
    return RispostaOut(
        risposta=_risposta_da_stato(risultato),
        conversation_id=payload.conversation_id,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversazioneOut)
def leggi_conversazione(conversation_id: str, utente: UtenteCorrente):
    """Rilegge i messaggi di una conversazione dal checkpointer.

    Serve al frontend per ricostruire la chat dopo un refresh: l'id vive in
    localStorage, i messaggi no. Qui NON costruiamo il grafo — servirebbe un
    retriever, e quindi il modello di embedding, per una semplice lettura.
    Il checkpointer sa già rispondere da solo.
    """
    # Stesso namespace usato in ask(): se l'id è di un altro utente il
    # thread_id non esiste e la conversazione risulta semplicemente vuota.
    config = {"configurable": {"thread_id": f"{utente.id}:{conversation_id}"}}

    checkpoint = get_checkpointer().get(config)
    messaggi = (checkpoint or {}).get("channel_values", {}).get("messaggi", [])

    return ConversazioneOut(
        conversation_id=conversation_id,
        messaggi=[
            MessaggioOut(
                ruolo="user" if isinstance(m, HumanMessage) else "assistant",
                contenuto=m.content,
            )
            for m in messaggi
            if isinstance(m, (HumanMessage, AIMessage))
        ],
    )

