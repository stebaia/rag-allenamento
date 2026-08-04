"""Componenti pesanti condivisi tra le richieste (caricati una sola volta).

HuggingFaceEmbeddings carica un modello di machine learning in memoria (dei
file da centinaia di MB): farlo ad ogni richiesta HTTP renderebbe l'API
lentissima. `@lru_cache(maxsize=1)` è un "decoratore" (una funzione che ne
avvolge un'altra per aggiungerle un comportamento) che memorizza il
risultato della prima chiamata e lo ritorna direttamente per tutte le
chiamate successive con gli stessi argomenti — qui le funzioni non hanno
argomenti, quindi in pratica "esegui il corpo solo alla primissima
chiamata, poi ritorna sempre lo stesso oggetto già creato".
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from rag.config import CHECKPOINT_DB, STORE_DB

from rag.config import MODELLO_CONTESTO, MODELLO_EMBED, MODELLO_LLM


@lru_cache(maxsize=1)
def get_llm_contesto() -> ChatOpenAI:
    """LLM economico dedicato a generare il contesto dei chunk (vedi rag/contextualize.py)."""
    return ChatOpenAI(model=MODELLO_CONTESTO, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Il modello che trasforma testo in vettori (vedi rag/vectorstore.py)."""
    return HuggingFaceEmbeddings(model_name=MODELLO_EMBED)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Il client verso il modello linguistico OpenAI usato per generare le risposte."""
    return ChatOpenAI(model=MODELLO_LLM, temperature=0)


@lru_cache(maxsize=1)
def get_store() -> SqliteStore:
    """Memoria a lungo termine, trasversale alle conversazioni.

    Il tutorial (§3.1) parte da InMemoryStore per imparare l'API, ma quello si
    perde a ogni riavvio: una memoria che dimentica non è una memoria. Qui
    usiamo SqliteStore — stessa interfaccia put/get/search/delete, su disco.
    Il file è separato da quello dei checkpoint: le conversazioni si possono
    buttare, le memorie no.
    """
    # isolation_level=None: SqliteStore emette da sé i BEGIN/COMMIT. Senza
    # questo, il modulo sqlite3 ne apre già una e si ottiene
    # "cannot start a transaction within a transaction".
    conn = sqlite3.connect(STORE_DB, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()  # crea le tabelle dello store se non esistono
    return store


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # crea le tabelle dei checkpoint se non esistono
    return saver
