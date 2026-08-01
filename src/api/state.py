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
