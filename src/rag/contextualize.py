"""Contextual Retrieval: arricchisce ogni chunk con una breve
collocazione nel documento, prima di calcolarne l'embedding.

Idea (Anthropic, "Introducing Contextual Retrieval", 2024): un chunk
isolato può perdere informazioni che nel documento originale erano
implicite nel contesto circostante (a quale piano/periodo appartiene,
ecc.). Chiediamo a un LLM economico di scrivere 1-2 frasi che collocano
il chunk nel documento intero, e le anteponiamo al testo del chunk PRIMA
di embedderlo. Il documento intero viene passato come parte "cacheable"
del prompt (vedi CONTEXTUAL_RETRIEVAL.md) per tenere i costi bassi.
"""

from langchain_core.prompts import ChatPromptTemplate

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sei un assistente che colloca brevi estratti (chunk) nel loro "
            "documento di origine, per aiutare un sistema di ricerca a "
            "recuperarli correttamente.",
        ),
        (
            "human",
            "<documento>\n{documento}\n</documento>\n\n"
            "Ecco il chunk che vogliamo collocare nel documento sopra:\n"
            "<chunk>\n{chunk}\n</chunk>\n\n"
            "Scrivi 1-2 frasi brevi e concrete (in italiano) che collocano "
            "questo chunk nel documento: di quale piano/scheda fa parte, a "
            "quale periodo o sezione appartiene. Non riassumere il "
            "contenuto del chunk stesso, aggiungi solo il contesto che gli "
            "manca. Rispondi SOLO con quelle frasi, senza introduzioni.",
        ),
    ]
)

def contestualizza_chunk(llm, documento_completo: str, chunk: str) -> str:
    """Genera 1-2 frasi di contesto per un chunk e le antepone al suo testo."""
    contesto = (_PROMPT | llm).invoke(
        {"documento": documento_completo, "chunk": chunk}
    ).content.strip()
    return f"{contesto}\n\n{chunk}"


def contestualizza_documento(llm, testo_completo: str, chunks: list[dict]) -> list[dict]:
    """Applica contestualizza_chunk a tutti i chunk di un documento.

    Ritorna una NUOVA lista di dict (stessa struttura di chunk_documento:
    {"testo", "fonte", "giorno"}), con "testo" sostituito dalla versione
    contestualizzata. Il documento completo viene passato identico ad ogni
    chiamata: è la parte che l'API di OpenAI mette in cache automaticamente
    quando supera una certa lunghezza, rendendo le chiamate 2, 3, ... N
    molto più economiche della prima.
    """
    return [
        {**c, "testo": contestualizza_chunk(llm, testo_completo, c["testo"])}
        for c in chunks
    ]