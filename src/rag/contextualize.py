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

import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

# Oltre questa lunghezza il documento non viene più allegato per intero: si
# manda una finestra di testo attorno al chunk. Con un documento da 425.000
# caratteri ogni chiamata costava ~79k token di input e ~22 secondi, cioè
# 2 ore e mezza per 400 chunk — e per collocare un chunk non serve l'opera
# completa, basta sapere in che sezione si trova.
_MAX_DOC_INTERO = 40_000

# Quanto testo prendere prima e dopo il chunk quando il documento è grande.
# L'intestazione del documento viene aggiunta a parte: contiene titolo e
# autore, che sono l'informazione più utile per la collocazione.
_FINESTRA = 6_000
_TESTA_DOCUMENTO = 1_500

# Chiamate OpenAI in parallelo. Sono indipendenti fra loro, quindi il collo di
# bottiglia è la latenza di rete. Il tetto vero non è la CPU ma il rate limit
# per token al minuto dell'account: con 8 thread e finestre da ~2k token si
# saturano i 200k TPM di gpt-4o-mini e l'API risponde 429.
_PARALLELE = 4

# Un 429 è temporaneo: senza retry basta una singola chiamata respinta per far
# fallire l'indicizzazione di un intero documento dopo minuti di lavoro.
_TENTATIVI = 5
_ATTESA_INIZIALE = 2.0

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

def _posizione_nel_documento(testo_completo: str, chunk: str) -> int:
    """Dove compare `chunk` nel documento originale, o -1 se non lo troviamo.

    Non basta cercare il prefisso del chunk: il chunking normalizza gli spazi
    e può anteporre righe di servizio (es. "1.2. Titolo (segue)") che
    nell'originale non esistono. Cerchiamo quindi sequenze di parole prese
    dal CORPO del chunk, saltando l'inizio, e confrontiamo su testo con gli
    spazi normalizzati da entrambe le parti.
    """
    parole = chunk.split()
    if len(parole) < 8:
        return -1

    # Il documento normalizzato una volta sola sarebbe più efficiente, ma qui
    # domina comunque la latenza di rete della chiamata LLM che segue.
    doc_norm = " ".join(testo_completo.split())
    scala = len(testo_completo) / max(len(doc_norm), 1)

    # Tre tentativi a profondità crescente: se il chunk inizia con righe
    # aggiunte da noi, più avanti si trova testo autentico.
    for salto in (2, 8, 16):
        if salto + 8 > len(parole):
            break
        ancora = " ".join(parole[salto : salto + 8])
        pos = doc_norm.find(ancora)
        if pos >= 0:
            # Riporta l'offset dal testo normalizzato a quello originale.
            return min(int(pos * scala), len(testo_completo) - 1)
    return -1


def _contesto_per_chunk(testo_completo: str, chunk: str) -> str:
    """Il testo da allegare al prompt per collocare `chunk`.

    Sotto la soglia si passa il documento intero, come prima: è la parte che
    OpenAI mette in cache fra una chiamata e l'altra, quindi costa poco.

    Sopra la soglia si passa l'intestazione del documento (titolo, autore,
    di cosa parla) più una finestra di testo attorno al punto in cui il chunk
    compare davvero. Per dire "questo pezzo sta nel capitolo VI, esecuzione
    immobiliare" non serve l'opera completa: serve sapere cosa lo circonda.
    """
    if len(testo_completo) <= _MAX_DOC_INTERO:
        return testo_completo

    pos = _posizione_nel_documento(testo_completo, chunk)
    if pos < 0:
        # Nessun ancoraggio trovato: meglio la sola testa del documento che
        # una finestra presa a caso, che collocherebbe il chunk nel posto
        # sbagliato — un contesto errato è peggio di uno generico.
        return testo_completo[:_TESTA_DOCUMENTO]

    inizio = max(0, pos - _FINESTRA // 2)
    fine = min(len(testo_completo), pos + _FINESTRA // 2)
    return (
        f"{testo_completo[:_TESTA_DOCUMENTO]}\n\n[...]\n\n"
        f"{testo_completo[inizio:fine]}"
    )


def contestualizza_chunk(llm, documento_completo: str, chunk: str) -> str:
    """Genera 1-2 frasi di contesto per un chunk e le antepone al suo testo.

    Ritenta con attesa crescente sugli errori temporanei (tipicamente il 429
    per rate limit). Esaurititi i tentativi restituisce il chunk originale:
    perdere il contesto di un chunk degrada un po' il retrieval, perdere
    l'intero documento lo azzera.
    """
    documento = _contesto_per_chunk(documento_completo, chunk)
    attesa = _ATTESA_INIZIALE
    for tentativo in range(_TENTATIVI):
        try:
            contesto = (
                (_PROMPT | llm)
                .invoke({"documento": documento, "chunk": chunk})
                .content.strip()
            )
            return f"{contesto}\n\n{chunk}"
        except Exception as e:
            if tentativo == _TENTATIVI - 1:
                print(f"   ↳ contesto non generato dopo {_TENTATIVI} tentativi: {e}")
                return chunk
            time.sleep(attesa)
            attesa *= 2
    return chunk


def contestualizza_documento(llm, testo_completo: str, chunks: list[dict]) -> list[dict]:
    """Applica contestualizza_chunk a tutti i chunk di un documento.

    Ritorna una NUOVA lista di dict con le stesse chiavi prodotte da
    chunk_documento ("testo", "fonte", "giorno", "reverse", "reverse_dal"),
    con solo "testo" sostituito dalla versione contestualizzata: `{**c, ...}`
    copia tutte le altre chiavi così come sono, quindi i metadati usati dal
    retriever (giorno/reverse) sopravvivono a questo passaggio.

    Le chiamate sono indipendenti e vengono eseguite in parallelo: in
    sequenza, un documento da 400 chunk richiedeva ore. `ThreadPoolExecutor`
    va bene nonostante il GIL perché il lavoro è attesa di rete, non CPU.
    `map` conserva l'ordine dei chunk, che il chunking ha reso significativo.
    """
    if not chunks:
        return []

    def contestualizza(c: dict) -> dict:
        return {**c, "testo": contestualizza_chunk(llm, testo_completo, c["testo"])}

    with ThreadPoolExecutor(max_workers=_PARALLELE) as pool:
        return list(pool.map(contestualizza, chunks))