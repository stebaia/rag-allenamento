"""Reranking con cross-encoder: riordina i candidati leggendo domanda e chunk insieme.

Perché serve, dato che il RetrieverIbrido già riordina. Il punteggio ibrido
somma due numeri calcolati SEPARATAMENTE: la similarità coseno tra il vettore
della domanda e quello del chunk (due embedding prodotti in momenti diversi,
che non si sono mai "visti") e la sovrapposizione lessicale. Un cross-encoder
fa un'altra cosa: mette domanda e chunk nello stesso input e li processa in un
unico passaggio, quindi può cogliere che un chunk risponde alla domanda anche
quando non ne condivide le parole né una somiglianza semantica generica.

Il prezzo è che va eseguito su OGNI coppia (domanda, chunk): non esiste un
indice da precalcolare come per gli embedding. Per questo il reranking arriva
sempre in seconda battuta, su un pool già ristretto dal retrieval — qui i ~40
candidati del RetrieverIbrido, non l'intera collection.

Il modello di default (`BAAI/bge-reranker-v2-m3`) è multilingue e regge bene
l'italiano tecnico, ma pesa ~2,3 GB e costa 1-3 s per domanda su CPU: è il
motivo per cui RERANKER è spento di default (vedi rag/config.py).
"""

from functools import lru_cache

from rag.config import MODELLO_RERANKER


@lru_cache(maxsize=1)
def get_reranker():
    """Carica il cross-encoder una volta sola per processo.

    Stessa logica di `get_embeddings` in api/state.py: il caricamento da disco
    costa parecchi secondi, e con @lru_cache lo si paga alla prima chiamata
    invece che a ogni domanda. L'import di sentence_transformers sta dentro la
    funzione, non in cima al modulo: con RERANKER=off questa funzione non viene
    mai chiamata, e importare la libreria (che tira dentro torch) allungherebbe
    l'avvio del server per una cosa che non si userà.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODELLO_RERANKER)


def riordina(query: str, candidati: list, testo_di, top_n: int | None = None) -> list:
    """Riordina `candidati` per rilevanza rispetto a `query`, dal più al meno.

    `testo_di` è una funzione che estrae il testo da un candidato: così questa
    funzione non deve sapere se sta ricevendo punti Qdrant, Document di
    LangChain o dizionari, e resta usabile da entrambi.

    Se il modello non è disponibile (non scaricato, senza rete al primo avvio,
    memoria insufficiente) la funzione NON solleva: ritorna i candidati
    nell'ordine in cui li ha ricevuti. Il reranking è un miglioramento del
    ranking, non un prerequisito per rispondere — un errore qui deve degradare
    al comportamento con RERANKER=off, non far fallire la domanda dell'utente.
    """
    if len(candidati) < 2:
        # Con zero o un candidato non c'è nulla da riordinare, e caricare 2,3 GB
        # di modello per scoprirlo sarebbe uno spreco.
        return candidati

    try:
        modello = get_reranker()
        coppie = [(query, testo_di(c)) for c in candidati]
        punteggi = modello.predict(coppie)
    except Exception as exc:  # noqa: BLE001 - vedi docstring: si degrada, non si fallisce
        print(f"   ↳ reranker non disponibile ({exc}): tengo l'ordine del ranking ibrido")
        return candidati

    # zip(...) accoppia ogni candidato al suo punteggio; ordiniamo per
    # punteggio decrescente e teniamo solo i candidati, buttando i punteggi.
    ordinati = [
        c for c, _ in sorted(zip(candidati, punteggi), key=lambda coppia: coppia[1], reverse=True)
    ]
    return ordinati[:top_n] if top_n else ordinati
