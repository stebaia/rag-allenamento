"""Vector store Qdrant: indicizzazione e retrieval isolati per utente/documento.

Concetti chiave di questo file:

- Un "embedding" è un vettore di numeri (es. 384 numeri in fila) che
  rappresenta il SIGNIFICATO di un testo. Testi con significato simile
  hanno vettori "vicini" tra loro (in senso geometrico). Qdrant è un
  database pensato apposta per salvare vettori e rispondere velocemente a
  domande tipo "quali sono i 5 vettori più vicini a questo?".

- "RAG" (Retrieval-Augmented Generation) funziona così: trasformiamo la
  domanda dell'utente in un embedding, cerchiamo i chunk di testo più
  vicini (cioè più pertinenti) nel database vettoriale, e li passiamo
  all'LLM come contesto per rispondere.

- Isolamento multi-utente: tutti gli utenti condividono la stessa
  collection Qdrant, ma ogni chunk porta con sé un "metadato" user_id.
  Ogni ricerca filtra SEMPRE per user_id, così un utente non vede mai i
  documenti di un altro (vedi retriever_per_utente più sotto).
"""

import math
import re
from collections import Counter
from datetime import date

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .chunking import _GIORNO, _norm_giorno
from .config import QDRANT_COLLECTION, QDRANT_URL

_REVERSE_QUERY = re.compile(r"(?i)reverse\s*(\d+)")

# Domande che chiedono la LISTA DELLA SPESA. Hanno un problema specifico: le
# parole della domanda ("spesa", "comprare") non compaiono nel testo dei
# chunk dei singoli alimenti ("Fette biscottate — 12 fette"), quindi né la
# ricerca lessicale né quella semantica li fanno emergere in modo affidabile
# — soprattutto quando la domanda nomina anche un giorno e il boost sul
# giorno riempie il contesto con la dieta. Riconosciamo l'intento e
# recuperiamo esplicitamente i chunk della spesa, come già facciamo per i
# giorni della settimana.
_SPESA_QUERY = re.compile(r"(?i)\b(spesa|spese|comprare|compro|acquistare|acquisto|supermercato|carrello)\b")

# Variabile "a livello di modulo": vive finché vive il processo Python, ed
# è condivisa da tutte le richieste HTTP gestite da questo processo. La
# usiamo per non aprire una nuova connessione di rete a Qdrant ad ogni
# chiamata, ma riutilizzare sempre la stessa (pattern "singleton").
_client: QdrantClient | None = None


def ottieni_client() -> QdrantClient:
    """Ritorna il client Qdrant, creandolo alla prima chiamata (lazy init)."""
    # `global _client` dice a Python: quando scrivo `_client = ...` qui
    # dentro, sto modificando la variabile del modulo definita sopra, non
    # creandone una nuova locale alla funzione.
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def assicura_collection(embeddings) -> None:
    """Crea la collection Qdrant se non esiste ancora, con la dimensione giusta per l'embedding in uso."""
    client = ottieni_client()
    if client.collection_exists(QDRANT_COLLECTION):
        return
    # Modelli di embedding diversi producono vettori di lunghezza diversa
    # (es. 384 numeri per questo modello, 1536 per un altro). Qdrant vuole
    # sapere in anticipo quanti numeri avrà ogni vettore, quindi lo
    # scopriamo "al volo" incorporando una stringa di prova e misurandone
    # la lunghezza, invece di scriverlo a mano e rischiare di sbagliare
    # se un giorno cambiamo modello.
    dimensione = len(embeddings.embed_query("dimensiona"))
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        # COSINE è il modo di misurare "quanto sono vicini due vettori"
        # più comune per gli embedding testuali.
        vectors_config=qmodels.VectorParams(size=dimensione, distance=qmodels.Distance.COSINE),
    )


def ottieni_vectorstore(embeddings) -> QdrantVectorStore:
    """Ritorna l'oggetto LangChain che sa parlare con la nostra collection Qdrant."""
    assicura_collection(embeddings)
    return QdrantVectorStore(
        client=ottieni_client(),
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


def upsert_documento(embeddings, user_id: str, document_id: str, chunks: list[dict]) -> int:
    """Embedda e inserisce i chunk di un documento, taggati con user_id/document_id. Ritorna il numero di chunk inseriti."""
    vectorstore = ottieni_vectorstore(embeddings)
    # Questa è una "list comprehension": un modo compatto di scrivere un
    # ciclo for che costruisce una nuova lista. È equivalente a:
    #   docs = []
    #   for c in chunks:
    #       docs.append(Document(...))
    # Ogni `Document` è l'oggetto standard di LangChain: un testo
    # (page_content) più dei metadati liberi associati (un dizionario).
    docs = [
        Document(
            page_content=c["testo"],
            metadata={
                "user_id": user_id,
                "document_id": document_id,
                "fonte": c["fonte"],
                "giorno": c.get("giorno", ""),
                "reverse": c.get("reverse", ""),
                "reverse_dal": c.get("reverse_dal", ""),
                "tipo": c.get("tipo", ""),
            },
        )
        for c in chunks
    ]
    if docs:
        # add_documents calcola l'embedding di ogni testo e lo salva su
        # Qdrant insieme ai metadati. È l'operazione "costosa" (chiama il
        # modello di embedding), per questo viene fatta in background
        # (vedi api/routers/documents.py) invece che dentro la richiesta HTTP.
        vectorstore.add_documents(docs)
    return len(docs)


def elimina_documento(document_id: str) -> None:
    """Rimuove da Qdrant tutti i chunk appartenenti a un documento."""
    ottieni_client().delete(
        collection_name=QDRANT_COLLECTION,
        # Un "filtro" Qdrant è una condizione su cui i punti (=i vettori
        # salvati) devono corrispondere per essere selezionati. Qui diciamo:
        # "seleziona tutti i punti il cui metadato document_id è uguale a
        # questo". `must=[...]` significa "tutte queste condizioni devono
        # essere vere" (equivalente a un AND logico).
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="metadata.document_id", match=qmodels.MatchValue(value=document_id))]
            )
        ),
    )


def _tokenizza(testo: str) -> set[str]:
    """Estrae le parole "significative" (più di 2 lettere) da un testo, in minuscolo.

    Stessa idea usata in rag.py (Indice._tok): serve per calcolare quanto
    la domanda e un chunk condividono le stesse parole chiave, indipendentemente
    dal loro significato semantico (che invece cattura l'embedding).
    """
    return {w for w in re.findall(r"\w+", testo.lower()) if len(w) > 2}


class RetrieverIbrido(BaseRetriever):
    """Retriever che replica il comportamento di Indice.cerca() in rag.py:

    1. Similarità SEMANTICA (embedding) su un pool ampio di candidati.
    2. Punteggio LESSICALE (TF-IDF approssimato sul pool) sommato a quello
       semantico, per premiare i chunk che condividono le parole esatte
       della domanda (utile per numeri, nomi di alimenti/esercizi, ecc.
       che l'embedding da solo potrebbe non pesare abbastanza).
    3. Se la domanda cita un giorno della settimana, TUTTI i chunk di
       quel giorno vengono inclusi in testa ai risultati, anche quando il
       loro punteggio semantico non li farebbe rientrare nei primi `k`.

    `BaseRetriever` è la classe base di LangChain per "qualunque oggetto
    che sappia rispondere a: data una domanda, dammi dei Document
    pertinenti". Ereditandola, questo retriever si comporta esattamente
    come quello standard di Qdrant agli occhi di graph.py, che continua a
    chiamare semplicemente `retriever.invoke(domanda)`.
    """

    # BaseRetriever è basato su Pydantic: gli attributi "normali" (come
    # self.client sotto) andrebbero dichiarati come campi del modello.
    # PrivateAttr li tiene invece fuori dalla validazione Pydantic, giusto
    # perché qui sono solo dipendenze interne, non dati di configurazione
    # che vogliamo esporre/validare come schema.
    _client: QdrantClient = PrivateAttr()
    _embeddings: object = PrivateAttr()
    _user_id: str = PrivateAttr()
    _k: int = PrivateAttr()
    _pool: int = PrivateAttr()
    _peso_lessicale: float = PrivateAttr()
    _max_boost: int = PrivateAttr()
    _min_altri: int = PrivateAttr()

    def __init__(
        self,
        client,
        embeddings,
        user_id: str,
        k: int,
        pool: int = 40,
        peso_lessicale: float = 0.4,
        max_boost: int = 8,
        min_altri: int = 4,
    ):
        super().__init__()
        self._client = client
        self._embeddings = embeddings
        self._user_id = user_id
        self._k = k
        self._pool = pool  # quanti candidati semantici considerare prima del rescoring
        self._peso_lessicale = peso_lessicale
        # Tetto ai chunk del boost-giorno: una giornata alimentare sono ~6
        # chunk, ma con due versioni dello stesso piano indicizzate
        # diventano 12+ e riempirebbero da sole tutto il contesto.
        self._max_boost = max_boost
        # Posti garantiti ai migliori del ranking ibrido, per le domande che
        # incrociano più documenti (vedi _unisci_giorno_e_ranking).
        self._min_altri = min_altri

    def _filtro_utente(self, extra: list | None = None) -> qmodels.Filter:
        condizioni = [qmodels.FieldCondition(key="metadata.user_id", match=qmodels.MatchValue(value=self._user_id))]
        if extra:
            condizioni += extra
        return qmodels.Filter(must=condizioni)

    def _reverse_da_privilegiare(self, query: str, candidati) -> str | None:
        """Determina quale settimana "Reverse N" usare per il boost sul giorno.

        Un piano con più settimane (es. CONTEXTUAL_RETRIEVAL.md, "Piano
        Reverse") ripete gli stessi nomi di giorno ("Lunedì") in ogni
        settimana con macro diverse. Senza restringere il boost a UNA
        settimana, una domanda come "cosa mangio lunedì?" recupererebbe tutti
        i "lunedì" di tutte le settimane mescolati. Priorità:
        1. la domanda nomina esplicitamente "Reverse N" -> usa quello;
        2. altrimenti, tra i "reverse" visti nei candidati di questo utente,
           usa quello con la data di inizio (reverse_dal) più recente non
           successiva ad oggi (la settimana "in corso");
        3. se oggi è PRIMA dell'inizio di ogni settimana nota (piano
           caricato in anticipo, vedi es. CONTEXTUAL_RETRIEVAL.md), usa la
           prima che deve ancora iniziare, così l'utente non resta senza
           risposta solo perché la sua dieta non è ancora partita;
        4. se nessun candidato ha metadati "reverse" (piano a settimana
           singola, come nel resto del progetto), ritorna None e il boost
           si comporta come prima.
        """
        match_esplicito = _REVERSE_QUERY.search(query)
        if match_esplicito:
            return match_esplicito.group(1)

        oggi = date.today().isoformat()
        migliori: dict[str, str] = {}  # numero reverse -> la sua data "dal"
        for c in candidati:
            meta = c.payload.get("metadata", {})
            numero, dal = meta.get("reverse"), meta.get("reverse_dal")
            if numero and dal:
                migliori[numero] = dal

        if not migliori:
            return None

        gia_iniziati = {n: d for n, d in migliori.items() if d <= oggi}
        if gia_iniziati:
            return max(gia_iniziati, key=gia_iniziati.get)
        return min(migliori, key=migliori.get)

    def _chunk_del_giorno(self, condizioni: list) -> list:
        """Tutti i chunk dell'utente che soddisfano `condizioni` (max 100).

        Usa `scroll` e non `query_points` perché qui non ci interessa
        l'ordine per similarità: vogliamo TUTTI i chunk del giorno citato,
        non i più vicini alla domanda. `scroll` ritorna la coppia
        (punti, offset_successivo): a noi basta il primo elemento.
        """
        return list(
            self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=self._filtro_utente(extra=condizioni),
                limit=100,
                with_payload=True,
            )[0]
        )

    def _unisci_giorno_e_ranking(self, chunk_giorno: list, chunk_spesa: list, altri: list) -> list:
        """Fonde i chunk "richiesti esplicitamente" con i migliori del ranking.

        I chunk del giorno e quelli della spesa vanno in testa perché
        rispondono a una parte esplicita della domanda (il giorno nominato, la
        richiesta di fare la spesa). Di ognuno ne teniamo al massimo
        `_max_boost`, e garantiamo comunque `_min_altri` posti ai migliori del
        ranking ibrido: senza queste quote una domanda che incrocia due
        documenti ("che spesa per la dieta di oggi?") riceverebbe solo il
        primo dei due, perché i ~12 chunk del giorno riempirebbero da soli
        tutto il contesto.
        """
        prioritari = chunk_giorno[: self._max_boost] + chunk_spesa[: self._max_boost]
        if not prioritari:
            return altri[: self._k]

        # Il contesto è grande almeno quanto i chunk prioritari più la quota
        # riservata, così il boost non toglie mai posti al ranking ibrido.
        k_effettivo = max(self._k, len(prioritari) + self._min_altri)
        return (prioritari + altri)[:k_effettivo]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Punto d'ingresso richiesto da BaseRetriever: LangChain lo chiama
        internamente quando qualcuno invoca `retriever.invoke(query)`."""
        vettore_query = self._embeddings.embed_query(query)

        # 1) Pool ampio di candidati per similarità semantica (coseno).
        candidati = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vettore_query,
            query_filter=self._filtro_utente(),
            limit=self._pool,
            with_payload=True,
        ).points

        if not candidati:
            return []

        # 2) Punteggio lessicale: quante parole della domanda compaiono nel
        # chunk, pesate per "quanto sono rare" nel pool (idea IDF: una
        # parola che compare in pochi chunk è più discriminante di una che
        # compare ovunque).
        termini_query = _tokenizza(query)
        testi = [c.payload.get("page_content", "") for c in candidati]
        token_per_chunk = [_tokenizza(t) for t in testi]

        n_doc = len(candidati)
        doc_freq = Counter()
        for token in token_per_chunk:
            doc_freq.update(termini_query & token)  # solo le parole che ci interessano

        idf = {w: math.log((n_doc + 1) / (doc_freq.get(w, 0) + 1)) + 1 for w in termini_query}

        punteggi_lex = [sum(idf[w] for w in termini_query if w in token_per_chunk[i]) for i in range(n_doc)]
        max_lex = max(punteggi_lex) or 1.0

        risultati = []
        for i, punto in enumerate(candidati):
            punteggio_sem = punto.score  # similarità coseno, già in [0, 1] circa
            punteggio_lex = punteggi_lex[i] / max_lex
            punteggio_finale = punteggio_sem + self._peso_lessicale * punteggio_lex
            risultati.append((punto, punteggio_finale))

        # Ordina dal punteggio ibrido più alto al più basso.
        risultati.sort(key=lambda coppia: coppia[1], reverse=True)

        # 3) Se la domanda nomina un giorno, recupera TUTTI i chunk di
        # quel giorno per l'utente (query separata, senza limite di pool),
        # e li mette in testa. Replica esattamente l'euristica di rag.py.
        chunk_giorno = []
        match_giorno = _GIORNO.search(query.lower())
        if match_giorno:
            giorno = _norm_giorno(match_giorno.group(0))
            condizione_giorno = qmodels.FieldCondition(
                key="metadata.giorno", match=qmodels.MatchValue(value=giorno)
            )

            # Se il documento ha più settimane "Reverse N" (vedi
            # rag/chunking.py), restringiamo il boost alla settimana giusta:
            # altrimenti "lunedì" recupererebbe i "lunedì" di TUTTE le
            # settimane insieme, mescolando macro/alimenti diversi.
            reverse_target = self._reverse_da_privilegiare(query, candidati)
            condizioni_extra = [condizione_giorno]
            if reverse_target:
                condizioni_extra.append(
                    qmodels.FieldCondition(key="metadata.reverse", match=qmodels.MatchValue(value=reverse_target))
                )

            chunk_giorno = self._chunk_del_giorno(condizioni_extra)

            # Fallback: se il filtro sul reverse non trova nulla (es. il
            # giorno citato non esiste in quella settimana specifica, o i
            # dati sono incompleti), ripetiamo la ricerca sul solo giorno
            # invece di lasciare l'utente senza risposta.
            if not chunk_giorno and reverse_target:
                chunk_giorno = self._chunk_del_giorno([condizione_giorno])

        # 4) Se la domanda chiede la spesa, recupera esplicitamente i chunk
        # della lista della spesa. Senza questo, una domanda che INCROCIA i
        # due documenti ("che spesa devo fare per la dieta di oggi?") riceve
        # solo la dieta: le parole "spesa"/"comprare" non compaiono nel testo
        # dei singoli alimenti, quindi il ranking ibrido non li fa emergere,
        # e il boost sul giorno occupa il resto del contesto.
        chunk_spesa = []
        if _SPESA_QUERY.search(query):
            chunk_spesa = self._chunk_del_giorno(
                [qmodels.FieldCondition(key="metadata.tipo", match=qmodels.MatchValue(value="spesa"))]
            )
            # Ordinati per pertinenza, non nell'ordine in cui stanno nel PDF:
            # ne teniamo solo `_max_boost`, e i primi del PDF (reparto
            # "Dispensa") non sono quasi mai quelli che servono. Se la domanda
            # nomina anche un giorno, la pertinenza è la sovrapposizione con
            # gli alimenti dei pasti di quel giorno, così "che spesa per la
            # dieta di lunedì?" fa emergere patate e pollo (nella cena di
            # lunedì) invece dei primi alimenti in elenco.
            riferimento = " ".join(p.payload.get("page_content", "") for p in chunk_giorno) or query
            termini_riferimento = _tokenizza(riferimento)
            chunk_spesa.sort(
                key=lambda p: len(_tokenizza(p.payload.get("page_content", "")) & termini_riferimento),
                reverse=True,
            )

        id_prioritari = {p.id for p in chunk_giorno} | {p.id for p in chunk_spesa}
        altri = [p for p, _ in risultati if p.id not in id_prioritari]

        # Il boost sul giorno può essere MOLTO grande: una giornata alimentare
        # sono ~6 chunk, e se l'utente ha indicizzato più versioni dello
        # stesso piano si moltiplicano (12 chunk per "lunedì" con due piani).
        # Prendendo solo `(chunk_giorno + altri)[:k]` quei chunk occuperebbero
        # tutti i posti tranne due, e una domanda che INCROCIA due documenti
        # ("che spesa devo fare per la dieta di oggi?") si ritroverebbe senza
        # i chunk del secondo documento: sono pertinenti per il ranking
        # ibrido, ma non hanno il metadato `giorno` e finiscono in coda.
        # Riserviamo quindi una quota fissa ai migliori del ranking ibrido,
        # indipendentemente da quanto è grande il boost.
        scelti = self._unisci_giorno_e_ranking(chunk_giorno, chunk_spesa, altri)

        return [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {}),
            )
            for p in scelti
        ]


def retriever_per_utente(embeddings, user_id: str, k: int) -> RetrieverIbrido:
    """Retriever filtrato sui soli documenti dell'utente, con boost su
    corrispondenze lessicali esatte e sui chunk del giorno citato nella
    domanda (vedi RetrieverIbrido)."""
    assicura_collection(embeddings)
    return RetrieverIbrido(client=ottieni_client(), embeddings=embeddings, user_id=user_id, k=k)
