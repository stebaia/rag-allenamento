# Tutorial: Contextual Retrieval nel nostro RAG

Questo documento spiega cos'è il **Contextual Retrieval** proposto da
Anthropic, perché serve, e ti guida passo passo nell'implementarlo in
questo progetto. È pensato come continuazione di `TUTORIAL.md`: se non
l'hai ancora letto (in particolare le sezioni 3, 5 e 6 su architettura,
upload e flusso di `/ask`), fallo prima — qui do per scontato che tu
sappia cosa sono chunk, embedding, retriever e vector store nel nostro
codice.

---

## 1. Il problema che risolve

Quando `chunk_documento` (in `src/rag/chunking.py`) spezza un documento,
ogni pezzo perde il contesto del documento intero. Guarda un chunk reale
prodotto oggi dal nostro chunking delle sedute di allenamento:

```
Sessione 2 — Upper Body, allenamento: Panca piana con bilanciere — 4x8, recupero 90", focus Petto
```

Isolato da tutto il resto, è già abbastanza chiaro — il nostro chunking
è già "su misura" e mantiene molte informazioni nel testo stesso (numero
sessione, nome, tipo tabella). Ma pensa a un chunk della dieta:

```
Lunedì, pranzo: 80g pasta, petto di pollo, verdura (macro: 80g carboidrati, 45g proteine, 10g grassi, 620 kcal)
```

Se un domani il documento avesse **più settimane** o **più piani dieta**
(es. "Piano dieta massa" vs "Piano dieta definizione"), questo chunk da
solo non direbbe più a quale piano appartiene: "lunedì" e "pranzo" non
bastano a distinguerlo. Il problema generale è questo:

> Il chunk contiene abbastanza informazione per essere **trovato** dalla
> ricerca (embedding + lessicale), ma non necessariamente abbastanza per
> essere **capito correttamente fuori contesto** una volta recuperato?

Anthropic ha pubblicato un articolo ("Introducing Contextual Retrieval",
2024) che descrive questo problema su documenti generici (es. un
10-K finanziario, dove un chunk può contenere solo "il fatturato è
cresciuto del 3%" senza dire di quale azienda o trimestre si parla) e
propone una soluzione semplice: **prima di calcolare l'embedding di ogni
chunk, si chiede a un LLM economico di scrivere 1-2 frasi che lo
collocano nel documento intero**, e quelle frasi vengono anteposte al
testo del chunk.

Esempio (dal loro articolo, tradotto):

```
Chunk originale:
"Il fatturato dell'azienda è cresciuto del 3% rispetto al trimestre precedente."

Chunk contestualizzato:
"Questo chunk proviene dal 10-K di ACME Corp relativo al Q2 2023;
il fatturato del trimestre precedente era di 314 milioni di dollari.
Il fatturato dell'azienda è cresciuto del 3% rispetto al trimestre precedente."
```

L'embedding calcolato sul chunk contestualizzato è "più informato": una
query come "fatturato ACME Q2 2023" avrà una similarità più alta con
questo chunk che con la versione senza contesto, perché il vettore ora
codifica anche l'azienda e il trimestre.

### Perché è utile *in questo progetto specifico*

Il nostro chunking (sedute → per esercizio, dieta → per pasto) produce
già chunk piuttosto "densi" di informazione locale, quindi il guadagno
qui è più limitato che nell'esempio finanziario di Anthropic. Ma resta
utile per due motivi concreti nel nostro dominio:

1. **Multi-documento per utente**: un utente può caricare più file (es.
   una scheda vecchia e una nuova, o dieta + allenamento). Un chunk come
   "Sessione 1 — Upper Body: ..." non dice da quale file/periodo proviene
   se un giorno avessimo due schede "Sessione 1" diverse in vigore.
2. **Le righe di chiusura/nota**, tipo `"Lunedì — Totale: 2100 kcal"`
   (il ramo `else` di `split_pasti` in `chunking.py`, quando la riga non
   fa match con `_MEAL`), sono poco informative da sole: non dicono
   nemmeno che sono un pasto o cosa contengono.

Contextual Retrieval non sostituisce il nostro `RetrieverIbrido` (che
resta l'ibrido semantico+lessicale+boost giorno) — lo **precede**: agisce
al momento dell'**indicizzazione**, arricchendo il testo che viene
embeddato, non al momento della ricerca.

### Perché non "semplicemente mettere più testo in ogni chunk"?

Potresti chiederti: perché non allargare i chunk invece di aggiungere
contesto generato da un LLM? Perché sono due problemi diversi:
- Chunk più grandi diluiscono l'embedding (mescolano più informazioni,
  peggiorando la precisione della ricerca) e riempiono la finestra di
  contesto del LLM finale con più testo irrilevante.
- Il contesto generato è **mirato**: poche parole che dicono "a cosa si
  riferisce" questo pezzo, non tutto il documento circostante.

### Il costo: perché serve il prompt caching

Generare il contesto per ogni chunk richiede una chiamata LLM che riceve
in input **l'intero documento** (per poter scrivere un riassunto
posizionale accurato) più il singolo chunk. Se un documento ha 50 chunk,
è come mandare il documento intero 50 volte. Anthropic risolve il costo
con il **prompt caching**: il documento (la parte grande e ripetuta) va
nella parte "cacheable" del prompt, e l'API fa pagare per intero solo la
prima chiamata; le successive (stesso documento, chunk diverso) pagano
una frazione del costo per la parte cachata. Nella nostra implementazione
useremo OpenAI (già usato nel progetto per LLM e generazione), che ha un
meccanismo di caching automatico simile (non richiede parametri
espliciti, si attiva da solo su prompt lunghi e ripetuti) — lo vedremo
al punto 4.

---

## 2. Dove si inserisce nella nostra pipeline

Guarda di nuovo il flusso di upload (sezione 5 di `TUTORIAL.md`):

```
POST /documents (file)
   │
   ▼
1. Validazione estensione
2. Salvataggio su disco
3. Riga DB "processing"
4. Risposta immediata
5. [BACKGROUND]
   a. carica_file        → estrae testo grezzo
   b. chunk_documento     → spezza in chunk
   c. upsert_documento    → embedda + salva su Qdrant
   d. stato → "ready"
```

Contextual Retrieval si inserisce **tra (b) e (c)**, come nuovo passo
`c`:

```
   a. carica_file        → estrae testo grezzo
   b. chunk_documento     → spezza in chunk
   b.5 contestualizza_chunk  → [NUOVO] antepone 1-2 frasi di contesto a ogni chunk (LLM)
   c. upsert_documento    → embedda + salva su Qdrant (il TESTO CONTESTUALIZZATO)
   d. stato → "ready"
```

Punto chiave: **il testo contestualizzato è quello che viene sia
embeddato sia mostrato all'LLM finale** (`nodo_genera` in `graph.py`
concatena `d.page_content` di ogni documento recuperato). Questo è
corretto e voluto: il contesto extra aiuta anche il modello che genera
la risposta finale a capire da dove viene l'informazione.

---

## 3. Implementazione, passo per passo

### Passo 1 — un nuovo modulo `contextualize.py`

Creiamo `src/rag/contextualize.py`, accanto a `chunking.py`, per tenere
questa logica separata (stesso principio di separazione già seguito nel
progetto: un file, una responsabilità).

```python
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
```

### Passo 2 — un LLM economico dedicato

Non vogliamo usare lo stesso `gpt-4o-mini` con `temperature=0` pensato
per rispondere all'utente finale (in `src/api/state.py`) per un compito
diverso: qui va bene un modello ancora più economico, dato che dobbiamo
chiamarlo una volta per ogni chunk. Aggiungiamo una costante e una
funzione cache dedicate.

In `src/rag/config.py`, accanto a `MODELLO_LLM`:

```python
MODELLO_CONTESTO = os.environ.get("MODELLO_CONTESTO", "gpt-4o-mini")
```

In `src/api/state.py`:

```python
from rag.config import MODELLO_CONTESTO, MODELLO_EMBED, MODELLO_LLM

@lru_cache(maxsize=1)
def get_llm_contesto() -> ChatOpenAI:
    """LLM economico dedicato a generare il contesto dei chunk (vedi rag/contextualize.py)."""
    return ChatOpenAI(model=MODELLO_CONTESTO, temperature=0)
```

Usiamo `gpt-4o-mini` come default (lo stesso già in uso, quindi zero
sorprese sui costi rispetto a oggi) ma la rendiamo configurabile: se in
futuro vuoi risparmiare ulteriormente puoi impostare
`MODELLO_CONTESTO=gpt-4o-mini` esplicitamente o un modello ancora più
piccolo via variabile d'ambiente, senza toccare il codice.

### Passo 3 — agganciarlo alla pipeline di indicizzazione

In `src/api/routers/documents.py`, dentro `_indicizza_documento`,
aggiungiamo il passo tra il chunking e l'upsert:

```python
from rag.contextualize import contestualizza_documento
from ..state import get_embeddings, get_llm_contesto

# ... dentro il try, dopo chunks = chunk_documento(doc):
chunks = chunk_documento(doc)
llm_contesto = get_llm_contesto()
chunks = contestualizza_documento(llm_contesto, doc["testo"], chunks)
embeddings = get_embeddings()
n = upsert_documento(embeddings, user_id, document_id, chunks)
```

Non serve toccare `vectorstore.py`: `upsert_documento` continua a
ricevere una lista di dict con la stessa struttura (`testo`, `fonte`,
`giorno`), semplicemente ora `testo` è già arricchito.

### Passo 4 — nulla da cambiare nel retrieval

Il punto più importante da capire: **`RetrieverIbrido` e `graph.py` non
cambiano di una riga**. Il contextual retrieval agisce solo a monte,
sull'indicizzazione. Dal punto di vista del retriever, i chunk in Qdrant
sono semplicemente "testo più lungo e più informativo" — la ricerca
semantica, quella lessicale (`_tokenizza`) e il boost per giorno
continuano a funzionare esattamente come prima, ma su testi con più
segnali utili dentro.

### Passo 5 (opzionale, consigliato) — non ricontestualizzare ad ogni riavvio

Un dettaglio pratico: la contestualizzazione avviene una volta per
documento, al momento dell'upload — non serve nessuna cache aggiuntiva
nel nostro codice, perché non ricalcoliamo mai i chunk di un documento
già `"ready"` (per rifarlo bisognerebbe eliminare e ricaricare il
documento, esattamente come già succede oggi se cambi il modello di
embedding — vedi le FAQ in `TUTORIAL.md`).

---

## 4. Le parti di codice spiegate

### `ChatPromptTemplate.from_messages([...])`

È l'oggetto LangChain che costruisce un prompt "a slot": invece di
concatenare stringhe a mano con f-string (come fa `graph.py` per i
prompt più semplici), definiamo dei placeholder `{documento}` e `{chunk}`
che vengono riempiti in automatico quando invochiamo il prompt con
`.invoke({"documento": ..., "chunk": ...})`. Il vantaggio rispetto a una
f-string diretta è che l'oggetto risultante si può comporre con `|`
(vedi sotto) e riusare più volte con input diversi.

I due elementi della lista sono messaggi con ruoli diversi (stessa idea
delle chat API di OpenAI/Anthropic):
- `"system"`: istruzioni generali su **come comportarsi** (qui: "colloca
  chunk nel documento").
- `"human"`: il contenuto specifico della richiesta, con i placeholder.

### `(_PROMPT | llm).invoke(...)`

L'operatore `|` (pipe) è la sintassi LCEL di LangChain ("LangChain
Expression Language") per comporre passi in sequenza: "prendi l'output
di `_PROMPT` e passalo come input a `llm`". È lo stesso pattern già
usato in `graph.py`:

```python
(prompt | llm | StrOutputParser()).invoke({...})
```

Qui ci fermiamo a `_PROMPT | llm` (senza `StrOutputParser()`) e leggiamo
`.content` a mano — funzionalmente equivalente, solo per mostrarti
l'alternativa; potresti scrivere identicamente `(_PROMPT | llm |
StrOutputParser()).invoke(...).strip()` se preferisci coerenza col resto
del progetto.

### Perché passare `documento_completo` intero ad ogni chiamata

```python
def contestualizza_documento(llm, testo_completo: str, chunks: list[dict]) -> list[dict]:
    return [
        {**c, "testo": contestualizza_chunk(llm, testo_completo, c["testo"])}
        for c in chunks
    ]
```

`{**c, "testo": ...}` è **unpacking di dizionario**: crea un nuovo
dizionario che ha tutte le coppie chiave/valore di `c` (quindi `fonte` e
`giorno` restano invariati), sovrascrivendo solo `"testo"` col nuovo
valore. È l'equivalente conciso di:

```python
nuovo = dict(c)
nuovo["testo"] = contestualizza_chunk(llm, testo_completo, c["testo"])
```

Il motivo per passare **lo stesso** `testo_completo` a ogni chiamata (una
per chunk) invece di, ad esempio, un riassunto già pronto, è il prompt
caching lato OpenAI: quando lo stesso prefisso di prompt (qui: le
istruzioni di sistema + il documento) si ripete identico su più
richieste consecutive e supera una soglia di lunghezza (1024 token per
i modelli GPT-4o), l'API rileva automaticamente il prefisso ripetuto e
fa pagare per intero solo la prima occorrenza; dalla seconda in poi
applica uno sconto sulla parte cachata. Non c'è nessun parametro da
passare esplicitamente (a differenza dell'API nativa Anthropic, dove il
caching va dichiarato con `cache_control`): con OpenAI/`langchain-openai`
è **automatico**, la nostra unica responsabilità è tenere quel prefisso
identico e chiamarlo in sequenza ravvicinata — esattamente quello che fa
il ciclo `for c in chunks` dentro la list comprehension.

### Perché una funzione dedicata `get_llm_contesto()` invece di riusare `get_llm()`

Tecnicamente potresti riusare lo stesso `ChatOpenAI` già in cache per le
risposte finali. Li teniamo separati per due ragioni pratiche:
1. **Costo/velocità configurabili indipendentemente**: potresti voler
   usare un modello diverso (più economico) solo per la contestualizzazione,
   senza toccare la qualità delle risposte finali all'utente.
   `temperature=0` in entrambi i casi perché vogliamo output deterministico
   (stesso chunk → stesso contesto), non creativo.
2. **Chiarezza**: leggendo `_indicizza_documento`, è immediato capire che
   quell'LLM serve per l'indicizzazione, non per rispondere a domande —
   stesso principio di leggibilità già seguito nel progetto (es. la
   separazione `rag/` vs `api/`).

### Perché il passo va PRIMA di `upsert_documento` e non dopo

`upsert_documento` (in `vectorstore.py`) fa `vectorstore.add_documents(docs)`,
che internamente chiama `embeddings.embed_query`/`embed_documents` sul
`page_content` di ogni `Document`. Se contestualizzassimo *dopo*
l'upsert, staremmo embeddando il testo "povero" e il contesto
generato non servirebbe a nulla per la ricerca (servirebbe solo, al più,
in output — ma qui vogliamo migliorare proprio il *retrieval*, cioè la
fase di ricerca).

---

## 5. Cosa aspettarti in pratica (costi e tempi)

Con il nostro chunking, un documento tipico (una scheda allenamento o
una settimana di dieta) produce qualche decina di chunk. Ogni chunk
aggiunge **una chiamata LLM in più** durante l'indicizzazione (che è già
in background, quindi non blocca la risposta HTTP — vedi sezione 5 di
`TUTORIAL.md`), ma allunga il tempo totale prima che lo stato diventi
`"ready"`. Con `gpt-4o-mini` e prompt caching attivo, il costo aggiuntivo
per documento è tipicamente frazioni di centesimo — ma se hai molti
documenti grandi, tienilo a mente: è un buon caso d'uso per il campo
`MODELLO_CONTESTO` configurabile del Passo 2.

Se in futuro l'attesa diventasse un problema, un'ottimizzazione naturale
(non implementata qui, per restare aderenti a quanto serve *ora*) sarebbe
parallelizzare le chiamate di `contestualizza_documento` con
`asyncio`/`ainvoke` invece del ciclo sequenziale — ma introdurrebbe
complessità (gestione della concorrenza, rate limit OpenAI) che non è
giustificata finché i documenti restano piccoli come oggi.

---

## 6. Riepilogo dei file toccati

| File | Cosa cambia |
|---|---|
| `src/rag/contextualize.py` | **Nuovo.** Prompt + funzioni di contestualizzazione. |
| `src/rag/config.py` | Nuova costante `MODELLO_CONTESTO`. |
| `src/api/state.py` | Nuova funzione cache `get_llm_contesto()`. |
| `src/api/routers/documents.py` | `_indicizza_documento` chiama `contestualizza_documento` tra chunking e upsert. |
| `src/rag/vectorstore.py` | **Nessuna modifica.** |
| `src/rag/graph.py` | **Nessuna modifica.** |

Vuoi che a questo punto scriva davvero il codice nei file (non solo nel
documento), così puoi testarlo caricando un documento reale?
