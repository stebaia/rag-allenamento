# Guida completa all'app: com'è fatta e perché

Questo documento è la guida di riferimento dell'applicazione: cosa fa ogni
pezzo, **perché** è stato costruito così (le alternative scartate e il
motivo), e come i pezzi comunicano tra loro.

Come si colloca rispetto agli altri documenti:

| Documento | Cosa copre |
|---|---|
| `TUTORIAL.md` | Concetti Python/FastAPI di base, avvio in locale, Docker, deploy su Coolify |
| `CONTEXTUAL_RETRIEVAL.md` | Approfondimento su una singola tecnica: la contestualizzazione dei chunk |
| **questo file** | Architettura completa, il chunking mirato e le sue regex, il retrieval ibrido, i flussi end-to-end |

> **Attenzione se modifichi le regex.** Il chunking di questo progetto è
> *mirato* ai PDF specifici del piano alimentare e della scheda di
> allenamento. Le regex non sono generiche: dipendono dal modo esatto in cui
> `pypdf` estrae il testo da quei file. La sezione 4 spiega ogni regex e
> cosa succede se la cambi.

---

## 1. Cos'è l'app, in una frase

Un assistente personale che risponde a domande in italiano su **dieta,
spesa e allenamento**, leggendo le risposte dai PDF che carichi (piano
alimentare, scheda palestra, lista della spesa) invece di inventarle.

Le tre interfacce con cui puoi usarlo:

1. **Web** — frontend React, chat + gestione documenti
2. **Alexa** — skill vocale ("Alexa, chiedi al mio assistente cosa mangio oggi")
3. **CLI** — `python main.py`, loop di domande nel terminale, utile per test rapidi

Tutte e tre passano per la stessa logica RAG: cambia solo il modo di
autenticare e il formato della risposta.

---

## 2. Architettura generale

### Il giro completo di una domanda

```
Utente scrive "cosa mangio oggi?"
        │
        ▼
frontend/src/components/chat.tsx      manda domanda + storico chat
        │  POST /ask  (JWT nell'header)
        ▼
src/api/routers/ask.py                autentica, prepara retriever + LLM
        │
        ├─► riformula_con_storico()    "e quella giornata?" -> domanda autonoma
        │
        ▼
src/rag/graph.py  (LangGraph)         il grafo agentico
        │
        ├─ nodo "recupera" ──► risolvi_giorni_relativi()    "oggi" -> "lunedì"
        │                      risolvi_sessioni_relative()  "prima sessione" -> "Sessione 1"
        │                              │
        │                              ▼
        │                      src/rag/vectorstore.py       RetrieverIbrido
        │                              │   semantico + lessicale + boost giorno
        │                              ▼
        │                      Qdrant (filtrato per user_id)
        │
        ├─ bivio "decidi"  ──► i documenti bastano? no -> "riformula" e ricicla
        │
        └─ nodo "genera"   ──► LLM (gpt-4o-mini) con i chunk come contesto
                                       │
                                       ▼
                            {"risposta": "..."}  ──► torna al frontend
```

### Le due basi di dati (e perché sono due)

| | **SQLite** (`app.db`) | **Qdrant** |
|---|---|---|
| Cosa contiene | utenti, metadati documenti | i vettori (embedding) dei chunk |
| Modulo | `src/rag/db.py` | `src/rag/vectorstore.py` |
| Serve per | login, "che documenti ho?", stato indicizzazione | "quali chunk somigliano a questa domanda?" |

Sono separati perché rispondono a domande di natura diversa. SQLite
risponde a domande **esatte** ("l'utente con questa email esiste?"), Qdrant
a domande **approssimate** ("cosa somiglia a questo significato?"). Un
database relazionale non sa cercare per similarità geometrica tra vettori;
un database vettoriale non sa fare join e vincoli di unicità.

### Struttura delle cartelle

```
src/
├── rag/                  logica di dominio, indipendente dal web
│   ├── config.py         tutte le costanti in un posto solo
│   ├── loaders.py        PDF/TXT -> testo grezzo
│   ├── chunking.py       testo -> chunk mirati        ⚠️ le regex stanno qui
│   ├── contextualize.py  arricchisce i chunk (vedi CONTEXTUAL_RETRIEVAL.md)
│   ├── vectorstore.py    Qdrant: indicizzazione + RetrieverIbrido
│   ├── reranking.py      cross-encoder opzionale (RERANKER=on)
│   ├── graph.py          LangGraph: il ciclo recupera/riformula/genera
│   ├── db.py             SQLModel: Utente, Documento
│   ├── auth.py           bcrypt + JWT
│   └── cli.py            entry point da terminale
│
└── api/                  lo strato web (FastAPI)
    ├── main.py           assembla l'app, CORS, lifespan
    ├── state.py          modelli pesanti caricati una volta (@lru_cache)
    ├── deps.py           dependency injection: sessione DB, utente da JWT
    └── routers/
        ├── auth.py       /auth/register, /auth/login
        ├── documents.py  upload + indicizzazione in background
        ├── ask.py        /ask  (JWT, con memoria conversazionale)
        └── alexa.py      /alexa/ask  (chiave condivisa, utente fisso)
```

**Perché `rag/` è separato da `api/`:** `rag/` non importa niente di
FastAPI. Questo permette di riusare la stessa logica dalla CLI e dalla
Lambda Alexa senza tirarsi dietro un web server, e rende ogni modulo
testabile chiamando funzioni normali. La dipendenza va in una sola
direzione: `api/` importa `rag/`, mai il contrario.

---

## 3. Il flusso di indicizzazione (upload di un PDF)

```
POST /documents  (file)
   │
   ├─ 1. valida l'estensione (.pdf/.txt)
   ├─ 2. salva il file su disco come {uuid}.pdf
   ├─ 3. crea la riga SQLite con stato "processing"
   ├─ 4. risponde SUBITO 202 Accepted + id documento
   │
   └─ 5. BackgroundTasks: _indicizza_documento()   ← dopo la risposta HTTP
             │
             ├─ carica_file()              PDF -> testo
             ├─ chunk_documento()          testo -> chunk mirati
             ├─ contestualizza_documento() +1-2 frasi di contesto per chunk
             ├─ upsert_documento()         embedding -> Qdrant
             └─ aggiorna stato: "ready" (o "error" + messaggio)
```

**Perché in background.** Estrarre il testo, chunkare, chiamare l'LLM di
contestualizzazione per ogni chunk e calcolare gli embedding può richiedere
decine di secondi. Se l'endpoint facesse tutto prima di rispondere, il
browser resterebbe appeso e finirebbe in timeout. Rispondendo subito con
`202 Accepted`, il frontend mostra "indicizzazione in corso" e fa polling
su `GET /documents/{id}` ogni 2 secondi (`document-list.tsx`) finché lo
stato non diventa `ready`.

**Perché il background task apre una sessione DB propria.** La `SessionDep`
iniettata nell'endpoint viene chiusa da FastAPI appena la risposta parte —
cioè *prima* che il task inizi. Quindi `_indicizza_documento` chiama
`get_session()` a mano. Ed è per questo che il `try/except` interno cattura
`Exception` in modo volutamente ampio: nessuno sta guardando un traceback
in background, quindi l'errore va **salvato nella colonna `errore`** del
documento, dove il frontend lo può mostrare all'utente.

---

## 4. Il chunking mirato ⚠️ (il cuore del progetto)

Questa è la parte più delicata e più specifica. Vale la pena capirla bene
prima di toccarla.

### Perché non lo splitter standard

Il modo normale di chunkare è tagliare il testo ogni N caratteri
(`RecursiveCharacterTextSplitter`). Sui nostri PDF funziona male:

- un taglio a caratteri fissi spezza **una tabella a metà**: i macro
  (carboidrati/proteine/grassi/kcal) finiscono in un chunk e gli alimenti a
  cui si riferiscono in un altro;
- il PDF estratto da `pypdf` è una sequenza di celle di tabella una per
  riga, quindi il testo grezzo è quasi illeggibile per l'LLM;
- un chunk "3x10 / 90''" senza il nome dell'esercizio accanto è inutile.

La scelta di questo progetto: **ricostruire il significato delle tabelle**
in frasi in italiano, una per unità logica (un pasto, un esercizio). Il
chunk finale non è un pezzo del PDF, è una frase costruita da noi:

```
Lunedì, pranzo: riso 80g, pollo 150g (macro: 190g carboidrati, 105g proteine, 65g grassi, 1761 kcal)
Sessione 1 — Upper, allenamento: Panca piana — 4x8, recupero 90'', focus Petto
```

Questo aiuta due volte: l'embedding cattura un significato pulito, e l'LLM
finale legge una frase che capisce senza sforzo.

### Le tre strade di `chunk_documento()`

```python
if any(k in nome for k in KEEP_COMPACT):   # "spesa", "lista", "shopping", "grocery"
    → split_spesa()                        # un chunk per alimento     tipo="spesa"
elif il blocco inizia con un giorno        # piano alimentare
    → split_pasti()                        # un chunk per pasto        tipo="dieta"
elif il blocco inizia con "SESSIONE n"     # scheda allenamento
    → split_sessione()                     # un chunk per esercizio    tipo="allenamento"
else
    → split_ricorsivo(800)                 # note, indicazioni, testo libero
```

La scelta dipende dal **nome del file** (`KEEP_COMPACT`) e dalla **forma
del contenuto**. Ogni chunk esce da `_record()`, che garantisce a tutti le
stesse chiavi di metadati — inclusa `tipo`, usata dai boost del retriever
(sezione 5).

### Perché anche la spesa ha il suo splitter

Inizialmente la lista della spesa cadeva su `split_ricorsivo(1500)`,
diventando **2 soli chunk** con ~40 alimenti mescolati dentro. Il problema
pratico: alla domanda "che spesa devo fare per la dieta di oggi?"
l'assistente elencava gli alimenti del pasto (`Fette biscottate (3 fette)`)
invece delle quantità da comprare (`■ Fette biscottate — 12 fette, 1
confezione`) — due informazioni diverse, che stanno in due documenti diversi.

`split_spesa()` ricostruisce la tabella `■ Alimento / Quantità /
Indicazioni` in una frase per alimento, tenendo il reparto:

```
Lista della spesa, dispensa: Fette biscottate — 12 fette (1 confezione)
Lista della spesa, ortofrutta: Verdura — 2,1–2,8 kg (14 porzioni da 150-200 g...)
```

Da 2 chunk-blob a 29 chunk granulari. Le quantità hanno forme molto
variabili (`12 fette`, `3 pz`, `550 ml`, `2,1–2,8 kg`, `200 g sgocciolato`),
quindi `_SPESA_QUANTITA` riconosce semplicemente "inizia con una cifra"
(eventualmente preceduta da `≈`) invece di elencare le unità di misura. Le
`Note` finali diventano un chunk unico: contengono le sostituzioni previste
(pollo → tacchino) ed è utile che restino insieme.

### Le regex, una per una

Tutte in `src/rag/chunking.py`. Il carattere `\x7f` che compare in alcune
non è un errore: è il modo in cui `pypdf` restituisce il **bullet point**
dei nostri PDF.

| Regex | Cosa riconosce | Nota |
|---|---|---|
| `_GIORNO` | `lunedì`…`domenica` | accetta sia `ì` che `i` (`luned[ìi]`) perché l'accento a volte si perde nell'estrazione |
| `_HEADER` | inizio di un blocco: giorno dieta, `SESSIONE n`, `Reverse n`, `Note generali`, `Indicazioni` | usa `(?=...)` (**lookahead**): vedi sotto |
| `_PASTO` | `Colazione`, `Pranzo`, `Spuntino serale`, `Spuntino`, `Cena` | `Spuntino serale` **prima** di `Spuntino`: vedi sotto |
| `_MEAL` | riga completa di un pasto con i 4 numeri dei macro | i 4 gruppi numerici sono carb/prot/grassi/kcal |
| `_REVERSE_HEADER` | `Reverse 1 — dal 03/08/2026` | cattura numero e data della settimana |
| `_SESSIONE_START` / `_SESSIONE` | `SESSIONE 3 — Upper` | il nome della seduta è il gruppo 2 |
| `_SOTTOTABELLA` | `RISCALDAMENTO` / `ALLENAMENTO` | separa le due tabelle dentro una seduta |
| `_HEADER_TABELLA` | `ESERCIZIO`, `SERIE × REP`, `RECUPERO`… | righe di intestazione da **scartare** |
| `_SERIE_REP` | `3x10`, `4 x 8`, `12"`, `3x10/8` | è il segnale che il nome dell'esercizio è finito |
| `_INIZIO_RECUPERO` | `/` oppure un numero con `'`/`"` | segnala il passaggio alla colonna recupero |
| `_FOCUS_NOTI` | `Petto`, `Dorsale`, `Bicipiti`… | lista chiusa di gruppi muscolari |
| `_SPESA_ALIMENTO` | `■ Fette biscottate` | accetta `\x7f`, `■`, `•`, `·`, `-` come bullet |
| `_SPESA_REPARTO` | `Dispensa`, `Banco frigo / freschi`, `Ortofrutta` | titolo di sezione, finisce nel chunk |
| `_SPESA_QUANTITA` | `12 fette`, `≈2 tavolette`, `550 ml` | "inizia con una cifra": non elenca le unità |

#### Due dettagli che sembrano bug e non lo sono

**1. `(?=...)` nel split.** `_PASTO` e `_HEADER` usano un *lookahead*:
```python
_PASTO = re.compile(r"(?=(?:Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b)")
```
Un `re.split` normale **consuma** il separatore, cioè lo butta via. Qui il
separatore è anche il dato che ci serve (la parola "Pranzo" deve restare
nel chunk!), quindi il lookahead dice "taglia *appena prima* di questa
parola, senza mangiarla".

**2. L'ordine delle alternative.** In `_PASTO`, `Spuntino serale` viene
prima di `Spuntino`. Le regex alternano **da sinistra a destra e si
fermano al primo match**: con l'ordine invertito, `Spuntino serale`
verrebbe spezzato in `Spuntino` + `serale` orfano. Stessa logica in
`_NUMERI_ORDINALI` in `graph.py`. **Se aggiungi un pasto, mettilo nel
posto giusto rispetto ai suoi prefissi.**

### Il parser a stati degli esercizi

`_parsa_esercizi()` è la parte più intricata, e per un motivo preciso:
`pypdf` restituisce le celle della tabella **una per riga, senza dirci a
quale colonna appartengono**. Il testo grezzo assomiglia a questo:

```
Panca piana          ← nome
4x8                  ← serie×rep
90''                 ← recupero
Petto                ← focus
Trazioni alla sbarra ← nome del prossimo esercizio...
```

Non esiste un separatore di colonna: bisogna **indovinare dalla forma**
del contenuto quando si passa da una colonna all'altra. Da qui la piccola
macchina a stati `NOME → SERIE_REP → RECUPERO`, dove le transizioni sono
guidate dalle regex: se una riga somiglia a `3x10` allora il nome è
finito, se somiglia a `90''` siamo nel recupero, e così via.

I casi particolari gestiti (tutti osservati nei PDF reali):

- nomi su più righe: `Curl (manubri)` → il pezzo tra parentesi viene unito
  al nome, non trattato come esercizio nuovo;
- recupero su due righe: `30-60" tra un` + `braccio e l'altro`;
- focus su due righe: `Centro` + `schiena`.

La regola usata per distinguere una continuazione da un esercizio nuovo è
l'**iniziale minuscola** (`r[:1].islower()`): i nomi degli esercizi nei
nostri PDF iniziano sempre in maiuscolo. Fragile ma sufficiente per questi
documenti — ed è il punto da controllare per primo se un giorno il parsing
di una scheda nuova sbaglia.

### I piani "Reverse" (settimane multiple)

Un piano *reverse diet* ha più settimane a calorie crescenti, e **ripete gli
stessi nomi di giorno** in ognuna con macro diverse:

```
Reverse 1 — dal 03/08/2026     Lunedì: 190g carb ...
Reverse 2 — dal 10/08/2026     Lunedì: 210g carb ...
```

Senza distinguerle, "cosa mangio lunedì?" restituirebbe **tutti** i lunedì
di tutte le settimane mescolati. La soluzione ha due metà:

1. **In indicizzazione** (`chunk_documento`): il numero e la data della
   settimana corrente vengono ricordati mentre si scorrono i blocchi e
   applicati a tutti i giorni successivi, come metadati `reverse` /
   `reverse_dal` **e** anteposti al testo del chunk. Il testo, non solo i
   metadati, perché la ricerca lessicale cerca parole nel testo e l'LLM
   finale legge solo il testo.
2. **In retrieval** (`_reverse_da_privilegiare`): sceglie *una* settimana,
   con questa priorità — la domanda nomina "Reverse 2" → quella; altrimenti
   la settimana in corso oggi; altrimenti, se il piano non è ancora
   iniziato, la prima che partirà. Con un fallback: se il filtro non trova
   niente, si riprova senza, per non lasciare l'utente senza risposta.

---

## 5. Il retrieval ibrido

`RetrieverIbrido` in `vectorstore.py` combina tre segnali. Il motivo è che
la ricerca puramente semantica sbaglia su questi documenti.

**1. Semantico (embedding).** La domanda diventa un vettore, Qdrant
restituisce i 40 chunk più vicini per distanza coseno. Cattura bene il
*significato* ("cosa mangio" ≈ "colazione, pranzo, cena") ma è **debole sui
dettagli esatti**: numeri, nomi di alimenti e di esercizi pesano poco in un
embedding.

**2. Lessicale (TF-IDF sul pool).** Rimedia proprio a questo: premia i
chunk che contengono le **parole esatte** della domanda, pesate per quanto
sono rare nel pool (idea IDF — una parola presente in pochi chunk è più
discriminante). Il punteggio finale è:

```python
punteggio = similarità_coseno + 0.4 * punteggio_lessicale_normalizzato
```

**3. Boost sul giorno.** Se la domanda nomina un giorno, una query separata
recupera **tutti** i chunk di quel giorno e li mette in testa. Serve perché
una giornata alimentare completa sono ~6 chunk (uno per pasto): con il solo
`k=5` semantico, la risposta a "cosa mangio lunedì?" rischierebbe di
contenere il pranzo di lunedì e pezzi di altri giorni.

**4. Boost sulla spesa.** Se la domanda contiene `spesa`/`comprare`/…
(`_SPESA_QUERY`), i chunk con `tipo="spesa"` vengono recuperati
esplicitamente. Vedi sotto il perché.

### Il reranking (attivo, si spegne con `RERANKER=off`)

I segnali 1 e 2 hanno un limite comune: sono due numeri calcolati
**separatamente** per domanda e chunk. L'embedding del chunk è stato prodotto
in fase di indicizzazione, quello della domanda al momento della ricerca: i
due testi non si "vedono" mai insieme, si confrontano solo i vettori. Il
punteggio lessicale conta parole in comune, e non legge nulla.

Un **cross-encoder** fa la cosa opposta: riceve la coppia (domanda, chunk)
come un unico input e la processa in un solo passaggio, producendo un
punteggio di rilevanza. Può così riconoscere che un chunk risponde alla
domanda anche quando non ne condivide le parole né una somiglianza semantica
generica. Il prezzo è che va eseguito su **ogni coppia**: non esiste un indice
da precalcolare, per questo il reranking arriva sempre in seconda battuta, sul
pool già ristretto dal retrieval (i ~40 candidati, non l'intera collection).

Serve soprattutto sui documenti **senza boost dedicato** — CCNL, codici, testi
di struttura ignota — dove il ranking non ha altri appigli oltre a semantica e
parole. Sui piani alimentari i boost fanno già un lavoro migliore.

Due scelte di progetto, entrambe deliberate:

**Agisce solo sul ranking ibrido, mai sui boost.** I chunk di
giorno/spesa/capitolo sono recuperati perché la domanda li nomina
esplicitamente, non perché somigliano alla domanda. Il boost sulla spesa
esiste *proprio perché* «Fette biscottate — 12 fette» non somiglia a «cosa
devo comprare»: passarlo a un reranker lo farebbe declassare, annullando il
lavoro descritto nella sezione qui sotto.

**Se il modello non si carica, degrada invece di fallire.** Il reranking
migliora il ranking, non è un prerequisito per rispondere: un modello non
scaricato o troppa memoria occupata lasciano l'ordine del ranking ibrido, con
una riga in log, invece di far fallire la domanda dell'utente.

**Costo.** `BAAI/bge-reranker-v2-m3` pesa ~2,3 GB, scaricati al primo avvio, e
aggiunge 1-3 s per domanda su CPU. Il modello viene caricato nel `lifespan` di
`api/main.py` insieme agli embedding, così l'attesa la paga l'avvio del
container e non l'utente che fa la prima domanda. Si spegne con `RERANKER=off`,
utile per confrontare le due modalità sulla stessa domanda. In Docker, `HF_HOME`
punta al volume: senza, il modello verrebbe riscaricato a ogni redeploy.

### Le domande che incrociano due documenti

Questo è il caso che ha richiesto più lavoro, perché fallisce in modo
silenzioso: *"che spesa devo fare per la dieta di oggi?"*

La domanda tocca **due documenti** — il piano alimentare (cosa mangio
lunedì) e la lista della spesa (quanto ne compro). Tre cause si sommavano:

1. **Il boost sul giorno saturava il contesto.** Con due versioni del piano
   indicizzate, "lunedì" produce 12 chunk. Con `k = len(boost) + 2 = 14`,
   restavano **2 posti** per tutto il resto, e la spesa (3ª-4ª nel ranking)
   veniva tagliata.
2. **Il ranking non poteva farla emergere.** Le parole della domanda
   ("spesa", "comprare") **non compaiono** nel testo dei chunk dei singoli
   alimenti (`Fette biscottate — 12 fette`). Né il lessicale né il semantico
   li premiano in modo affidabile.
3. **L'LLM non sapeva di dover incrociare** i due documenti.

Le tre contromisure, una per causa:

- `_unisci_giorno_e_ranking()` mette un **tetto** al boost (`max_boost=8`) e
  garantisce una **quota** ai migliori del ranking (`min_altri=4`): il boost
  non può più occupare tutto il contesto.
- `_SPESA_QUERY` riconosce l'intento e recupera i chunk `tipo="spesa"`,
  **ordinandoli per pertinenza** — non nell'ordine del PDF. La pertinenza è
  la sovrapposizione lessicale con i chunk del giorno già recuperati, così
  per il lunedì emergono fette biscottate, pasta di legumi e yogurt (che
  sono nei pasti di lunedì) invece dei primi alimenti in elenco.
- Il prompt in `ask.py` istruisce l'LLM a riportare le quantità **della
  lista**, non i grammi dei pasti, e a incrociare i due documenti quando
  trova entrambi nel contesto.

> Nota utile per il debug: se hai indicizzato **due versioni** dello stesso
> piano (es. `_compatto` e `_v2`), il retriever le tratta come documenti
> distinti e il contesto contiene due lunedì con macro diverse. È il motivo
> per cui il tetto `max_boost` esiste. Se le risposte mescolano quantità
> incoerenti, cancella la versione vecchia da `/documents`.

**Perché una classe custom invece del retriever standard di LangChain:** il
retriever di Qdrant sa fare solo il punto 1. I punti 2 e 3 sono euristiche
legate alla forma di *questi* documenti. Ereditando da `BaseRetriever`,
`graph.py` continua a chiamare `retriever.invoke(domanda)` senza sapere
nulla di tutto questo.

### L'isolamento tra utenti

Tutti gli utenti condividono una collection Qdrant, ma **ogni ricerca
filtra sempre per `user_id`** (`_filtro_utente`, applicato sia alla query
semantica sia alle query del boost). Lo stesso vale sul lato SQLite: ogni
endpoint di `documents.py` verifica `documento.user_id != utente.id` e
risponde `404` — non `403` — così non si rivela nemmeno l'esistenza di un
documento altrui.

---

## 6. Il grafo LangGraph

`costruisci_grafo()` monta tre nodi e un bivio:

```
START ──► recupera ──► [decidi] ──► genera ──► END
              ▲            │
              └─ riformula ◄┘   (max 2 tentativi)
```

- **recupera** — risolve prima le parole relative (`oggi` → `lunedì`,
  `prima sessione` → `Sessione 1`), poi interroga il retriever. La
  risoluzione avviene **solo al primo tentativo**: dal secondo, la domanda
  è già stata riscritta dall'LLM.
- **decidi** — chiede all'LLM se i documenti recuperati bastano. Se no, si
  passa da `riformula`. Il limite di 2 tentativi evita cicli infiniti (e
  costi che crescono).
- **riformula** — l'LLM riscrive la domanda con più parole chiave.
- **genera** — costruisce la risposta finale dai chunk recuperati.

**Perché risolvere le parole relative con delle regex e non chiedendolo
all'LLM.** Perché serve che nella stringa passata al retriever ci siano
**le parole esatte** presenti nei chunk: `RetrieverIbrido` cerca
letteralmente `lunedì` nei metadati e `Sessione 1` nel testo. Una
sostituzione deterministica è gratis, istantanea e non può allucinare. Il
compromesso: le regex coprono solo le forme previste
(`oggi/domani/ieri/dopodomani`, `primo/secondo/terzo`).

> Dettaglio di lingua: `risolvi_sessioni_relative` riscrive anche
> l'articolo al femminile (`nel primo giorno` → `nella Sessione 1`), perché
> "giorno" è maschile ma "Sessione" è femminile e la domanda riscritta
> finisce dentro il prompt dell'LLM.

### La memoria conversazionale

Il backend è **stateless**: non ricorda niente tra due richieste. È il
frontend che rimanda lo storico della chat ad ogni domanda
(`chat-context.tsx` → `ask.py`), troncato agli ultimi `MAX_STORICO = 8`
messaggi per non far crescere costi e latenza senza limite.

Lo storico viene usato in **due punti diversi**, ed è una distinzione
importante:

1. `riformula_con_storico()`, **prima** del retrieval: trasforma "e quella
   giornata?" in una domanda autonoma. Indispensabile perché il retriever
   non ha alcuna nozione di conversazione: cercherebbe alla lettera "quella
   giornata".
2. Il segnaposto `{storico}` nel prompt finale: serve solo a dare
   continuità di tono e a risolvere riferimenti impliciti nella risposta.

---

## 7. Autenticazione

Due meccanismi diversi per due tipi di client.

**Web — JWT.** Registrazione/login restituiscono un token firmato che il
frontend salva in `localStorage` e rimanda nell'header `Authorization`. Il
server non tiene sessioni: verifica solo la firma e la scadenza. Le
password sono salvate come hash **bcrypt** (mai in chiaro), e il login
risponde con un errore volutamente generico ("Credenziali non valide") per
non rivelare quali email sono registrate.

`JWT_SECRET` **non ha un default**: se manca, `rag/auth.py` solleva
un'eccezione all'import e l'app non parte. Un default silenzioso sarebbe un
segreto debole scelto da noi, cioè il tipo di problema che si scopre troppo
tardi.

**Alexa — chiave condivisa.** Alexa non fa login, quindi `/alexa/ask` è
protetto da una chiave segreta fissa (`X-Api-Key`) e risponde sempre per un
utente fisso (`ALEXA_USER_ID`). Il confronto usa
`secrets.compare_digest` invece di `==`: impiega un tempo costante e non
lascia dedurre la chiave misurando i tempi di risposta.

---

## 8. Il frontend

React 19 + TanStack Router/Query + Tailwind + shadcn/ui.

| File | Ruolo |
|---|---|
| `lib/api.ts` | unico punto che parla col backend; aggiunge il token, normalizza gli errori in `ApiError` |
| `lib/auth.tsx` | contesto di autenticazione |
| `lib/chat-context.tsx` | i messaggi vivono qui, **non** dentro `Chat` |
| `components/chat.tsx` | la chat |
| `components/document-list.tsx` | lista + polling adattivo |
| `routes/_authenticated.tsx` | rotta "guardia": redirige al login se non autenticato |

Due scelte da notare:

- **I messaggi stanno in un context, non nel componente.** Passando da
  `/chat` a `/documents` e tornando, il router smonta e rimonta `Chat`:
  se lo stato vivesse lì, la conversazione si perderebbe ad ogni cambio di
  tab.
- **Il polling si spegne da solo.** `refetchInterval` è una funzione:
  ritorna 2000 ms solo se almeno un documento è in `processing`, altrimenti
  `false`. Nessuna richiesta inutile a riposo.
- **Ottimismo con rollback.** Il messaggio dell'utente appare subito; se la
  richiesta fallisce, `onError` lo rimuove con `slice(0, -1)`.

---

## 9. Configurazione

Tutte le costanti stanno in `src/rag/config.py`. Quelle che cambiano tra
locale e produzione si leggono dall'ambiente con un default sensato, così
lo stesso codice gira in entrambi i posti:

| Variabile | Default | Note |
|---|---|---|
| `OPENAI_API_KEY` | — | obbligatoria |
| `JWT_SECRET` | — | **nessun default**, l'app non parte senza |
| `QDRANT_URL` | `http://localhost:6333` | in Docker: `http://qdrant:6333` |
| `DB_PATH` | `app.db` | in Docker su volume: `/data/app.db` |
| `CARTELLA_UPLOAD` | `uploads` | in Docker: `/data/uploads` |
| `ORIGINI_CONSENTITE` | `http://localhost:3000` | CORS, lista separata da virgole |
| `ALEXA_API_KEY` / `ALEXA_USER_ID` | vuote | senza di esse `/alexa/ask` risponde 503 |
| `K` | `5` | chunk passati all'LLM |
| `MAX_STORICO` | `8` | messaggi di storico inviati |

I modelli pesanti vengono creati **una volta sola** grazie a
`@lru_cache(maxsize=1)` in `api/state.py`, e il modello di embedding viene
precaricato nel `lifespan` all'avvio: così i ~30 secondi di caricamento li
paga il boot del container, non il primo utente che fa una domanda.

---

## 10. Se qualcosa non funziona

| Sintomo | Dove guardare |
|---|---|
| Documento resta in `error` | la colonna `errore` contiene il messaggio; PDF scansionato → serve OCR |
| "cosa mangio lunedì" risponde male | il boost sul giorno dipende dal metadato `giorno`: controlla che il chunking lo popoli |
| Risposte che mescolano settimane diverse | metadati `reverse`/`reverse_dal`: rivedi `_REVERSE_HEADER` sul formato reale dell'header |
| Chiedi la spesa e ti elenca i pasti | serve il boost `_SPESA_QUERY` + `tipo="spesa"`: **i documenti già indicizzati vanno ricaricati** per avere il nuovo metadato |
| Le quantità della spesa non tornano | probabile doppia versione dello stesso piano indicizzata: cancella la vecchia |
| Esercizi spezzati o uniti male | `_parsa_esercizi`: stampa le `righe` in ingresso e verifica la regola dell'iniziale minuscola |
| L'app non parte | manca `JWT_SECRET` (errore esplicito all'avvio) |
| Il browser blocca le chiamate | `ORIGINI_CONSENTITE` non contiene il dominio del frontend |

**Il modo più rapido di ispezionare il chunking** senza toccare l'API:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from rag.loaders import carica_documenti
from rag.chunking import chunk_documento
for doc in carica_documenti('documenti'):
    ch = chunk_documento(doc)
    print(f'--- {doc[\"fonte\"]}: {len(ch)} chunk')
    for c in ch[:5]:
        print('   ', c['giorno'] or '-', '|', c['testo'][:90])
"
```

Nota che questo non richiede né Qdrant né la chiave OpenAI: `chunking.py`
è puro Python, ed è esattamente il motivo per cui la logica di dominio è
tenuta separata dallo strato web.

---

## 11. Riepilogo delle scelte di progetto

| Scelta | Perché | Costo accettato |
|---|---|---|
| Chunking mirato con regex | le tabelle dei PDF, tagliate a caratteri fissi, perdono senso | va adattato se cambia il formato dei PDF |
| Retrieval ibrido | l'embedding da solo sbaglia su numeri e nomi propri | più codice e una query in più |
| Boost sul giorno | una giornata alimentare sono ~6 chunk, `k=5` non basta | `k` variabile a runtime |
| Chunk per alimento nella spesa | un blob da 40 alimenti non fa emergere la quantità del singolo | 29 chunk invece di 2 |
| Boost + quote per la spesa | le parole "spesa"/"comprare" non sono nel testo dei chunk degli alimenti | due euristiche in più da mantenere |
| Regex per "oggi"/"prima sessione" | serve la parola *esatta* nel testo; deterministico e gratis | copre solo le forme previste |
| Contestualizzazione dei chunk | un chunk isolato perde il piano a cui appartiene | una chiamata LLM per chunk in indicizzazione |
| Indicizzazione in background | l'upload sarebbe andato in timeout | serve il polling lato client |
| Embedding locale (HuggingFace) | gratis e nessun dato inviato a terzi | ~30 s di caricamento al boot, RAM |
| SQLite + Qdrant | due tipi di domanda diversi, esatta vs approssimata | due sistemi da gestire nel deploy |
| `rag/` separato da `api/` | riuso da CLI/Lambda, testabilità | un livello di indirezione in più |
| Backend stateless | nessuna sessione server da gestire o scalare | il client rimanda lo storico ogni volta |
