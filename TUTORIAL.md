# Tutorial: come è fatto questo backend RAG (e come si usa)

Questo documento spiega, passo per passo, com'è costruita l'API che hai
adesso: cosa fa ogni pezzo, quali concetti Python/FastAPI servono per
capirlo, come farlo girare sul tuo computer e come metterlo online su
Coolify. È scritto pensando che tu non abbia molta esperienza Python:
dove serve, mi fermo a spiegare il concetto del linguaggio, non solo il
codice del progetto.

Il codice sorgente ha commenti approfonditi negli stessi punti spiegati
qui — apri i file mentre leggi, è il modo più efficace di impararlo.

---

## 1. Concetti Python usati nel progetto

Prima di guardare il codice, un elenco dei concetti che ricorrono
ovunque. Se già li conosci, salta alla sezione 2.

### Type hints (annotazioni di tipo)

```python
def somma(a: int, b: int) -> int:
    return a + b
```

`a: int` dice "mi aspetto che `a` sia un intero", `-> int` dice "questa
funzione ritorna un intero". Python **non obbliga** a rispettarle (non è
come Java o TypeScript), ma editor e strumenti come FastAPI le leggono e
le usano per: 1) darti autocompletamento e avvisi in caso di errore, 2)
validare automaticamente dati in ingresso/uscita dalle API.

`str | None` significa "una stringa, oppure `None`" (l'equivalente Python
di "nullo"/"vuoto").

### f-string

```python
nome = "Stefano"
saluto = f"Ciao {nome}!"  # "Ciao Stefano!"
```

Una stringa con `f` davanti alle virgolette permette di inserire il
valore di variabili dentro `{}`.

### List comprehension

```python
quadrati = [x * x for x in [1, 2, 3]]  # [1, 4, 9]
```

Un modo compatto di scrivere un ciclo `for` che costruisce una lista.
Equivale a:

```python
quadrati = []
for x in [1, 2, 3]:
    quadrati.append(x * x)
```

### Generatori (`yield`)

```python
def conta_fino_a(n):
    for i in range(n):
        yield i
```

Una funzione con `yield` invece di `return` non esegue tutto il corpo in
una volta: si "mette in pausa" ad ogni `yield`, restituisce quel valore,
e riprende da lì quando richiesto di nuovo. Nel progetto lo usiamo in
`get_session()` (vedi `src/rag/db.py`): il codice prima di `yield` apre
una connessione al database, il valore restituito è la connessione
stessa, e — grazie al blocco `with` — la connessione si chiude da sola
quando chi la usa ha finito.

### Decoratori (`@qualcosa`)

```python
@lru_cache(maxsize=1)
def carica_modello():
    ...
```

Un decoratore è una funzione che "avvolge" un'altra funzione per
aggiungerle un comportamento, senza modificarne il codice interno. Nel
progetto ne troviamo diversi tipi:
- `@lru_cache(maxsize=1)`: memorizza il risultato della prima chiamata e
  lo riusa per sempre (utile per modelli pesanti da caricare, vedi
  `src/api/state.py`).
- `@router.post("/ask")`: collega una funzione a un URL e un metodo HTTP
  (questo è lo stile di FastAPI).

### Enum

```python
from enum import Enum

class Colore(str, Enum):
    ROSSO = "rosso"
    VERDE = "verde"
```

Serve a limitare un campo a un insieme fisso di valori validi. Nel
progetto: `StatoDocumento` (`processing`, `ready`, `error`).

### `Annotated` e dependency injection (FastAPI)

```python
SessionDep = Annotated[Session, Depends(get_session)]

@router.get("/documents")
def lista(session: SessionDep):
    ...
```

`Annotated[Tipo, Depends(funzione)]` dice a FastAPI: "questo parametro è
di tipo `Tipo`, e per ottenerne il valore chiama `funzione`". FastAPI
chiama da solo tutte le "dependency" necessarie, nell'ordine giusto,
prima di eseguire il codice dell'endpoint. È il modo in cui, nel
progetto, ogni endpoint ottiene automaticamente una connessione al
database (`SessionDep`) o l'utente autenticato (`UtenteCorrente`), senza
doverlo scrivere a mano ogni volta.

---

## 2. Come era strutturato il progetto originale

Prima di questa trasformazione, il progetto era uno script a riga di
comando (`main.py` → `src/rag/cli.py`):

1. Legge tutti i PDF dalla cartella `documenti/`.
2. Li spezza in "chunk" (piccoli pezzi di testo) con `src/rag/chunking.py`.
3. Calcola l'embedding di ogni chunk e li salva in un database vettoriale
   locale su file (Chroma, cartella `chroma_lc/`).
4. Apre un ciclo interattivo: l'utente scrive una domanda, il programma
   cerca i chunk più pertinenti e chiede a un LLM (GPT-4o-mini) di
   rispondere usando quel contesto (`src/rag/graph.py`).

Questa logica di base **non è cambiata**: l'abbiamo riorganizzata dietro
un'API HTTP, in modo che un frontend web (o qualunque altro client)
possa usarla, con più utenti che caricano ciascuno i propri documenti.

I file `rag.py` e `ragV2.py` nella root sono le versioni storiche dello
script, lasciate intatte apposta.

---

## 3. Architettura del backend

```
Utente (frontend, curl, Postman...)
        │  HTTP + JWT
        ▼
   FastAPI (src/api/)
   ├── /auth/register, /auth/login   → autenticazione
   ├── /documents (POST/GET/DELETE)  → upload e gestione file
   └── /ask                          → interrogazione RAG
        │
        ├──► SQLite (src/rag/db.py)     — chi sono gli utenti, quali documenti hanno, a che stato
        ├──► Qdrant (src/rag/vectorstore.py) — i vettori (embedding) dei chunk di testo
        └──► OpenAI (gpt-4o-mini)        — genera la risposta finale in linguaggio naturale
```

Due database, per due scopi diversi:
- **SQLite**: dati strutturati e piccoli (utenti, metadati documenti). Un
  singolo file su disco, zero configurazione.
- **Qdrant**: milioni di vettori, ottimizzato per cercare velocemente "i
  chunk più simili a questa domanda". Gira come servizio separato
  (container Docker).

### Struttura delle cartelle

```
src/
├── rag/                    # logica di dominio, indipendente da HTTP
│   ├── config.py           # costanti e variabili d'ambiente
│   ├── db.py                # modelli SQLModel (Utente, Documento)
│   ├── auth.py               # hashing password + JWT
│   ├── vectorstore.py        # Qdrant: upsert, delete, retriever
│   ├── loaders.py             # estrazione testo da PDF/TXT
│   ├── chunking.py            # spezza il testo in chunk
│   ├── graph.py                # grafo LangGraph di retrieval+generazione
│   └── cli.py                   # script da terminale (demo/debug)
└── api/                     # tutto ciò che è HTTP
    ├── main.py               # crea l'app FastAPI, monta i router
    ├── deps.py                # dependency condivise (sessione DB, utente corrente)
    ├── state.py                # modelli pesanti (embedding, LLM) caricati una volta
    └── routers/
        ├── auth.py             # /auth/register, /auth/login
        ├── documents.py         # /documents
        └── ask.py                # /ask
```

La separazione `rag/` vs `api/` non è decorativa: `rag/` contiene la
logica che avrebbe senso anche senza un server HTTP (potresti riusarla
in uno script, in un job schedulato, ecc.), mentre `api/` è lo strato
sottile che la espone via HTTP.

---

## 4. Il flusso di autenticazione (JWT)

File: `src/rag/auth.py`, `src/api/routers/auth.py`, `src/api/deps.py`.

1. `POST /auth/register` con `{"email": "...", "password": "..."}`.
   Il server calcola l'hash bcrypt della password (mai salvata in
   chiaro), crea l'utente nel database, e ritorna un **token JWT**.
2. Il client salva il token e lo manda in ogni richiesta successiva
   nell'header:
   ```
   Authorization: Bearer <token>
   ```
3. Ogni endpoint protetto dichiara `utente: UtenteCorrente` come
   parametro. FastAPI, tramite la dependency `utente_corrente` in
   `deps.py`, estrae il token dall'header, lo decodifica, verifica che
   non sia scaduto/manomesso, e recupera l'utente dal database. Se
   qualcosa non va, risponde da solo con `401 Unauthorized` — il codice
   dell'endpoint non viene nemmeno eseguito.

Perché JWT e non "sessioni" classiche? Con JWT il server non deve
ricordarsi chi è loggato: tutta l'informazione (l'id dell'utente) è nel
token stesso, firmato in modo che non si possa falsificare senza
conoscere `JWT_SECRET`. Più semplice da scalare su più server.

---

## 5. Il flusso di upload documenti

File: `src/api/routers/documents.py`.

```
POST /documents (file)
   │
   ▼
1. Validazione estensione (.pdf o .txt)
2. Salvataggio del file su disco (cartella uploads/)
3. Riga nel DB con stato "processing"
4. Risposta immediata al client: {"id": "...", "stato": "processing"}
5. [IN BACKGROUND, dopo la risposta]
   a. Estrazione testo dal file (loaders.carica_file)
   b. Chunking del testo (chunking.chunk_documento)
   c. Calcolo embedding + salvataggio su Qdrant (vectorstore.upsert_documento)
   d. Aggiornamento stato → "ready" (o "error" se qualcosa fallisce)
```

Il client può interrogare `GET /documents/{id}` finché lo stato non
diventa `"ready"` (o `"error"`). Questo pattern si chiama **polling** ed
è la soluzione più semplice per "sapere quando un lavoro lungo è
finito" senza tecnologie aggiuntive (WebSocket, code di messaggi...).

`DELETE /documents/{id}` fa il percorso inverso: rimuove i vettori da
Qdrant, la riga dal database, e il file da disco.

---

## 6. Il flusso di una domanda (`/ask`)

File: `src/api/routers/ask.py`, `src/rag/graph.py`, `src/rag/vectorstore.py`.

```
POST /ask {"domanda": "quali esercizi per le gambe?"}
   │
   ▼
1. Autenticazione (chi sta chiedendo?)
2. Creazione di un "retriever" filtrato: cercherà SOLO tra i vettori
   con user_id uguale a quello dell'utente autenticato (isolamento
   multi-utente)
3. Esecuzione del grafo LangGraph (src/rag/graph.py):
   a. recupera i chunk più pertinenti dal vector store
   b. un LLM giudica se bastano a rispondere
      - se sì → genera la risposta
      - se no (max 2 tentativi) → riformula la domanda e ricerca di nuovo
4. Risposta al client: {"risposta": "..."}
```

Il grafo (`costruisci_grafo` in `src/rag/graph.py`) non è stato
modificato rispetto alla versione a riga di comando originale: riceve
semplicemente un retriever "già pronto e filtrato" invece di uno
generico.

---

## 7. Isolamento multi-utente: come funziona davvero

Non esiste un database Qdrant per ogni utente. Esiste **una collection
sola**, condivisa, dove ogni chunk salvato porta con sé due metadati:

```python
metadata = {
    "user_id": "...",       # a chi appartiene
    "document_id": "...",   # da quale documento proviene
    ...
}
```

Ogni ricerca (`retriever_per_utente` in `src/rag/vectorstore.py`)
applica **sempre** un filtro `user_id == <utente autenticato>`. Il
risultato pratico: anche se due utenti caricano PDF con contenuti simili,
nessuno dei due vedrà mai i chunk dell'altro nelle risposte. Lo stesso
principio protegge `GET/DELETE /documents/{id}`: si controlla sempre che
`documento.user_id == utente.id` prima di restituire o cancellare
qualcosa.

---

## 8. Come avviare il progetto in locale (senza Docker)

Utile per sviluppare e fare piccoli test veloci.

```bash
# 1. Crea un ambiente virtuale Python (isola le dipendenze del progetto)
python3 -m venv .venv
source .venv/bin/activate        # su macOS/Linux

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Copia il file di esempio delle variabili d'ambiente e compilalo
cp .env.example .env
# apri .env e imposta OPENAI_API_KEY e JWT_SECRET (una stringa lunga a caso)

# 4. Avvia Qdrant in locale (serve comunque, anche fuori da Docker
#    compose, a meno di non avere già un'istanza altrove)
docker run -p 6333:6333 qdrant/qdrant

# 5. Avvia il server API (in un altro terminale)
cd src
uvicorn api.main:app --reload --port 8000
```

`--reload` fa ripartire il server automaticamente quando modifichi un
file — comodo in sviluppo, da non usare in produzione.

Apri `http://localhost:8000/docs`: è la documentazione automatica di
FastAPI (Swagger UI), generata dai tipi Pydantic e dai commenti del
codice. Da lì puoi anche provare gli endpoint direttamente dal browser
(pulsante "Authorize" per inserire il token dopo il login).

### Provare gli endpoint da terminale con curl

```bash
# Registrazione (ritorna un token)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"tu@esempio.com","password":"unapassword"}'

# Salva il token ricevuto in una variabile
TOKEN="incolla_qui_il_token"

# Upload di un documento
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@documenti/Scheda_Allenamento_Stefano_Baiardi.pdf"

# Controlla lo stato (ripeti finché non è "ready")
curl http://localhost:8000/documents/<id_ricevuto> \
  -H "Authorization: Bearer $TOKEN"

# Fai una domanda
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"domanda":"quali esercizi per le gambe?"}'
```

---

## 9. Come avviare tutto con Docker (locale)

Il modo più simile a come girerà in produzione. Richiede Docker
installato e avviato (su Mac, se usi Colima invece di Docker Desktop:
`colima start`).

```bash
# Assicurati che .env contenga OPENAI_API_KEY e JWT_SECRET
docker compose up -d --build
```

Questo comando:
1. Legge `docker-compose.yml`, che definisce due servizi:
   - `qdrant`: l'immagine ufficiale di Qdrant, con un volume persistente
     (`qdrant_data`) così i vettori non si perdono se il container si
     riavvia.
   - `api`: costruita dal nostro `Dockerfile`, con un volume
     (`app_data`) per il database SQLite e i file caricati.
2. Passa alla `api` le variabili d'ambiente lette dal tuo file `.env`
   locale (`OPENAI_API_KEY`, `JWT_SECRET`) più altre già cablate nel
   compose (`QDRANT_URL` punta al servizio `qdrant` per nome, non a
   `localhost`, perché dentro Docker i container si raggiungono per nome
   di servizio).

Comandi utili:

```bash
docker compose ps              # stato dei container
docker compose logs -f api     # log in tempo reale dell'API
docker compose down            # ferma e rimuove i container (i volumi restano)
docker compose down -v         # come sopra, ma cancella anche i volumi (DATI PERSI)
```

### Perché il Dockerfile installa torch separatamente

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
```

`sentence-transformers` (usato per l'embedding) dipende da PyTorch
(`torch`). Se lo si installa "normalmente", pip scarica anche il
supporto per GPU NVIDIA (pacchetti CUDA, oltre 1 GB) — inutile su un
server senza GPU come una VM Coolify. Installandolo prima dall'indice
`download.pytorch.org/whl/cpu` otteniamo la sola versione CPU, molto più
leggera e veloce da scaricare/deployare.

---

## 10. Deploy su Coolify

Coolify sa leggere direttamente un `docker-compose.yml`, quindi il
deploy consiste principalmente nel puntarlo al repository e impostare le
variabili d'ambiente sensibili (che NON stanno nel repository).

### Passo 1 — metti il progetto su un repository Git

Se non l'hai già fatto:

```bash
git init
git add .
git commit -m "Backend RAG con FastAPI, Qdrant, JWT"
```

Push su GitHub/GitLab (Coolify supporta entrambi, e anche deploy da Git
generico via webhook).

**Prima di pushare**, controlla che `.env` (quello vero, con le chiavi)
sia elencato in `.gitignore` — lo è già in questo progetto — così le tue
chiavi non finiscono online.

### Passo 2 — crea una nuova risorsa su Coolify

1. Nel pannello Coolify: **New Resource → Docker Compose** (oppure
   "Application" collegata al repo, a seconda della versione di
   Coolify).
2. Collega il repository Git del progetto.
3. Indica `docker-compose.yml` come file di compose (di solito è già il
   default se sta nella root del repo).

### Passo 3 — variabili d'ambiente

Nella sezione **Environment Variables** della risorsa, imposta:

| Nome | Valore | Note |
|---|---|---|
| `OPENAI_API_KEY` | la tua chiave OpenAI | **rigenerala** se è quella vecchia esposta in passato |
| `JWT_SECRET` | una stringa lunga e casuale | genera con `python3 -c "import secrets; print(secrets.token_hex(32))"` |

Queste sono le uniche due che il `docker-compose.yml` si aspetta
dall'ambiente esterno (`${OPENAI_API_KEY}`, `${JWT_SECRET}`); le altre
(`QDRANT_URL`, `DB_PATH`, ecc.) sono già scritte nel compose stesso e non
vanno duplicate su Coolify.

### Passo 4 — volumi persistenti

Il `docker-compose.yml` dichiara già due volumi:

```yaml
volumes:
  qdrant_data:   # i vettori Qdrant
  app_data:      # il database SQLite + i file caricati
```

Coolify li gestisce automaticamente come volumi Docker nominati — **non
serve configurazione aggiuntiva**, ma è importante NON cancellarli
quando fai redeploy, altrimenti perdi tutti i documenti indicizzati e
gli utenti registrati. Controlla nelle impostazioni della risorsa che
l'opzione (spesso chiamata "Persistent Storage" o simile) non sia
disattivata.

### Passo 5 — deploy e verifica

1. Premi **Deploy** su Coolify. La prima build richiede qualche minuto
   (scarica PyTorch e le altre dipendenze, come hai visto in locale).
2. A build completata, Coolify ti assegna un dominio (o ne configuri uno
   tuo). Verifica che il servizio sia vivo:
   ```bash
   curl https://tuo-dominio.coolify.app/health
   # {"status":"ok"}
   ```
3. Ripeti il test end-to-end (register → login → upload → ask) con
   `curl`, sostituendo `http://localhost:8000` con il tuo dominio.

### Note per quando arriverà il frontend

- Il frontend dovrà salvare il token JWT ricevuto da `/auth/login` (es.
  in `localStorage`) e allegarlo come header `Authorization: Bearer
  <token>` a ogni chiamata verso `/documents` e `/ask`.
- Dopo l'upload, il frontend dovrà fare polling su
  `GET /documents/{id}` (ad es. ogni 2 secondi) finché lo stato non è
  `"ready"` o `"error"`, per mostrare all'utente quando il documento è
  pronto da interrogare.
- Se il frontend gira su un dominio diverso dall'API, andrà configurato
  CORS in `src/api/main.py` (FastAPI ha `CORSMiddleware` pronto
  all'uso) — non è ancora presente in questo progetto perché non
  serve finché testi con curl/Swagger.

---

## 11. Domande frequenti

**Perché due database (SQLite + Qdrant) invece di uno solo?**
Sono specializzati per compiti diversi. SQLite è ottimo per dati
strutturati piccoli con relazioni semplici (utenti, stato documenti).
Qdrant è ottimizzato per un'operazione che SQLite non sa fare
efficientemente: "trovami i vettori più simili a questo, tra milioni di
vettori". Usarli entrambi, ognuno per il suo scopo, è più semplice ed
efficiente che forzarli a fare il lavoro dell'altro.

**Cosa succede se il server si riavvia mentre un documento è in
"processing"?**
Al momento, quel documento resta bloccato in `"processing"` per sempre
(il background task viene perso col riavvio del processo). Per un uso
personale con pochi documenti è un limite accettabile; se in futuro
capitasse spesso, la soluzione più semplice sarebbe un piccolo comando
di "retry" che rilancia l'indicizzazione dei documenti rimasti in
`processing` all'avvio dell'app.

**Posso caricare altri formati oltre PDF e TXT?**
Non ancora: `ESTENSIONI_VALIDE` in `src/api/routers/documents.py`
limita a `.pdf` e `.txt`. Per aggiungerne altri (es. `.docx`) servirebbe
estendere anche `src/rag/loaders.py` con la logica di estrazione testo
per quel formato.

**Come cambio il modello LLM o di embedding?**
Sono costanti in `src/rag/config.py` (`MODELLO_LLM`, `MODELLO_EMBED`).
Attenzione: se cambi il modello di embedding, i vettori già salvati su
Qdrant restano nel "vecchio" spazio vettoriale e non sono più
confrontabili con quelli nuovi — servirebbe re-indicizzare tutto da
capo.
