# Tutorial: da RAG stateless ad agente con memoria (LangGraph)

Percorso guidato in 4 fasi per implementare, **nell'ordine giusto**:

1. **Checkpointer + `thread_id`** — memoria per-conversazione nativa di LangGraph
2. **Retrieval come tool** (`ToolNode` + `tools_condition`) — l'LLM decide se/cosa cercare
3. **`Store`** — memoria a lungo termine sull'utente, trasversale alle conversazioni
4. **`interrupt()` + streaming** — human-in-the-loop e output incrementale

Ogni fase ha: cosa cambia, dove metterlo, il codice, **come testarlo**, e cosa
dovresti aver capito. Non passare alla fase successiva se il test non passa.

> **Regola d'oro del tutorial**: lavora su un branch. La fase 2 in particolare
> può peggiorare la qualità delle risposte rispetto ad oggi (vedi §2.0).
>
> ```bash
> git checkout -b langgraph-memoria
> ```

---

## Fase 0 — Preparazione

### 0.1 Dipendenza mancante

`langgraph-checkpoint-sqlite` **non è installato** (hai solo `langgraph-checkpoint`,
che contiene solo `InMemorySaver` e le classi base). Serve per la Fase 1:

```bash
.venv/bin/pip install langgraph-checkpoint-sqlite
```

Aggiungilo a `requirements.txt`, sotto `langgraph>=1.2`:

```
langgraph-checkpoint-sqlite>=2.0
```

### 0.2 Verifica di partenza

Prima di toccare qualsiasi cosa, assicurati che l'app funzioni **adesso**:

```bash
.venv/bin/uvicorn api.main:app --app-dir src --reload
```

In un altro terminale, prendi un token e fai una domanda di controllo. Tieni
questo token da parte: lo riuserai in ogni fase.

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=TUA_EMAIL&password=TUA_PASSWORD' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
echo $TOKEN

curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"cosa mangio lunedì?","storico":[]}' | python3 -m json.tool
```

Se questo non funziona, **fermati qui**: non vuoi debuggare due problemi insieme.

### 0.3 La mappa mentale

Il punto di partenza, oggi:

| Pezzo | Dove | Stato |
|---|---|---|
| Grafo `recupera → decidi → (riformula ↺ / genera)` | `src/rag/graph.py:113` | ✅ c'è |
| Memoria conversazionale | `frontend/src/lib/chat-context.tsx:18` (`useState`) | ⚠️ nel client, volatile |
| `riformula_con_storico` | `src/api/routers/ask.py:101` — **fuori** dal grafo | ⚠️ posizione sbagliata |
| Checkpointer | — | ❌ `compile()` senza argomenti (`graph.py:173`) |
| Tool | — | ❌ nessuno, il retrieval è una chiamata fissa |
| Store | — | ❌ |
| Alexa con memoria | `src/api/routers/alexa.py:48` | ❌ nemmeno il campo `storico` |

Alla fine delle 4 fasi tutte le righe saranno ✅.

---

## Fase 1 — Checkpointer e `thread_id`

### 1.0 Il concetto

Oggi passi lo storico **come dato dentro lo stato**: il frontend lo rimanda ad
ogni richiesta, `ask.py:94` lo tronca, `ask.py:107` lo appiattisce in stringa.

Un **checkpointer** salva lo stato del grafo dopo ogni nodo, indicizzato per
`thread_id`. Alla chiamata successiva con lo stesso `thread_id`, LangGraph
**ricarica lo stato da solo**. Non passi più lo storico: passi un identificatore.

Due concetti da tenere distinti fin da subito:

- **`thread_id`** → una conversazione. È la Fase 1.
- **`user_id`** → la persona, che ha molte conversazioni. È la Fase 3 (`Store`).

### 1.1 Lo stato deve diventare accumulabile

Problema: `Stato` (`graph.py:105`) ha `domanda: str`, che viene **sovrascritta**
ad ogni turno. Per accumulare i messaggi serve un campo con un *reducer* — una
funzione che dice a LangGraph come **combinare** il vecchio valore col nuovo
invece di rimpiazzarlo. `add_messages` è il reducer standard per i messaggi.

In `src/rag/graph.py`, aggiungi agli import:

```python
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
```

e modifica lo stato:

```python
class Stato(TypedDict):
    # `Annotated[..., add_messages]` è il REDUCER: quando un nodo ritorna
    # {"messaggi": [x]}, LangGraph non sostituisce la lista ma ci APPENDE x.
    # È ciò che rende lo stato una conversazione che cresce invece di un
    # singolo turno che si sovrascrive.
    messaggi: Annotated[list[AnyMessage], add_messages]
    domanda: str
    documenti: list[Document]
    risposta: str
    tentativi: int
```

Nota che `storico: str` sparisce: da ora la conversazione **è** `messaggi`.

### 1.2 `riformula_con_storico` entra nel grafo

Oggi è chiamata da `ask.py:101`, cioè fuori dal grafo — è il motivo per cui ogni
client deve reimplementare la memoria. Diventa un nodo.

In `costruisci_grafo`, aggiungi come **primo** nodo:

```python
    # NODO: rende autonoma una domanda ellittica usando i messaggi precedenti
    # già presenti nello stato (caricati dal checkpointer, non passati dal client).
    def nodo_contestualizza(stato: Stato):
        precedenti = stato["messaggi"][:-1]  # esclude la domanda appena arrivata
        domanda = stato["messaggi"][-1].content
        if not precedenti:
            return {"domanda": domanda}
        scambi = "\n".join(f"{m.type}: {m.content}" for m in precedenti[-MAX_STORICO:])
        nuova = llm.invoke(
            "Questa è una conversazione tra un utente e un assistente su dieta e "
            "allenamento. Riscrivi l'ULTIMA domanda dell'utente come domanda "
            "autonoma e completa, esplicitando ciò a cui si riferisce implicitamente. "
            "Se è già autonoma, restituiscila invariata. Rispondi SOLO con la domanda.\n\n"
            f"CONVERSAZIONE PRECEDENTE:\n{scambi}\n\n"
            f"ULTIMA DOMANDA: {domanda}"
        ).content.strip()
        print(f"   ↳ contestualizzata: {nuova}")
        return {"domanda": nuova}
```

Serve l'import `from .config import MAX_STORICO` in cima a `graph.py`.

`nodo_genera` deve restituire anche il messaggio dell'assistente, così finisce
nello stato persistito:

```python
    def nodo_genera(stato: Stato):
        contesto = "\n\n".join(d.page_content for d in stato["documenti"])
        precedenti = stato["messaggi"][:-1]
        testo_storico = "\n".join(f"{m.type}: {m.content}" for m in precedenti[-MAX_STORICO:])
        risposta = (prompt | llm | StrOutputParser()).invoke({
            "context": contesto,
            "question": stato["domanda"],
            "oggi": giorno_oggi(),
            "storico": testo_storico,
        })
        # Il messaggio dell'assistente entra nello stato: al turno successivo
        # sarà già lì, senza che nessun client lo rimandi.
        return {"risposta": risposta, "messaggi": [AIMessage(content=risposta)]}
```

(import: `from langchain_core.messages import AIMessage`)

E ricabla l'ingresso:

```python
    builder.add_node("contestualizza", nodo_contestualizza)
    builder.add_edge(START, "contestualizza")
    builder.add_edge("contestualizza", "recupera")
```

(rimuovi il vecchio `builder.add_edge(START, "recupera")`)

### 1.3 Il checkpointer

`SqliteSaver.from_conn_string()` è un **context manager**: chiude la connessione
all'uscita del `with`, quindi non va bene per un'app long-running. Usa il
costruttore diretto.

In `src/rag/config.py`:

```python
# Database dei checkpoint LangGraph (stato delle conversazioni). File separato
# da DB_PATH: sono dati con ciclo di vita diverso (si possono cancellare senza
# perdere utenti e documenti) e schema gestito da LangGraph, non da noi.
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "checkpoints.sqlite")
```

In `src/api/state.py`:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from rag.config import CHECKPOINT_DB


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Salva lo stato del grafo dopo ogni nodo, indicizzato per thread_id.

    NON usiamo SqliteSaver.from_conn_string(): è un context manager che chiude
    la connessione all'uscita del `with`, adatto agli script, non a un server
    che deve tenerla aperta per tutta la vita del processo.

    check_same_thread=False: FastAPI serve le richieste su thread diversi
    (stesso motivo dell'engine in rag/db.py).
    """
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # crea le tabelle dei checkpoint se non esistono
    return saver
```

E in `graph.py`, `costruisci_grafo` accetta il checkpointer:

```python
def costruisci_grafo(retriever, llm, prompt, checkpointer=None):
    ...
    return builder.compile(checkpointer=checkpointer)
```

Il default `None` mantiene funzionante la CLI (`src/rag/cli.py`) senza modifiche.

### 1.4 L'endpoint

Riscrivi il corpo di `ask` in `src/api/routers/ask.py`:

```python
class DomandaIn(BaseModel):
    domanda: str
    # Identifica la conversazione. Il client non manda più i messaggi: manda
    # solo QUALE conversazione, e il checkpointer ricarica il resto.
    conversation_id: str | None = None


class RispostaOut(BaseModel):
    risposta: str
    conversation_id: str


@router.post("/ask", response_model=RispostaOut)
def ask(payload: DomandaIn, utente: UtenteCorrente):
    embeddings = get_embeddings()
    retriever = retriever_per_utente(embeddings, utente.id, K)
    llm = get_llm()
    grafo = costruisci_grafo(retriever, llm, _PROMPT, checkpointer=get_checkpointer())

    # Se il client non manda un id, ne creiamo uno: è l'inizio di una nuova
    # conversazione. Lo restituiamo così il client sa cosa rimandare dopo.
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    # Il thread_id è namespaced sull'utente: senza il prefisso, un utente che
    # indovina l'id di una conversazione altrui potrebbe leggerla.
    config = {"configurable": {"thread_id": f"{utente.id}:{conversation_id}"}}

    risultato = grafo.invoke(
        {"messaggi": [HumanMessage(content=payload.domanda)], "tentativi": 0},
        config,
    )
    return RispostaOut(risposta=risultato["risposta"], conversation_id=conversation_id)
```

Import da aggiungere: `import uuid`, `from langchain_core.messages import HumanMessage`,
`from ..state import get_checkpointer`. Rimuovi `riformula_con_storico` dagli
import (ora è un nodo) e le classi `MessaggioIn`/il campo `storico`.

> **Nota sicurezza** — il prefisso `utente.id:` sul `thread_id` non è un
> dettaglio estetico: è ciò che impedisce a un utente di leggere la
> conversazione di un altro passando un `conversation_id` indovinato.

### 1.5 Alexa, finalmente con memoria

In `src/api/routers/alexa.py` — è il punto dove la Fase 1 ripaga di più, perché
Alexa oggi non ha **nessuno** storico:

```python
@router.post("/ask", response_model=RispostaOut)
def alexa_ask(payload: DomandaIn, user_id: AlexaUserId):
    embeddings = get_embeddings()
    retriever = retriever_per_utente(embeddings, user_id, K)
    llm = get_llm()
    grafo = costruisci_grafo(retriever, llm, _PROMPT, checkpointer=get_checkpointer())

    # Thread fisso: la skill la usa una persona sola e non ha modo di gestire
    # un id di conversazione. Una sola conversazione lunga è esattamente il
    # comportamento che ci si aspetta da un assistente vocale.
    config = {"configurable": {"thread_id": f"{user_id}:alexa"}}
    risultato = grafo.invoke(
        {"messaggi": [HumanMessage(content=payload.domanda)], "tentativi": 0}, config
    )
    return RispostaOut(risposta=risultato["risposta"])
```

### 1.6 Frontend

`frontend/src/lib/api.ts`:

```ts
export function ask(domanda: string, conversationId: string | null) {
  return request<{ risposta: string; conversation_id: string }>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ domanda, conversation_id: conversationId }),
  })
}
```

`frontend/src/lib/chat-context.tsx` — aggiungi `conversationId` accanto a `messages`:

```tsx
const [conversationId, setConversationId] = useState<string | null>(null)
```

esponilo nel context, e in `chat.tsx` salvalo da `data.conversation_id` in
`onSuccess`, passandolo in `mutation.mutate`. I `messages` restano nel client,
ma **solo per disegnare la UI** — non sono più la fonte di verità.

> **Bonus opzionale**: persisti `conversationId` in `localStorage` e aggiungi un
> endpoint `GET /conversations/{id}` che legge `grafo.get_state(config).values["messaggi"]`.
> A quel punto la chat sopravvive al refresh — cosa oggi impossibile.

### 1.7 ✅ Test della Fase 1

**Test A — la memoria esiste lato server.** Il secondo comando **non** manda
nessuno storico: se risponde correttamente, la memoria è nel backend.

```bash
CID=$(curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"cosa mangio lunedì?"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["conversation_id"])')
echo "conversazione: $CID"

curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"domanda\":\"e a cena?\",\"conversation_id\":\"$CID\"}" | python3 -m json.tool
```

✅ Atteso: risponde sulla cena **di lunedì**. Nei log di uvicorn deve comparire
`↳ contestualizzata: ...` con la domanda riscritta esplicitamente.

**Test B — persistenza reale.** Ferma uvicorn (`Ctrl+C`), riavvialo, e rifai la
seconda domanda con lo stesso `$CID`.

✅ Atteso: risponde ancora correttamente. Questa è la differenza tra
`InMemorySaver` e `SqliteSaver` — se fallisce, stai usando quello sbagliato.

**Test C — ispeziona lo stato.** È il momento in cui il concetto diventa concreto:

```bash
.venv/bin/python -c "
import sys, sqlite3; sys.path.insert(0,'src')
conn = sqlite3.connect('checkpoints.sqlite')
for row in conn.execute('SELECT DISTINCT thread_id FROM checkpoints'):
    print(row)
"
```

✅ Atteso: vedi i tuoi `thread_id` nella forma `<user_id>:<conversation_id>`.

**Test D — isolamento.** Registra un secondo utente, prendi il suo token e prova
a usare il `$CID` del primo.

✅ Atteso: non vede la conversazione dell'altro (thread_id diverso per prefisso).

### 1.8 Cosa hai imparato

- Un **reducer** (`add_messages`) definisce come lo stato si combina, non solo cosa contiene
- Il **checkpointer** rende il grafo ripristinabile: stessa `config`, stato ricaricato
- `thread_id` è la chiave della memoria per-conversazione
- Spostare `riformula` dentro il grafo l'ha resa disponibile a **tutti** i client insieme

---

## Fase 2 — Il retrieval come tool

### 2.0 Avvertenza, prima di iniziare

Il tuo `RetrieverIbrido` è tarato bene: boost sul giorno, boost sulla spesa,
regex deterministiche. Rendere il retrieval un tool significa **lasciar decidere
all'LLM**, e quasi certamente qualche risposta peggiorerà.

Va benissimo per imparare — è *il* pattern agentico di LangGraph — ma:

```bash
git checkout -b langgraph-tools   # branch separato dalla Fase 1
```

Tieni la Fase 1 su un branch che puoi mergiare anche se la Fase 2 non ti convince.

### 2.1 Il concetto

Oggi `nodo_recupera` chiama **sempre** il retriever, una volta, con la domanda.
Con un tool, il flusso diventa: l'LLM riceve la domanda e la lista dei tool, e
**decide** se chiamarne uno, quale, e con che argomenti — eventualmente più volte.

Guadagni: l'agente può cercare due volte con query diverse ("dieta di lunedì" +
"lista della spesa"), o non cercare affatto se la domanda non lo richiede.

### 2.2 Definisci i tool

Nuovo file `src/rag/tools.py`:

```python
"""Tool esposti all'LLM nel grafo agentico.

Differenza rispetto a nodo_recupera: lì il retrieval è una tappa OBBLIGATA del
flusso, qui è una capacità che il modello può decidere di usare — anche più
volte, con query diverse. La docstring di ogni tool non è documentazione per
noi: è il testo che l'LLM legge per decidere se e come chiamarlo, quindi va
scritta pensando a lui.
"""

from langchain_core.tools import tool


def crea_tool_ricerca(retriever):
    """Fabbrica: il retriever è già filtrato per utente, quindi il tool va
    creato per-richiesta (non può essere una costante di modulo)."""

    @tool
    def cerca_documenti(query: str) -> str:
        """Cerca informazioni nei documenti personali dell'utente su dieta,
        piano alimentare, lista della spesa e schede di allenamento.

        Usa query specifiche e ricche di parole chiave. Se la domanda riguarda
        sia i pasti sia cosa comprare, fai DUE ricerche separate.

        Args:
            query: cosa cercare, es. "colazione lunedì" o "lista spesa verdura"
        """
        docs = retriever.invoke(query)
        if not docs:
            return "Nessun documento trovato per questa ricerca."
        return "\n\n".join(d.page_content for d in docs)

    return cerca_documenti
```

### 2.3 Il grafo agentico

Nuova funzione in `graph.py` — **affiancala** a `costruisci_grafo`, non
sostituirla, così puoi confrontare i due comportamenti:

```python
def costruisci_grafo_agente(retriever, llm, system_prompt: str, checkpointer=None):
    """Variante agentica: l'LLM decide se e quante volte cercare.

    Il ciclo `agente → tools → agente` è il pattern ReAct: il modello ragiona,
    chiama un tool, legge il risultato, e decide se gli basta o se cercare
    ancora. `tools_condition` è il bivio già pronto: guarda l'ultimo messaggio
    e instrada verso "tools" se contiene una tool call, verso END altrimenti.
    """
    from langgraph.prebuilt import ToolNode, tools_condition
    from .tools import crea_tool_ricerca

    tools = [crea_tool_ricerca(retriever)]
    # bind_tools: comunica al modello quali funzioni può chiamare. Senza questo
    # l'LLM non sa che i tool esistono e risponde solo a parole.
    llm_con_tools = llm.bind_tools(tools)

    class StatoAgente(TypedDict):
        messaggi: Annotated[list[AnyMessage], add_messages]

    def nodo_agente(stato: StatoAgente):
        messaggi = [SystemMessage(content=system_prompt.format(oggi=giorno_oggi()))]
        messaggi += stato["messaggi"]
        return {"messaggi": [llm_con_tools.invoke(messaggi)]}

    builder = StateGraph(StatoAgente)
    builder.add_node("agente", nodo_agente)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agente")
    builder.add_conditional_edges("agente", tools_condition)
    builder.add_edge("tools", "agente")  # torna all'agente col risultato
    return builder.compile(checkpointer=checkpointer)
```

(import: `from langchain_core.messages import SystemMessage`)

Nota che `tools_condition` funziona **solo** se il campo dei messaggi si chiama
`messages`; il tuo si chiama `messaggi`, quindi passa `messages_key`:

```python
    builder.add_conditional_edges(
        "agente", lambda s: tools_condition(s, messages_key="messaggi"),
        {"tools": "tools", "__end__": END},
    )
```

Stesso discorso per `ToolNode`: `ToolNode(tools, messages_key="messaggi")`.

> Se preferisci evitare l'attrito, chiama il campo `messages` in `StatoAgente`.
> È un buon momento per notare quanto le convenzioni dei prebuilt siano
> vincolanti — è una lezione sul framework, non un dettaglio.

### 2.4 Esporlo dietro un flag

In `ask.py`, invece di sostituire, aggiungi un campo per scegliere:

```python
class DomandaIn(BaseModel):
    domanda: str
    conversation_id: str | None = None
    agente: bool = False  # True → grafo con tool, False → grafo deterministico
```

e nell'endpoint scegli quale costruire. Così confronti le due modalità **sulla
stessa domanda** senza cambiare branch.

### 2.5 ✅ Test della Fase 2

**Test A — il tool viene chiamato.** Aggiungi un `print` dentro `cerca_documenti`
e osserva i log:

```bash
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"cosa mangio lunedì?","agente":true}' | python3 -m json.tool
```

✅ Atteso: il tool viene invocato almeno una volta.

**Test B — ricerche multiple.** La domanda che oggi richiede il boost speciale:

```bash
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"cosa devo comprare per i pasti di lunedì?","agente":true}' | python3 -m json.tool
```

✅ Atteso: **due** chiamate al tool (pasti + spesa). È il comportamento che oggi
ottieni con un'euristica scritta a mano in `vectorstore.py`.

**Test C — confronto onesto.** Stessa domanda con `"agente": false` e `true`.
Annota quale risponde meglio. Non dare per scontato che vinca l'agente.

**Test D — nessuna chiamata.** Chiedi `"ciao, come funzioni?"`.
✅ Atteso: risponde senza chiamare il tool.

### 2.6 Cosa hai imparato

- La **docstring di un tool è un prompt**, non documentazione
- `bind_tools` + `ToolNode` + `tools_condition` = il ciclo ReAct
- I prebuilt impongono convenzioni (`messages_key`): comodità in cambio di libertà
- Un agente che decide è più flessibile e **meno prevedibile** di una pipeline

---

## Fase 3 — `Store`: memoria a lungo termine

### 3.0 Il concetto (la distinzione che conta)

|  | Checkpointer (Fase 1) | Store (Fase 3) |
|---|---|---|
| Cosa salva | i messaggi di **una** conversazione | fatti sull'**utente** |
| Chiave | `thread_id` | namespace, es. `("memorie", user_id)` |
| Vive | quanto la conversazione | per sempre, tra tutte le conversazioni |
| Esempio | "hai chiesto di lunedì" | "è intollerante al lattosio" |

Questa è la tua idea di "agente che si ricorda dell'utente". **Non** è lo storico.

### 3.1 Lo store

In `src/api/state.py`:

```python
from langgraph.store.memory import InMemoryStore

@lru_cache(maxsize=1)
def get_store() -> InMemoryStore:
    """Memoria a lungo termine, trasversale alle conversazioni.

    InMemoryStore si perde al riavvio: va bene per imparare l'API. Per
    persistere davvero serve uno store SQL/Postgres — passaggio successivo,
    ma l'interfaccia (put/get/search) resta identica.
    """
    return InMemoryStore()
```

Passalo a `compile(checkpointer=..., store=get_store())`.

### 3.2 Tool per scrivere e leggere le memorie

In `src/rag/tools.py`:

```python
from langgraph.prebuilt import InjectedStore
from typing import Annotated
from langgraph.store.base import BaseStore
import uuid


def crea_tool_memoria(user_id: str):
    @tool
    def ricorda(fatto: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Salva un'informazione duratura e personale sull'utente: preferenze,
        intolleranze, obiettivi, abitudini di allenamento.

        NON usarlo per fatti già presenti nei documenti né per domande estemporanee.

        Args:
            fatto: il fatto da ricordare, in una frase, es. "è intollerante al lattosio"
        """
        # InjectedStore: lo store arriva a runtime da LangGraph e NON è un
        # parametro che l'LLM deve riempire — non compare nello schema del tool.
        store.put(("memorie", user_id), str(uuid.uuid4()), {"fatto": fatto})
        return f"Ricordato: {fatto}"

    return ricorda
```

Per **leggere** le memorie, non serve un tool: iniettale nel system prompt del
`nodo_agente`, così sono sempre disponibili senza una chiamata in più:

```python
    def nodo_agente(stato, *, store):
        memorie = store.search(("memorie", user_id), limit=10)
        blocco = "\n".join(f"- {m.value['fatto']}" for m in memorie)
        testo = system_prompt.format(oggi=giorno_oggi())
        if blocco:
            testo += f"\n\nCOSE CHE SAI SULL'UTENTE:\n{blocco}"
        ...
```

### 3.3 ✅ Test della Fase 3

**Test A — scrittura.**

```bash
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"ricorda che sono intollerante al lattosio","agente":true}' | python3 -m json.tool
```

**Test B — lettura in una conversazione DIVERSA.** È il test che conta: **non**
passare `conversation_id`, così parte un thread nuovo.

```bash
curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"domanda":"cosa posso mangiare a colazione?","agente":true}' | python3 -m json.tool
```

✅ Atteso: tiene conto del lattosio pur essendo una conversazione nuova. Se
funzionasse solo nello stesso thread, staresti testando il checkpointer, non lo store.

**Test C — isolamento tra utenti.** Col token del secondo utente, verifica che
la memoria del primo **non** sia visibile (namespace diverso).

**Test D — il rischio.** Fai salvare un fatto sbagliato di proposito
("ricorda che mi alleno il lunedì" quando non è vero) e osserva come inquina le
risposte successive senza essere visibile in chat.

> Questo è **il** problema della memoria agentica: un fatto errato si
> auto-perpetua e non compare nella conversazione. Per questo, in un'app vera,
> le memorie vanno rese **ispezionabili e cancellabili** dall'utente. Considera
> un `GET /memorie` + `DELETE /memorie/{id}` come esercizio finale.

### 3.4 Cosa hai imparato

- Checkpointer e Store sono **due assi diversi**: per-thread vs cross-thread
- `InjectedStore` inietta dipendenze runtime senza esporle all'LLM
- I namespace sono il confine di sicurezza tra utenti
- La memoria a lungo termine introduce un rischio che lo storico non ha

---

## Fase 4 — `interrupt()` e streaming

### 4.1 Streaming (il più utile dei due)

`.stream()` invece di `.invoke()` ti fa vedere lo stato evolvere nodo per nodo:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
# ... costruisci grafo e config come nell'endpoint ...
for step in grafo.stream({'messaggi':[HumanMessage(content='cosa mangio lunedì?')],'tentativi':0}, config, stream_mode='values'):
    print('---', list(step.keys()))
"
```

Prova i vari `stream_mode`: `values` (stato completo), `updates` (solo il delta
per nodo), `messages` (token per token). Per l'API servirebbe SSE — è il passo
naturale ma è lavoro frontend, non LangGraph.

### 4.2 `interrupt()` — human-in-the-loop

Mette in pausa il grafo e restituisce il controllo. Esempio sensato qui: chiedere
conferma prima di salvare una memoria permanente.

```python
from langgraph.types import interrupt, Command

def nodo_conferma(stato):
    # interrupt() SOSPENDE l'esecuzione: il valore torna al chiamante e lo
    # stato resta congelato nel checkpointer. Richiede un checkpointer —
    # senza, non c'è dove salvare la sospensione.
    risposta = interrupt({"domanda": f"Salvo '{stato['fatto']}'?"})
    if risposta != "sì":
        return {"messaggi": [AIMessage(content="Ok, non salvo nulla.")]}
    ...
```

Si riprende con `grafo.invoke(Command(resume="sì"), config)`.

**✅ Test**: invoca il grafo, verifica che ritorni senza aver completato, controlla
`grafo.get_state(config).next` (mostra il nodo in attesa), poi riprendi con
`Command(resume=...)` e verifica che finisca.

### 4.3 Time travel (bonus, gratis)

Avendo il checkpointer, hai già questo:

```python
for stato in grafo.get_state_history(config):
    print(stato.config["configurable"]["checkpoint_id"], stato.next)
```

Puoi ripartire da un checkpoint passato passando il suo `checkpoint_id` nella
config: rigiochi la conversazione da un punto qualsiasi. È la feature che
giustifica da sola il checkpointer.

---

## Checklist finale

- [ ] **F0** `langgraph-checkpoint-sqlite` installato e in `requirements.txt`
- [ ] **F1** `add_messages` come reducer nello stato
- [ ] **F1** `riformula` spostata **dentro** il grafo come nodo
- [ ] **F1** `SqliteSaver` con costruttore diretto (non `from_conn_string`)
- [ ] **F1** `thread_id` prefissato con `user_id`
- [ ] **F1** Alexa usa il checkpointer → ha memoria per la prima volta
- [ ] **F1** Test B passa (memoria sopravvive al riavvio)
- [ ] **F2** Tool con docstring scritta per l'LLM
- [ ] **F2** `ToolNode` + `tools_condition` con `messages_key` corretto
- [ ] **F2** Confronto onesto agente vs deterministico
- [ ] **F3** Store con namespace per utente
- [ ] **F3** Test B passa (memoria visibile in una conversazione **nuova**)
- [ ] **F3** Verificato il rischio del fatto sbagliato
- [ ] **F4** Provati i `stream_mode`
- [ ] **F4** `interrupt` + `Command(resume=...)`

## Se qualcosa non funziona

| Sintomo | Causa probabile |
|---|---|
| `No module named 'langgraph.checkpoint.sqlite'` | manca il pacchetto (§0.1) |
| Memoria persa al riavvio | stai usando `InMemorySaver`, non `SqliteSaver` |
| Memoria persa tra richieste | manca la `config` con `thread_id` in `invoke` |
| `ValueError: Checkpointer requires...` | `invoke` senza `config` su un grafo compilato col checkpointer |
| `tools_condition` non instrada | il campo si chiama `messaggi`, serve `messages_key` (§2.3) |
| I messaggi si sovrascrivono | manca `Annotated[..., add_messages]` |
| Lo store è vuoto ad ogni riavvio | è `InMemoryStore`, atteso (§3.1) |
| `interrupt` non sospende | manca il checkpointer |
