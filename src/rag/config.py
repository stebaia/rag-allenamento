"""Configurazione centrale: percorsi, modelli, parametri di retrieval.

Questo file NON contiene logica, solo costanti. Tenerle tutte qui invece che
sparse nel codice significa che per cambiare, ad esempio, il modello LLM
basta modificare una riga in un posto solo, invece di cercarla in 10 file.

Le costanti scritte in MAIUSCOLO (convenzione Python, non obbligo del
linguaggio) segnalano "questo valore non cambia durante l'esecuzione".
"""

import os

# --- Parametri del progetto RAG (usati da cli.py e da src/api) ---
CARTELLA_DOCUMENTI = "documenti"  # cartella letta dalla CLI (vedi cli.py)
MODELLO_EMBED = "paraphrase-multilingual-MiniLM-L12-v2"  # locale, gratis
MODELLO_LLM = "gpt-4o-mini"
MODELLO_CONTESTO = os.environ.get("MODELLO_CONTESTO", "gpt-4o-mini")
K = 5  # blocchi recuperati
MAX_STORICO = 8  # messaggi di chat precedenti passati ad /ask come contesto

# --- Reranking (spento di default) ---
# Il RetrieverIbrido ordina i candidati sommando similarità coseno e
# punteggio lessicale: due numeri calcolati separatamente per domanda e
# chunk, che non si "vedono" mai insieme. Un cross-encoder legge invece la
# COPPIA (domanda, chunk) in un unico passaggio, e riconosce che un chunk
# risponde alla domanda anche quando non ne condivide le parole né una
# somiglianza semantica generica. Serve soprattutto sui documenti senza
# boost dedicato (CCNL, codici), dove il ranking non ha altri appigli.
#
# SPENTO di default: chi non lo configura non ne paga nulla — né i ~2,3 GB di
# download al primo avvio, né la RAM, né 1-3 s di latenza per domanda. Si
# accende con RERANKER=on, e allora vale la pena impostare anche HF_HOME su un
# volume persistente (vedi docker-compose.yaml).
#
# Il default è "off" e non "on" per un motivo preciso: questa è l'unica riga
# che decide se un deploy si porta dietro un modello da 2,3 GB. Un componente
# opzionale che si attiva da solo non è opzionale, ed è esattamente il caso in
# cui un ambiente con poca RAM o banda lenta degrada senza che nessuno abbia
# chiesto nulla.
RERANKER_ATTIVO = os.environ.get("RERANKER", "off").strip().lower() in ("on", "1", "true")
MODELLO_RERANKER = os.environ.get("MODELLO_RERANKER", "BAAI/bge-reranker-v2-m3")

# Nomi file che vengono trattati come liste (chunking compatto invece che per-pasto)
KEEP_COMPACT = ("spesa", "lista", "shopping", "grocery")

# --- Configurazione letta dall'ambiente (usata da src/api) ---
# os.environ.get("NOME", default) legge la variabile d'ambiente NOME.
# Se non è impostata (es. in locale, senza Docker), usa il valore di default
# indicato come secondo argomento. Così lo stesso codice funziona sia in
# locale (con i default) sia in Docker/Coolify (dove passiamo valori diversi
# tramite le variabili d'ambiente del container, vedi docker-compose.yml).
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documenti")

DB_PATH = os.environ.get("DB_PATH", "app.db")
# f"..." è una f-string: inserisce il valore di DB_PATH dentro la stringa.
# Il risultato è qualcosa come "sqlite:///app.db", il formato di indirizzo
# (URL) che SQLAlchemy/SQLModel si aspettano per aprire un database SQLite.
SQLITE_URL = f"sqlite:///{DB_PATH}"

CARTELLA_UPLOAD = os.environ.get("CARTELLA_UPLOAD", "uploads")

# Il segreto usato per firmare i token JWT (vedi rag/auth.py). Di proposito
# NON ha un default sensato: se manca, deve essere un errore esplicito
# (vedi il controllo in auth.py), non un segreto debole scelto per noi.
JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# --- Skill Alexa: la Lambda della skill chiama il nostro endpoint /alexa/ask
# con una chiave segreta condivisa (niente JWT: Alexa non fa login). Stesso
# principio di JWT_SECRET: nessun default reale, deve essere impostata a
# mano nell'ambiente. ALEXA_USER_ID è l'utente RAG a cui associare tutte le
# domande fatte tramite la skill (la usi solo tu, quindi è fissa).
ALEXA_API_KEY = os.environ.get("ALEXA_API_KEY", "")
ALEXA_USER_ID = os.environ.get("ALEXA_USER_ID", "")

CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "checkpoints.sqlite")

# Memoria a lungo termine (lo Store), tenuta separata dai checkpoint: sono due
# assi diversi — i checkpoint sono per-conversazione e si possono buttare, le
# memorie valgono per l'utente e devono sopravvivere.
STORE_DB = os.environ.get("STORE_DB", "store.sqlite")

