"""Modelli e sessione del database dei metadati (utenti, documenti).

SQLModel è una libreria che unisce due cose in una: Pydantic (validazione
dati, usata anche da FastAPI) e SQLAlchemy (l'ORM, cioè lo strato che
traduce oggetti Python in righe di tabelle SQL e viceversa). Il vantaggio è
che ogni classe che scriviamo qui sotto è CONTEMPORANEAMENTE:
  1. lo schema della tabella nel database (grazie a `table=True`)
  2. un modello dati Python normale, con validazione automatica dei tipi

Nota bene: qui dentro salviamo solo i METADATI (chi è l'utente, come si
chiama il file, a che punto è l'indicizzazione). I vettori/embedding dei
documenti vivono altrove, su Qdrant (vedi vectorstore.py) — sono due
database diversi per due scopi diversi.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, Session, SQLModel, create_engine

from .config import SQLITE_URL


# Funzioni "di fabbrica": generano un valore diverso ogni volta che vengono
# chiamate. Le usiamo sotto con default_factory (vedi Field più in basso),
# perché un default normale (es. id: str = _uuid()) verrebbe calcolato UNA
# SOLA VOLTA quando Python legge la classe, e tutte le righe finirebbero
# con lo stesso id. Con default_factory, invece, la funzione viene invocata
# ogni volta che si crea una nuova riga.
def _uuid() -> str:
    """Genera un identificativo univoco (es. 'a1b2c3d4-...'), usato come id."""
    return str(uuid.uuid4())


def _ora() -> datetime:
    """Restituisce l'istante attuale in UTC (fuso orario universale)."""
    return datetime.now(timezone.utc)


# Un Enum elenca i soli valori ammessi per un campo. Ereditare anche da
# `str` (class StatoDocumento(str, Enum)) fa sì che il valore si comporti
# come una stringa vera e propria quando viene serializzato in JSON dalle
# risposte dell'API (altrimenti FastAPI dovrebbe convertirlo a mano).
class StatoDocumento(str, Enum):
    PROCESSING = "processing"  # appena caricato, indicizzazione in corso
    READY = "ready"            # indicizzato, pronto per essere interrogato
    ERROR = "error"            # qualcosa è andato storto (vedi campo `errore`)


# `SQLModel, table=True` dice: "questa classe è anche una tabella SQL".
# Ogni attributo con un type hint (es. `email: str`) diventa una colonna.
class Utente(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    # unique=True: due utenti non possono avere la stessa email (il database
    # rifiuterebbe l'inserimento). index=True: crea un indice per rendere
    # veloci le ricerche "trova l'utente con questa email".
    email: str = Field(unique=True, index=True)
    password_hash: str  # MAI salvare la password in chiaro, solo il suo hash
    creato_il: datetime = Field(default_factory=_ora)


class Documento(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True)  # a chi appartiene questo documento
    nome_file: str
    stato: StatoDocumento = Field(default=StatoDocumento.PROCESSING)
    errore: str | None = None  # `str | None` = può essere una stringa oppure None
    n_chunk: int = 0
    creato_il: datetime = Field(default_factory=_ora)


# L'"engine" è l'oggetto che sa come parlare fisicamente col database (dove
# si trova il file, come aprire connessioni, ecc.). Lo creiamo una sola
# volta qui, a livello di modulo, e lo riusiamo per tutta la vita dell'app.
# check_same_thread=False è specifico di SQLite: di default SQLite vieta di
# usare la stessa connessione da thread diversi, ma FastAPI può gestire le
# richieste su thread diversi, quindi disattiviamo il controllo (è sicuro
# nel nostro caso perché apriamo una Session nuova per ogni richiesta).
_engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Crea le tabelle nel database se non esistono già.

    Va chiamata una volta all'avvio dell'app (vedi api/main.py). Se le
    tabelle esistono già non fa nulla: è sicuro chiamarla ad ogni avvio.
    """
    SQLModel.metadata.create_all(_engine)


def get_session():
    """Apre una sessione (una specie di "conversazione" col database) e la
    chiude automaticamente alla fine, anche in caso di errore.

    Questa funzione è scritta come GENERATORE: invece di `return`, usa
    `yield`. La differenza pratica: il codice PRIMA di `yield` viene
    eseguito subito, il valore dopo `yield` viene "prestato" a chi ha
    chiamato la funzione, e il codice DOPO yield (qui non c'è, ma sarebbe
    la chiusura della sessione) viene eseguito solo quando chi la usa ha
    finito. Il blocco `with Session(...) as session:` si occupa già di
    chiudere la sessione automaticamente all'uscita del blocco.
    FastAPI usa esattamente questo pattern per le sue "dependency" (vedi
    api/deps.py): una funzione con `yield` gli permette di eseguire del
    codice sia PRIMA che DOPO la richiesta.
    """
    with Session(_engine) as session:
        yield session
