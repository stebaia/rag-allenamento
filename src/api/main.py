"""Entry point dell'API FastAPI.

Questo è il file che "assembla" l'applicazione: crea l'oggetto FastAPI,
collega i router (auth, documents, ask) e definisce cosa deve succedere
all'avvio del server. È anche il file che passiamo a uvicorn per far
partire il server (vedi il comando nel Dockerfile e nel tutorial).
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# load_dotenv() legge il file .env nella cartella corrente e imposta le
# variabili trovate come variabili d'ambiente del processo (le stesse lette
# da os.environ.get in rag/config.py). Va chiamato PRIMA di importare
# qualsiasi modulo che legga quelle variabili al momento dell'import (come
# rag.auth, che controlla subito JWT_SECRET) — per questo sta qui in cima,
# prima degli import successivi. In produzione su Docker/Coolify le
# variabili vengono di solito impostate direttamente dalla piattaforma, e
# load_dotenv() semplicemente non trova un file .env e non fa nulla.
load_dotenv()

from rag.db import init_db  # noqa: E402 - va importato dopo load_dotenv

from .routers import alexa, ask, auth, documents  # noqa: E402
from .state import get_embeddings  # noqa: E402


# Un "lifespan" è codice che gira attorno a tutta la vita del server: la
# parte prima di `yield` viene eseguita all'avvio, quella dopo (qui non
# c'è) alla chiusura. Lo usiamo per creare le tabelle del database e
# caricare in memoria il modello di embedding una volta sola quando il
# server parte, prima di accettare richieste.
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # get_embeddings() è decorata con @lru_cache (vedi state.py): la prima
    # chiamata carica il modello da disco (alcune decine di secondi), le
    # successive ritornano subito lo stesso oggetto già in memoria.
    # Chiamandola qui, quel costo si paga una volta all'avvio del
    # container invece che sulla prima richiesta di un utente reale.
    get_embeddings()
    yield


app = FastAPI(title="RAG Allenamento API", lifespan=lifespan)

# Il frontend gira su un dominio diverso da questa API (deploy separato su
# Coolify), quindi il browser blocca le richieste finché non abilitiamo
# esplicitamente CORS. ORIGINI_CONSENTITE viene da una variabile d'ambiente
# (lista separata da virgole) per poter puntare al dominio giusto in
# produzione senza modificare il codice.
origini_consentite = [
    origine.strip()
    for origine in os.environ.get("ORIGINI_CONSENTITE", "http://localhost:3000").split(",")
    if origine.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origini_consentite,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ogni router porta con sé il proprio "prefix" (es. /auth), quindi qui
# basta agganciarli tutti all'app principale.
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(alexa.router)


@app.get("/health")
def health():
    """Endpoint minimale per verificare che il server sia vivo e risponda.

    Utile per Coolify/Docker: la piattaforma può chiamare periodicamente
    questo URL per capire se il container va riavviato.
    """
    return {"status": "ok"}
