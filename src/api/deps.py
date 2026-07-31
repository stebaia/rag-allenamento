"""Dependency FastAPI condivise: sessione DB e utente corrente da JWT.

FastAPI ha un meccanismo chiamato "Dependency Injection": invece di
scrivere dentro OGNI endpoint il codice per aprire una sessione DB o
verificare il token, dichiariamo queste operazioni una volta come
"dependency" (funzioni normali) e le "chiediamo in prestito" negli
endpoint semplicemente elencandole come parametri. FastAPI le esegue da
solo, nell'ordine giusto, prima di chiamare la funzione dell'endpoint.

`Annotated[X, Depends(f)]` è la sintassi per dire: "questo parametro è di
tipo X, e il suo valore va ottenuto chiamando la dependency f". Definiamo
degli alias (SessionDep, UtenteCorrente) così negli endpoint scriviamo solo
`session: SessionDep` invece di ripetere `Annotated[Session, Depends(...)]`
ogni volta — pura leggibilità, il comportamento è identico.
"""

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from rag.auth import decodifica_access_token
from rag.config import ALEXA_API_KEY, ALEXA_USER_ID
from rag.db import Utente, get_session

# OAuth2PasswordBearer è una utility di FastAPI che: 1) sa estrarre il
# token dall'header "Authorization: Bearer <token>" delle richieste, e
# 2) permette al pulsante "Authorize" della documentazione automatica
# (Swagger UI su /docs) di sapere che deve chiedere un login. `tokenUrl`
# indica solo QUALE endpoint genera il token, a scopo di documentazione.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Alias riusabile: "dammi una Session aperta tramite get_session()".
SessionDep = Annotated[Session, Depends(get_session)]


def utente_corrente(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep,
) -> Utente:
    """Dependency che verifica il JWT e ritorna l'utente autenticato.

    Nota come questa funzione USA a sua volta altre due dependency
    (oauth2_scheme e SessionDep) come parametri: le dependency possono
    dipendere l'una dall'altra, FastAPI risolve tutta la catena da solo.
    """
    errore_auth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token non valido o scaduto",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user_id = decodifica_access_token(token)
    if user_id is None:
        # `raise` interrompe subito l'esecuzione e fa rispondere FastAPI
        # con l'errore HTTP indicato, senza mai raggiungere l'endpoint.
        raise errore_auth
    utente = session.get(Utente, user_id)
    if utente is None:
        raise errore_auth
    return utente


# Alias finale: qualunque endpoint scriva `utente: UtenteCorrente` come
# parametro riceve automaticamente l'oggetto Utente già autenticato, o la
# richiesta viene rifiutata con 401 PRIMA che il codice dell'endpoint parta.
UtenteCorrente = Annotated[Utente, Depends(utente_corrente)]


def verifica_alexa_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Dependency di autenticazione per la skill Alexa: nessun JWT, nessun
    login. La Lambda della skill manda una chiave segreta fissa nell'header
    `X-Api-Key`, qui la confrontiamo con quella configurata in ambiente.

    Alexa non ha un concetto di "utente che fa login": la skill la usa una
    persona sola, quindi invece di autenticare un Utente del database
    ritorniamo direttamente l'id fisso (ALEXA_USER_ID) a cui vanno
    associate tutte le domande poste tramite la skill.
    """
    if not ALEXA_API_KEY or not ALEXA_USER_ID:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Skill Alexa non configurata (ALEXA_API_KEY/ALEXA_USER_ID mancanti)",
        )
    # `secrets.compare_digest` invece di `==`: confronta le stringhe in un
    # tempo costante indipendente da quanti caratteri coincidono, per non
    # dare a un attaccante informazioni utili misurando i tempi di risposta
    # (attacco a "timing side-channel").
    if not x_api_key or not secrets.compare_digest(x_api_key, ALEXA_API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Chiave API non valida")
    return ALEXA_USER_ID


# Alias riusabile per l'endpoint della skill Alexa: chi lo dichiara come
# parametro riceve direttamente lo user_id fisso associato alla skill.
AlexaUserId = Annotated[str, Depends(verifica_alexa_api_key)]
