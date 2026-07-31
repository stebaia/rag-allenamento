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

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from rag.auth import decodifica_access_token
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
