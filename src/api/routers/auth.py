"""Endpoint di registrazione e login.

Un "APIRouter" è un gruppo di endpoint correlati (qui tutto ciò che
riguarda l'autenticazione). `main.py` li assembla tutti nell'app finale con
`app.include_router(...)`: è un modo di organizzare il codice in file più
piccoli invece di avere tutti gli endpoint in un unico main.py enorme.

Le classi che ereditano da Pydantic `BaseModel` (RegistrazioneIn, TokenOut)
descrivono la FORMA dei dati JSON in ingresso/uscita. FastAPI le usa per:
  - validare automaticamente il body delle richieste (se manca un campo o
    ha il tipo sbagliato, risponde da solo con un errore 422, senza che
    dobbiamo scrivere controlli a mano);
  - generare la documentazione automatica su /docs;
  - serializzare l'oggetto Python di ritorno in JSON.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlmodel import select

from rag.auth import crea_access_token, hash_password, verifica_password
from rag.db import Utente

from ..deps import SessionDep

# prefix="/auth": tutti gli endpoint di questo router iniziano con /auth,
# quindi "/register" qui sotto diventa "/auth/register" nell'app finale.
# tags=["auth"] serve solo a raggruppare questi endpoint nella pagina /docs.
router = APIRouter(prefix="/auth", tags=["auth"])


class RegistrazioneIn(BaseModel):
    # EmailStr non è "una stringa qualunque": Pydantic verifica che il
    # valore abbia davvero la forma di un indirizzo email valido.
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"  # valore di default se non specificato altrove


# Il decoratore @router.post("/register", ...) collega questa funzione
# all'URL POST /auth/register. `response_model=TokenOut` dice a FastAPI:
# "qualunque cosa ritorni questa funzione, valida/serializzala come TokenOut".
@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def registra(dati: RegistrazioneIn, session: SessionDep):
    """Crea un nuovo utente e ritorna subito un token (login automatico dopo la registrazione)."""
    # `select(Utente).where(...)` costruisce una query SQL del tipo
    # "SELECT * FROM utente WHERE email = ...", ma scritta con oggetti
    # Python invece che stringhe SQL a mano. `.first()` prende il primo
    # risultato, o None se non c'è nessuna riga corrispondente.
    esistente = session.exec(select(Utente).where(Utente.email == dati.email)).first()
    if esistente:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email già registrata")

    utente = Utente(email=dati.email, password_hash=hash_password(dati.password))
    session.add(utente)     # segna l'oggetto come "da salvare"
    session.commit()        # scrive davvero su disco (il file SQLite)
    session.refresh(utente)  # ricarica l'oggetto per avere i valori generati dal DB (es. creato_il)

    return TokenOut(access_token=crea_access_token(utente.id))


@router.post("/login", response_model=TokenOut)
def login(
    # OAuth2PasswordRequestForm è lo standard che FastAPI usa per i login:
    # si aspetta un form (non JSON) con campi "username" e "password". Lo
    # usiamo passando l'email nel campo "username" — è solo una
    # convenzione del form, non ha a che fare con l'avere o meno uno
    # username separato dall'email.
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
):
    """Verifica le credenziali e ritorna un nuovo token JWT."""
    utente = session.exec(select(Utente).where(Utente.email == form.username)).first()
    if not utente or not verifica_password(form.password, utente.password_hash):
        # Messaggio d'errore volutamente generico: non riveliamo se è
        # l'email a non esistere o la password a essere sbagliata, per non
        # facilitare chi sta provando a indovinare email registrate.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenziali non valide")

    return TokenOut(access_token=crea_access_token(utente.id))
