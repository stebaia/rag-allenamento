"""Upload e gestione documenti: indicizzazione asincrona in background.

Il punto delicato di questo file è capire PERCHÉ l'upload non fa tutto il
lavoro subito. Estrarre testo da un PDF, spezzarlo in chunk e calcolare gli
embedding può richiedere alcuni secondi (o di più, per file grandi). Se
l'endpoint facesse tutto questo prima di rispondere, l'utente che carica il
file resterebbe con la richiesta HTTP "appesa" per tutto quel tempo.

Invece: l'endpoint salva subito il file, crea una riga nel database con
stato "processing" e risponde IMMEDIATAMENTE con l'id del documento. Il
lavoro vero e proprio (funzione `_indicizza_documento`) viene schedulato
con `BackgroundTasks`, che lo esegue DOPO che la risposta HTTP è già stata
inviata al client. Il client può poi controllare periodicamente
(`GET /documents/{id}`) se lo stato è diventato "ready" (o "error").
"""

import os
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlmodel import Session, select

from rag.chunking import chunk_documento
from rag.config import CARTELLA_UPLOAD
from rag.contextualize import contestualizza_documento
from rag.db import Documento, StatoDocumento, get_session
from rag.loaders import carica_file
from rag.vectorstore import elimina_documento, upsert_documento

from ..deps import SessionDep, UtenteCorrente
from ..state import get_embeddings, get_llm_contesto

router = APIRouter(prefix="/documents", tags=["documents"])

ESTENSIONI_VALIDE = (".pdf", ".txt")


class DocumentoOut(BaseModel):
    """Forma JSON con cui esponiamo un Documento nelle risposte dell'API."""

    id: str
    nome_file: str
    stato: StatoDocumento
    errore: str | None
    n_chunk: int


def _indicizza_documento(document_id: str, user_id: str, percorso: str) -> None:
    """Eseguito in background: estrae testo, chunka, contestualizza, embedda, upserta su Qdrant, aggiorna lo stato.

    Questa funzione gira DOPO che la risposta HTTP dell'upload è già
    partita, quindi non possiamo riusare la `SessionDep` iniettata
    dall'endpoint (quella sessione FastAPI la chiude appena la richiesta
    finisce). Apriamo perciò una sessione DB "a mano" chiamando
    `get_session()` e prendendo il primo (e unico) valore che produce con
    `next(...)` — ricorda che `get_session` è un generatore (vedi db.py):
    `next()` esegue il codice fino al primo `yield` e ci dà quel valore.
    """
    session_gen = get_session()
    session: Session = next(session_gen)
    try:
        documento = session.get(Documento, document_id)
        if documento is None:
            return
        try:
            doc = carica_file(percorso)
            if doc is None:
                raise ValueError("Nessun testo estraibile dal file (forse un PDF scansionato)")

            chunks = chunk_documento(doc)
            llm_contesto = get_llm_contesto()
            chunks = contestualizza_documento(llm_contesto, doc["testo"], chunks)
            embeddings = get_embeddings()
            n = upsert_documento(embeddings, user_id, document_id, chunks)

            documento.stato = StatoDocumento.READY
            documento.n_chunk = n
        except Exception as exc:  # noqa: BLE001 - vogliamo intercettare qualsiasi errore di parsing/embedding
            # Qui intercettiamo DELIBERATAMENTE qualunque tipo di errore
            # (PDF corrotto, Qdrant irraggiungibile, ecc.): questo codice
            # gira in background, senza nessuno che possa "vedere" un
            # traceback come farebbe una richiesta HTTP normale. Meglio
            # salvare l'errore nello stato del documento che perderlo.
            documento.stato = StatoDocumento.ERROR
            documento.errore = str(exc)
        finally:
            # `finally` esegue sempre, sia che il blocco `try` sia andato a
            # buon fine sia che abbia sollevato un'eccezione: garantisce
            # che lo stato (READY o ERROR) venga comunque salvato.
            session.add(documento)
            session.commit()
    finally:
        session_gen.close()


# `async def` invece di `def`: questo endpoint fa I/O di rete (leggere il
# file caricato dal client un pezzo alla volta) e usiamo `await` per
# aspettarlo senza bloccare il resto del server nel frattempo. Gli altri
# endpoint di questo file sono `def` normali perché il loro lavoro (query
# al database) è già gestito in modo efficiente da FastAPI dietro le quinte.
@router.post("", response_model=DocumentoOut, status_code=status.HTTP_202_ACCEPTED)
async def carica_documento(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    utente: UtenteCorrente,
    session: SessionDep,
):
    """Riceve un file, lo salva su disco, crea il record 'processing' e schedula l'indicizzazione."""
    nome_file = file.filename or ""
    estensione = os.path.splitext(nome_file)[1].lower()
    if estensione not in ESTENSIONI_VALIDE:
        # Copre anche il caso di un upload senza nome file: `splitext("")[1]`
        # è la stringa vuota, che non è tra le estensioni valide. Da qui in
        # poi `nome_file` è garantito non vuoto (la colonna `nome_file` in
        # db.py non ammette NULL).
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Sono supportati solo file PDF e TXT")

    os.makedirs(CARTELLA_UPLOAD, exist_ok=True)
    document_id = str(uuid.uuid4())
    percorso = os.path.join(CARTELLA_UPLOAD, f"{document_id}{estensione}")

    # `await file.read()` legge tutto il contenuto del file caricato in
    # memoria (bytes); per file molto grandi si potrebbe leggere a blocchi,
    # ma per PDF di poche decine di MB va benissimo così.
    contenuto = await file.read()
    with open(percorso, "wb") as f:  # "wb" = write binary, scriviamo bytes grezzi
        f.write(contenuto)

    documento = Documento(
        id=document_id,
        user_id=utente.id,
        nome_file=nome_file,
        stato=StatoDocumento.PROCESSING,
    )
    session.add(documento)
    try:
        session.commit()
    except Exception:
        # Se la riga non entra nel database, il file appena scritto non
        # sarebbe più raggiungibile da nessuno (nessun record lo nomina) e
        # resterebbe a occupare spazio per sempre: lo rimuoviamo subito
        # invece di lasciare un orfano su disco.
        session.rollback()
        if os.path.exists(percorso):
            os.remove(percorso)
        raise
    session.refresh(documento)

    # Questo NON esegue subito _indicizza_documento: la registra per essere
    # eseguita da FastAPI subito dopo aver inviato la risposta HTTP.
    background_tasks.add_task(_indicizza_documento, document_id, utente.id, percorso)

    return documento


@router.get("", response_model=list[DocumentoOut])
def lista_documenti(utente: UtenteCorrente, session: SessionDep):
    """Elenca solo i documenti dell'utente autenticato (mai quelli di altri)."""
    return session.exec(select(Documento).where(Documento.user_id == utente.id)).all()


@router.get("/{document_id}", response_model=DocumentoOut)
def dettaglio_documento(document_id: str, utente: UtenteCorrente, session: SessionDep):
    """Stato di un documento; usato dal client per fare polling durante l'indicizzazione."""
    documento = session.get(Documento, document_id)
    # Controlliamo anche `documento.user_id != utente.id`, non solo
    # l'esistenza: senza questo controllo, un utente potrebbe indovinare
    # l'id di un documento altrui e leggerne i dettagli.
    if documento is None or documento.user_id != utente.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Documento non trovato")
    return documento


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def rimuovi_documento(document_id: str, utente: UtenteCorrente, session: SessionDep):
    """Elimina il documento: vettori su Qdrant, riga nel database, file su disco."""
    documento = session.get(Documento, document_id)
    if documento is None or documento.user_id != utente.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Documento non trovato")

    elimina_documento(document_id)  # rimuove i vettori da Qdrant
    session.delete(documento)       # rimuove la riga dal database
    session.commit()

    for estensione in ESTENSIONI_VALIDE:
        percorso = os.path.join(CARTELLA_UPLOAD, f"{document_id}{estensione}")
        if os.path.exists(percorso):
            os.remove(percorso)
            break
