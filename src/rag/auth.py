"""Hashing password e creazione/verifica dei JWT.

Due concetti distinti in questo file:

1. HASHING DELLE PASSWORD (bcrypt): non salviamo mai la password
   dell'utente così com'è. Calcoliamo un "hash" — una trasformazione che
   non si può invertire — e salviamo solo quello. Per verificare un
   login, rifacciamo l'hash della password inserita e controlliamo che
   corrisponda. Così, anche se qualcuno rubasse il database, non
   otterrebbe le password in chiaro.

2. JWT (JSON Web Token): dopo login/registrazione, invece di far
   ricordare al server "questo utente è loggato" (sessione), gli diamo un
   "biglietto" firmato digitalmente (il token). Ad ogni richiesta
   successiva, l'utente manda il token nell'header Authorization, e noi
   verifichiamo solo che la firma sia valida e non scaduta — senza dover
   consultare un database per sapere chi è loggato. Il token contiene
   dentro di sé (in chiaro, ma firmato) l'id dell'utente.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from .config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

# Controllo eseguito UNA VOLTA, quando questo file viene importato per la
# prima volta (cioè all'avvio dell'app). Se manca il segreto, l'app non
# deve nemmeno partire: meglio un crash immediato e chiaro a boot, che
# scoprire il problema più tardi con token generati in modo insicuro.
if not JWT_SECRET:
    raise RuntimeError("Manca JWT_SECRET nell'ambiente: impostalo prima di avviare l'API.")


def hash_password(password: str) -> str:
    """Calcola l'hash bcrypt di una password.

    bcrypt lavora su `bytes`, non su `str` (il tipo testo "normale" di
    Python), quindi `.encode()` converte la stringa in bytes usando UTF-8.
    `bcrypt.gensalt()` genera un "sale" casuale che rende ogni hash diverso
    anche per due password identiche, per proteggersi da attacchi con
    tabelle precalcolate. Il risultato è di nuovo bytes, quindi lo
    riconvertiamo in stringa con `.decode()` per poterlo salvare comodamente
    come testo nel database (colonna `password_hash` in db.py).
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verifica_password(password: str, password_hash: str) -> bool:
    """Controlla se `password` corrisponde all'hash salvato in precedenza."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def crea_access_token(user_id: str) -> str:
    """Crea un JWT che identifica l'utente, valido per JWT_EXPIRE_MINUTES minuti."""
    scadenza = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    # Il "payload" è il contenuto del token. "sub" (subject) è il nome
    # standard JWT per "a chi si riferisce questo token"; "exp" (expiry) è
    # il nome standard per la scadenza: la libreria `jose` lo legge da solo
    # e rifiuta il token scaduto quando lo decodifichiamo più sotto.
    payload = {"sub": user_id, "exp": scadenza}
    # jwt.encode "firma" il payload con JWT_SECRET: chiunque può LEGGERE il
    # contenuto del token (non è cifrato, solo firmato), ma solo chi
    # conosce JWT_SECRET può crearne uno valido o modificarlo senza che la
    # firma smetta di corrispondere.
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decodifica_access_token(token: str) -> str | None:
    """Ritorna lo user_id contenuto nel token, o None se non valido/scaduto."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        # Firma non valida, token manomesso, scaduto, malformato... in
        # tutti questi casi trattiamo il token come "non autenticato",
        # senza distinguere il motivo esatto (chi chiama, vedi api/deps.py,
        # risponderà con un generico 401 Unauthorized).
        return None
    return payload.get("sub")
