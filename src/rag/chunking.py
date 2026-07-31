"""Chunking su misura dei documenti: split per pasto/giorno (dieta e allenamento)
o split ricorsivo generico (liste della spesa e testi non strutturati)."""

import re

from .config import KEEP_COMPACT

_GIORNO = re.compile(r"(?i)(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|domenica)")
_PASTO = re.compile(r"(?=(?:Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b)")
_HEADER = re.compile(
    r"(?=(?:[A-E]\s*[\x7f•·]\s*(?:Luned|Marted|Mercoled|Gioved|Venerd))"  # giorni dieta
    r"|(?:SESSIONE\s*\d)"  # sedute allenamento
    r"|(?:Note generali)|(?:Indicazioni\b)|(?:Note e terminologia))"
)
_MEAL = re.compile(r"^(Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b[^\d]*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$")

# --- Sedute di allenamento (tabelle esercizio/serie×rep/recupero/focus) ---
_SESSIONE_START = re.compile(r"^SESSIONE\s*\d")
_SESSIONE = re.compile(r"^SESSIONE\s*(\d+)\s*[—-]\s*(.+?)\s*$", re.MULTILINE)
_SOTTOTABELLA = re.compile(r"(?=(?:RISCALDAMENTO|ALLENAMENTO)\b)")
_HEADER_TABELLA = re.compile(r"^(ESERCIZIO|SERIE\s*[×xX]\s*REP|RECUPERO|FOCUS|PESO|FEELING)$")
_SERIE_REP = re.compile(r'^\d+\s*x\s*\d+(?:/\d+)?["\']?(?:\s*x\s*\w*)?$|^\d+["\']?$')
_INIZIO_RECUPERO = re.compile(r'^(?:/|\d+.*[\'"])')
_FOCUS_NOTI = re.compile(
    r"(?i)^(Petto|Dorsale|Bicipiti|Tricipiti|Spalle|Quadricipiti|Femorali|"
    r"Centro|Polpacci|Addome|Core|Glutei|Adduttori|Abduttori)$"
)


def _norm_giorno(s: str) -> str:
    return s.lower().replace("ì", "i")


def chunk_documento(doc: dict) -> list[dict]:
    nome = doc["fonte"].lower()
    records = []
    if any(k in nome for k in KEEP_COMPACT):
        for p in split_ricorsivo(doc["testo"], max_caratteri=1500):
            records.append({"testo": p, "fonte": doc["fonte"], "giorno": ""})
    else:
        for blocco in split_strutturato(doc["testo"]):
            g = _GIORNO.search(blocco[:20])
            if g:
                giorno = _norm_giorno(g.group(0))
                for p in split_pasti(blocco):
                    records.append({"testo": p, "fonte": doc["fonte"], "giorno": giorno})
            elif _SESSIONE_START.match(blocco):
                for p in split_sessione(blocco):
                    records.append({"testo": p, "fonte": doc["fonte"], "giorno": ""})
            else:
                pezzi = split_ricorsivo(blocco, 800) if len(blocco) > 1200 else [re.sub(r"\s+", " ", blocco)]
                for p in pezzi:
                    records.append({"testo": p, "fonte": doc["fonte"], "giorno": ""})
    return records


def split_pasti(blocco: str) -> list[str]:
    m = _GIORNO.search(blocco[:20])
    giorno = m.group(0).capitalize() if m else "Giorno"
    out = []
    for p in _PASTO.split(blocco):
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        mm = _MEAL.match(p)
        if mm:
            pasto, carb, prot, gr, kcal, cibi = mm.groups()
            out.append(
                f"{giorno}, {pasto.lower()}: {cibi.strip()} "
                f"(macro: {carb}g carboidrati, {prot}g proteine, {gr}g grassi, {kcal} kcal)"
            )
        else:
            out.append(f"{giorno} — {p}")  # es. riga dei totali giornalieri
    return out


def split_sessione(blocco: str) -> list[str]:
    m = _SESSIONE.match(blocco)
    if not m:
        return [re.sub(r"\s+", " ", blocco).strip()]
    numero, nome_sessione = m.group(1), m.group(2)

    out = []
    for parte in _SOTTOTABELLA.split(blocco):
        parte = parte.strip()
        if not parte:
            continue
        prima_riga, _, resto = parte.partition("\n")
        etichetta = prima_riga.strip().lower()
        if etichetta not in ("riscaldamento", "allenamento"):
            continue
        # una riga vuota separa la tabella dall'intestazione ripetuta di pagina
        # (es. "Scheda di allenamento — ...") che precede la prossima sessione
        resto = resto.split("\n\n", 1)[0]
        righe = [r.strip() for r in resto.splitlines() if r.strip() and not _HEADER_TABELLA.match(r.strip())]
        for es in _parsa_esercizi(righe):
            out.append(_format_esercizio(numero, nome_sessione, etichetta, es))
    return out


def _parsa_esercizi(righe: list[str]) -> list[dict]:
    esercizi = []
    corrente = {"nome": [], "serie_rep": [], "recupero": [], "focus": []}
    stato = "NOME"

    def chiudi_esercizio():
        if corrente["nome"] or corrente["serie_rep"]:
            esercizi.append(corrente)

    def nuovo_esercizio():
        nonlocal corrente
        chiudi_esercizio()
        corrente = {"nome": [], "serie_rep": [], "recupero": [], "focus": []}

    for r in righe:
        if stato == "NOME":
            if _SERIE_REP.match(r):
                corrente["serie_rep"].append(r)
                stato = "SERIE_REP"
            else:
                corrente["nome"].append(r)
        elif stato == "SERIE_REP":
            if _INIZIO_RECUPERO.match(r):
                corrente["recupero"].append(r)
                stato = "RECUPERO"
            else:
                corrente["serie_rep"].append(r)
        else:  # stato == "RECUPERO"
            if _SERIE_REP.match(r):
                # innesco serie×rep senza passare per un nuovo nome: capita solo
                # se il nome del prossimo esercizio è vuoto (non osservato nei dati)
                nuovo_esercizio()
                corrente["serie_rep"].append(r)
                stato = "SERIE_REP"
            elif not corrente["focus"] and _FOCUS_NOTI.match(r):
                corrente["focus"].append(r)
            elif corrente["focus"] and r.count(" ") == 0 and r[:1].islower():
                # singola parola minuscola dopo il focus: sua continuazione
                # su due righe (es. "schiena" dopo "Centro")
                corrente["focus"].append(r)
            elif not corrente["focus"] and r[:1].islower():
                # riga minuscola prima del focus: continuazione del recupero
                # multi-riga (es. "braccio e l'altro" dopo "30-60\" tra un")
                corrente["recupero"].append(r)
            else:
                # non è continuazione: è il nome del prossimo esercizio
                nuovo_esercizio()
                corrente["nome"].append(r)
                stato = "NOME"
    chiudi_esercizio()
    return esercizi


def _unisci_nome(righe: list[str]) -> str:
    nome = righe[0]
    for riga in righe[1:]:
        if riga.startswith("("):
            nome += f" {riga}"
        elif riga.startswith("-"):
            nome += f", {riga.lstrip('- ').strip()}"
        else:
            nome += f", {riga}"
    return nome


def _format_esercizio(numero: str, nome_sessione: str, etichetta: str, es: dict) -> str:
    nome = _unisci_nome(es["nome"]) if es["nome"] else "?"
    serie_rep = re.sub(r"\s+", " ", " ".join(es["serie_rep"])).strip()
    recupero = re.sub(r"\s+", " ", " ".join(es["recupero"])).strip()
    riga = f"Sessione {numero} — {nome_sessione}, {etichetta}: {nome} — {serie_rep}, recupero {recupero}"
    if es["focus"]:
        focus = re.sub(r"\s+", " ", " ".join(es["focus"])).strip()
        riga += f", focus {focus}"
    return riga


def split_strutturato(testo: str) -> list[str]:
    parti = [p.strip() for p in _HEADER.split(testo) if p.strip()]
    return parti if parti else [testo]


def split_ricorsivo(testo: str, max_caratteri: int = 600, separatori: list[str] | None = None) -> list[str]:
    if separatori is None:
        separatori = ["\n\n", "\n", ". ", " ", ""]
    if len(testo) <= max_caratteri:
        return [testo.strip()] if testo.strip() else []
    sep, resto = separatori[0], separatori[1:]

    if sep == "":
        return [testo[i : i + max_caratteri] for i in range(0, len(testo), max_caratteri)]

    chunk, corrente = [], ""
    for parte in testo.split(sep):
        candidato = parte if not corrente else corrente + sep + parte
        if len(candidato) <= max_caratteri:
            corrente = candidato
        else:
            if corrente:
                chunk.append(corrente.strip())
            if len(parte) > max_caratteri:
                chunk += split_ricorsivo(parte, max_caratteri, resto)
                corrente = ""
            else:
                corrente = parte
    if corrente.strip():
        chunk.append(corrente.strip())
    return chunk
