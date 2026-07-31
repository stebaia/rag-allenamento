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
