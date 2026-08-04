"""Caricamento dei documenti sorgente (PDF/TXT) da disco."""

import glob
import os

from .parsing import parse_pdf


def carica_file(percorso: str, backend: str | None = None) -> dict | None:
    """Estrae testo e struttura da un file PDF/TXT.

    Ritorna None se non c'è testo estraibile. Oltre a "testo", il dizionario
    può contenere "parsato": le righe con i loro attributi tipografici, che il
    chunking usa per riconoscere i titoli senza espressioni regolari.
    """
    fonte = os.path.basename(percorso)
    if percorso.lower().endswith(".pdf"):
        doc = parse_pdf(percorso, backend)
        if not doc.testo.strip():
            # PDF probabilmente scansionato (una "foto" del testo): servirebbe l'OCR
            return None
        return {"fonte": fonte, "testo": doc.testo, "parsato": doc}

    with open(percorso, encoding="utf-8") as f:
        testo = f.read()
    if not testo.strip():
        return None
    return {"fonte": fonte, "testo": testo}


def carica_documenti(cartella: str) -> list[dict]:
    documenti = []
    percorsi = sorted(
        glob.glob(os.path.join(cartella, "*.pdf"))
        + glob.glob(os.path.join(cartella, "*.txt"))
    )
    if not percorsi:
        print(f"[!] Nessun file in '{cartella}/'. Mettici i tuoi PDF e rilancia.")
        return documenti

    for percorso in percorsi:
        doc = carica_file(percorso)
        if doc:
            documenti.append(doc)
        else:
            print(f"[!] '{os.path.basename(percorso)}' non contiene testo estraibile (forse scansionato). Lo salto.")
    return documenti
