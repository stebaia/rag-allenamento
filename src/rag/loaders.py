"""Caricamento dei documenti sorgente (PDF/TXT) da disco."""

import glob
import os

from pypdf import PdfReader


def carica_file(percorso: str) -> dict | None:
    """Estrae testo da un singolo file PDF/TXT. Ritorna None se non c'è testo estraibile."""
    fonte = os.path.basename(percorso)
    if percorso.lower().endswith(".pdf"):
        reader = PdfReader(percorso)
        testo = "\n".join((p.extract_text() or "") for p in reader.pages)
    else:
        with open(percorso, encoding="utf-8") as f:
            testo = f.read()

    if not testo.strip():
        # PDF probabilmente scansionato (una "foto" del testo): servirebbe l'OCR
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
