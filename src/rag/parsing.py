"""Strato di parsing: da PDF a struttura, prima del chunking.

Il problema che risolve. `pypdf` restituisce il testo come flusso piatto di
caratteri: butta via dimensione del font, grassetto, colonne, tabelle. Da lì
l'unico modo per capire dove finisce una sezione è indovinarlo con espressioni
regolari sul testo — e ogni documento nuovo ne richiede di nuove. È il motivo
per cui chunking.py era arrivato a diciassette regex su misura.

Qui il testo viene estratto insieme ai suoi attributi tipografici. Un titolo si
riconosce perché è in grassetto o più grande del corpo, non perché corrisponde
a "Articolo \\d+": un criterio che vale su qualsiasi documento, senza sapere in
anticipo come numera le sue sezioni.

Due backend:
- PyMuPDF (default): locale, ~0,2 s per un libro di 176 pagine, espone font e
  posizione. Copre i PDF con testo estraibile, che sono la quasi totalità.
- Unstructured `hi_res` (opzionale): classifica gli elementi in Title,
  NarrativeText, Table, Image usando modelli di layout detection. Riconosce le
  tabelle, che PyMuPDF non distingue dalla prosa. Costa però ~2,4 s per pagina
  — undici minuti per un codice di 282 pagine — quindi va acceso per documento,
  non tenuto come default.
"""

import os
import statistics
from dataclasses import dataclass, field

# Backend disponibili. "auto" = PyMuPDF, con ricaduta su pypdf se fallisce.
BACKEND_DEFAULT = os.environ.get("PARSER_PDF", "auto")


@dataclass
class Riga:
    """Una riga di testo con gli attributi che servono a capire cosa sia."""

    testo: str
    dimensione: float = 0.0
    grassetto: bool = False
    pagina: int = 0
    # Valorizzato solo dal backend Unstructured: "Title", "NarrativeText",
    # "Table", "ListItem". Con PyMuPDF resta vuoto e il ruolo si deduce dagli
    # attributi tipografici.
    categoria: str = ""

    @property
    def e_tabella(self) -> bool:
        return self.categoria == "Table"


@dataclass
class DocumentoParsato:
    """Il risultato del parsing: righe con attributi più il testo piatto.

    `testo` resta disponibile perché i chunker dei documenti personali
    (dieta, spesa, allenamento) lavorano su quello e non vanno toccati.
    """

    testo: str
    righe: list[Riga] = field(default_factory=list)
    backend: str = ""

    @property
    def dimensione_corpo(self) -> float:
        """La dimensione di font del testo normale.

        È la mediana pesata sulla quantità di testo: i titoli sono pochi
        caratteri in corpo grande, il testo normale è tanti caratteri in corpo
        piccolo, quindi la mediana pesata cade sempre sul corpo.
        """
        campione = []
        for r in self.righe:
            if r.dimensione and r.testo.strip():
                # Ogni riga pesa per quanto testo contiene.
                campione.extend([r.dimensione] * max(1, len(r.testo) // 20))
        return statistics.median(campione) if campione else 0.0

    def e_titolo(self, riga: Riga) -> bool:
        """Se la riga è un'intestazione di sezione.

        Un titolo è più grande del corpo, oppure in grassetto e corto. La
        lunghezza conta: in molti documenti giuridici interi paragrafi sono in
        grassetto per enfasi, e prenderli per titoli spezzerebbe il testo in
        frammenti inutilizzabili.
        """
        if riga.categoria:  # backend Unstructured: la classificazione è esplicita
            return riga.categoria == "Title"
        testo = riga.testo.strip()
        if not testo or len(testo) > 120:
            return False
        corpo = self.dimensione_corpo
        if corpo and riga.dimensione > corpo * 1.15:
            return True
        return riga.grassetto and len(testo) <= 90


def _parse_pymupdf(percorso: str) -> DocumentoParsato:
    import pymupdf

    doc = pymupdf.open(percorso)
    righe: list[Riga] = []
    for n_pagina, pagina in enumerate(doc, start=1):
        for blocco in pagina.get_text("dict")["blocks"]:
            for linea in blocco.get("lines", []):
                pezzi = linea.get("spans", [])
                if not pezzi:
                    continue
                testo = "".join(p["text"] for p in pezzi).strip()
                if not testo:
                    continue
                # Gli attributi del primo span rappresentano la riga: quando una
                # riga mescola stili, è l'inizio a dirci se è un titolo.
                primo = pezzi[0]
                righe.append(
                    Riga(
                        testo=testo,
                        dimensione=round(primo["size"], 1),
                        grassetto="Bold" in primo["font"] or bool(primo["flags"] & 16),
                        pagina=n_pagina,
                    )
                )
    doc.close()
    return DocumentoParsato(
        testo="\n".join(r.testo for r in righe), righe=righe, backend="pymupdf"
    )


def _parse_unstructured(percorso: str) -> DocumentoParsato:
    from unstructured.partition.pdf import partition_pdf

    # strategy="fast" non estrae nulla su questa versione: si usa hi_res, che
    # richiede tesseract installato e i modelli di layout detection.
    elementi = partition_pdf(filename=percorso, strategy="hi_res", languages=["ita"])
    righe = []
    for e in elementi:
        testo = (e.text or "").strip()
        if not testo:
            continue
        righe.append(
            Riga(
                testo=testo,
                pagina=getattr(e.metadata, "page_number", 0) or 0,
                categoria=type(e).__name__,
            )
        )
    return DocumentoParsato(
        testo="\n".join(r.testo for r in righe), righe=righe, backend="unstructured"
    )


def _parse_pypdf(percorso: str) -> DocumentoParsato:
    """Ricaduta senza attributi tipografici: il chunking userà solo le regex."""
    from pypdf import PdfReader

    testo = "\n".join((p.extract_text() or "") for p in PdfReader(percorso).pages)
    return DocumentoParsato(testo=testo, righe=[], backend="pypdf")


def parse_pdf(percorso: str, backend: str | None = None) -> DocumentoParsato:
    """Estrae struttura e testo da un PDF.

    `backend`: "auto"/"pymupdf" (default), "unstructured" per i documenti in
    cui contano le tabelle, "pypdf" per tornare al comportamento originale.
    Se il backend scelto fallisce si ricade su pypdf: un parsing degradato è
    sempre meglio di un documento non indicizzabile.
    """
    scelto = (backend or BACKEND_DEFAULT).lower()
    tentativi = {
        "unstructured": _parse_unstructured,
        "pymupdf": _parse_pymupdf,
        "auto": _parse_pymupdf,
        "pypdf": _parse_pypdf,
    }
    funzione = tentativi.get(scelto, _parse_pymupdf)
    try:
        risultato = funzione(percorso)
        if risultato.testo.strip():
            return risultato
        print(f"   ↳ [parsing] {scelto} non ha estratto testo, provo pypdf")
    except Exception as e:
        print(f"   ↳ [parsing] {scelto} fallito ({type(e).__name__}), provo pypdf")
    return _parse_pypdf(percorso)
