"""Grafo agentico LangGraph: recupero -> (riformula <-> recupera) -> genera."""

import re
from datetime import datetime, timedelta
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

_GIORNI_SETTIMANA = (
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
)

# Mappa parola relativa -> offset in giorni rispetto a oggi, usata per
# risolvere "oggi"/"domani"/"ieri" nel nome del giorno reale (es. "venerdì")
# PRIMA di interrogare il retriever: RetrieverIbrido riconosce solo nomi di
# giorni espliciti nel testo della domanda (vedi rag/vectorstore.py), quindi
# senza questa sostituzione il boost per giorno non scatterebbe mai.
_RELATIVI = {"dopodomani": 2, "domani": 1, "oggi": 0, "ieri": -1}
_RELATIVI_RE = re.compile(r"(?i)\b(dopodomani|domani|oggi|ieri)\b")


def giorno_oggi() -> str:
    """Nome del giorno della settimana corrente, in italiano (es. 'venerdì')."""
    return _GIORNI_SETTIMANA[datetime.now().weekday()]


def risolvi_giorni_relativi(domanda: str) -> str:
    """Sostituisce 'oggi'/'domani'/'ieri'/'dopodomani' col nome del giorno reale."""

    def _sostituisci(match: re.Match) -> str:
        offset = _RELATIVI[match.group(1).lower()]
        indice = (datetime.now() + timedelta(days=offset)).weekday()
        return _GIORNI_SETTIMANA[indice]

    return _RELATIVI_RE.sub(_sostituisci, domanda)


class Stato(TypedDict):
    domanda: str
    documenti: list[Document]
    risposta: str
    tentativi: int


def costruisci_grafo(retriever, llm, prompt):
    # NODO: recupera i documenti dall'indice
    def nodo_recupera(stato: Stato):
        # Risolta solo al primo passaggio (tentativi == 0): dal secondo in
        # poi la domanda è già stata riscritta da nodo_riformula, che lavora
        # a valle di questa sostituzione e non reintroduce parole relative.
        domanda = stato["domanda"]
        if stato.get("tentativi", 0) == 0:
            domanda = risolvi_giorni_relativi(domanda)
        docs = retriever.invoke(domanda)
        return {"documenti": docs, "domanda": domanda, "tentativi": stato.get("tentativi", 0) + 1}

    # NODO: riscrive la domanda quando il recupero è scarso
    def nodo_riformula(stato: Stato):
        nuova = llm.invoke(
            "Riscrivi questa domanda in modo più esplicito e ricco di parole "
            "chiave, per cercarla in documenti su dieta e allenamento. Rispondi "
            f"solo con la nuova domanda.\n\nDomanda: {stato['domanda']}"
        ).content.strip()
        print(f"   ↳ riformulo in: {nuova}")
        return {"domanda": nuova}

    # NODO: genera la risposta finale a partire dai documenti recuperati
    def nodo_genera(stato: Stato):
        contesto = "\n\n".join(d.page_content for d in stato["documenti"])
        risposta = (prompt | llm | StrOutputParser()).invoke(
            {"context": contesto, "question": stato["domanda"], "oggi": giorno_oggi()}
        )
        return {"risposta": risposta}

    # BIVIO: i documenti bastano? -> genera ; altrimenti -> riformula
    def decidi(stato: Stato):
        if stato["tentativi"] >= 2:
            return "genera"
        contesto = "\n\n".join(d.page_content for d in stato["documenti"])
        giudizio = llm.invoke(
            "I documenti qui sotto contengono le informazioni per rispondere alla "
            "domanda? Rispondi solo 'sì' o 'no'.\n\n"
            f"DOMANDA: {stato['domanda']}\n\nDOCUMENTI:\n{contesto[:1500]}"
        ).content.strip().lower()
        print(f"   ↳ i documenti bastano? {giudizio}")
        return "genera" if giudizio.startswith("s") else "riformula"

    builder = StateGraph(Stato)
    builder.add_node("recupera", nodo_recupera)
    builder.add_node("riformula", nodo_riformula)
    builder.add_node("genera", nodo_genera)

    builder.add_edge(START, "recupera")
    builder.add_conditional_edges(
        "recupera", decidi, {"genera": "genera", "riformula": "riformula"}
    )
    builder.add_edge("riformula", "recupera")  # ciclo: torna a recuperare
    builder.add_edge("genera", END)
    return builder.compile()
