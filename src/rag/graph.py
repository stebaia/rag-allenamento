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

# Le schede di allenamento numerano gli allenamenti "Sessione 1/2/3" (vedi
# rag/chunking.py, split_sessione), non "primo/secondo/terzo giorno/sessione/
# seduta". Stessa situazione dei giorni della settimana: senza questa
# sostituzione, una domanda come "come mi alleno nella prima sessione" trova
# match solo per similarità semantica generica (mischia pezzi di sessioni
# diverse) invece che per le parole esatte "Sessione 1" presenti nel testo
# del chunk.
_NUMERI_ORDINALI = {"primo": 1, "prima": 1, "secondo": 2, "seconda": 2, "terzo": 3, "terza": 3}
# L'articolo/preposizione che precede l'ordinale viene CATTURATO (gruppo 1) e
# riscritto al femminile: senza catturarlo produrremmo "come mi alleno
# Sessione 1" (perdendo il "nella"), ma riportandolo tale e quale
# produrremmo "il Sessione 1", perché "giorno"/"allenamento" sono maschili
# mentre "Sessione" è femminile. La domanda riscritta finisce nel prompt
# dell'LLM, quindi vale la pena che resti in italiano corretto.
_ARTICOLO_FEMMINILE = {"il": "la", "la": "la", "del": "della", "della": "della", "nel": "nella", "nella": "nella"}
_SESSIONE_RELATIVA_RE = re.compile(
    r"(?i)\b(?:(il|la|del|della|nel|nella)\s+)?"
    r"(primo|prima|secondo|seconda|terzo|terza)\s+(?:giorno|allenamento|sessione|seduta)\b"
    r"(?:\s+di\s+allenamento\b)?"
)


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


def risolvi_sessioni_relative(domanda: str) -> str:
    """Sostituisce 'primo/secondo/terzo giorno (di allenamento)' con 'Sessione N',
    preservando l'articolo che lo precede ('nella prima sessione' ->
    'nella Sessione 1')."""

    def _sostituisci(match: re.Match) -> str:
        articolo, ordinale = match.group(1), match.group(2)
        numero = _NUMERI_ORDINALI[ordinale.lower()]
        if not articolo:
            return f"Sessione {numero}"
        return f"{_ARTICOLO_FEMMINILE[articolo.lower()]} Sessione {numero}"

    return _SESSIONE_RELATIVA_RE.sub(_sostituisci, domanda)


def riformula_con_storico(llm, domanda: str, storico: list[dict]) -> str:
    """Riscrive una domanda ellittica ('e nella giornata a?') in una domanda
    autonoma, usando gli ultimi scambi della conversazione come contesto.

    Serve PRIMA del retrieval: il RetrieverIbrido (vedi rag/vectorstore.py)
    cerca solo le parole/il significato della domanda che riceve, senza
    alcuna nozione di "conversazione" — se l'utente scrive "e cosa mangio in
    quella giornata?" senza questo passaggio il retriever cercherebbe alla
    lettera "quella giornata", senza sapere a cosa si riferisce. Se non c'è
    storico (prima domanda della chat), non c'è nulla da riformulare.
    """
    if not storico:
        return domanda

    scambi = "\n".join(f"{m['ruolo']}: {m['contenuto']}" for m in storico)
    nuova = llm.invoke(
        "Questa è una conversazione tra un utente e un assistente su dieta e "
        "allenamento. Riscrivi l'ULTIMA domanda dell'utente come domanda "
        "autonoma e completa, esplicitando ciò a cui si riferisce implicitamente "
        "(es. un giorno, una sessione, un piano citati prima nella conversazione). "
        "Se la domanda è già autonoma, restituiscila invariata. Rispondi SOLO con "
        "la domanda riscritta, senza introduzioni.\n\n"
        f"CONVERSAZIONE PRECEDENTE:\n{scambi}\n\n"
        f"ULTIMA DOMANDA DELL'UTENTE: {domanda}"
    ).content.strip()
    print(f"   ↳ riformulata con lo storico: {nuova}")
    return nuova


class Stato(TypedDict):
    domanda: str
    documenti: list[Document]
    risposta: str
    tentativi: int
    storico: str


def costruisci_grafo(retriever, llm, prompt):
    # NODO: recupera i documenti dall'indice
    def nodo_recupera(stato: Stato):
        # Risolta solo al primo passaggio (tentativi == 0): dal secondo in
        # poi la domanda è già stata riscritta da nodo_riformula, che lavora
        # a valle di questa sostituzione e non reintroduce parole relative.
        domanda = stato["domanda"]
        if stato.get("tentativi", 0) == 0:
            domanda = risolvi_giorni_relativi(domanda)
            domanda = risolvi_sessioni_relative(domanda)
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
            {
                "context": contesto,
                "question": stato["domanda"],
                "oggi": giorno_oggi(),
                "storico": stato.get("storico", ""),
            }
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
