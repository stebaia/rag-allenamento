"""Grafo agentico LangGraph: recupero -> (riformula <-> recupera) -> genera."""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph


class Stato(TypedDict):
    domanda: str
    documenti: list[Document]
    risposta: str
    tentativi: int


def costruisci_grafo(retriever, llm, prompt):
    # NODO: recupera i documenti dall'indice
    def nodo_recupera(stato: Stato):
        docs = retriever.invoke(stato["domanda"])
        return {"documenti": docs, "tentativi": stato.get("tentativi", 0) + 1}

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
            {"context": contesto, "question": stato["domanda"]}
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
