"""Fase 4 del TUTORIAL_LANGGRAPH: streaming, time travel, get_state.

Non è codice di produzione: è uno strumento per *vedere* cosa fa il grafo,
cose che l'endpoint HTTP nasconde perché restituisce solo la risposta finale.

    .venv/bin/python scripts/esplora_grafo.py <email> [modo]

modi: values | updates | messages | history
"""

import sys
import uuid

sys.path.insert(0, "src")

import sqlite3

from dotenv import load_dotenv

load_dotenv(".env")

from langchain_core.messages import HumanMessage  # noqa: E402

from api.state import (  # noqa: E402
    get_checkpointer,
    get_embeddings,
    get_llm,
)
from rag.config import K  # noqa: E402
from rag.graph import costruisci_grafo  # noqa: E402
from rag.vectorstore import retriever_per_utente  # noqa: E402

_PROMPT_MINIMO = (
    "Sei un assistente su dieta, spesa e allenamento. Oggi è {oggi}. Rispondi "
    "usando SOLO il contesto. Rispondi in italiano, conciso.\n\n"
    "CONVERSAZIONE PRECEDENTE:\n{storico}\n\n"
    "CONTESTO:\n{context}\n\nDOMANDA: {question}"
)


def user_id_da_email(email: str) -> str:
    riga = (
        sqlite3.connect("app.db")
        .execute("SELECT id FROM utente WHERE email = ?", (email,))
        .fetchone()
    )
    if not riga:
        sys.exit(f"Utente {email} non trovato in app.db")
    return riga[0]


def main() -> None:
    email = sys.argv[1] if len(sys.argv) > 1 else "test@test.it"
    modo = sys.argv[2] if len(sys.argv) > 2 else "values"

    uid = user_id_da_email(email)
    from langchain_core.prompts import ChatPromptTemplate

    grafo = costruisci_grafo(
        retriever_per_utente(get_embeddings(), uid, K),
        get_llm(),
        ChatPromptTemplate.from_template(_PROMPT_MINIMO),
        checkpointer=get_checkpointer(),
    )

    cid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{uid}:{cid}"}}
    domanda = "cosa mangio lunedì?"
    ingresso = {"messaggi": [HumanMessage(content=domanda)], "tentativi": 0}

    if modo == "history":
        # Time travel: serve prima una conversazione da ripercorrere.
        grafo.invoke(ingresso, config)
        print(f"\n=== STORIA DEI CHECKPOINT ({cid}) ===")
        stati = list(grafo.get_state_history(config))
        for st in stati:
            print(
                f"  {st.config['configurable']['checkpoint_id']}  "
                f"next={st.next}  messaggi={len(st.values.get('messaggi', []))}"
            )
        # Ripartire da un checkpoint passato: basta passarne l'id nella config.
        if len(stati) > 2:
            passato = stati[len(stati) // 2]
            print(
                f"\n=== RIPARTO DA {passato.config['configurable']['checkpoint_id']} "
                f"(next={passato.next}) ==="
            )
            ripreso = grafo.invoke(None, passato.config)
            print("risposta rigiocata:", ripreso.get("risposta", "")[:160])
        return

    print(f"\n=== STREAM mode={modo} — '{domanda}' ===")
    for step in grafo.stream(ingresso, config, stream_mode=modo):
        if modo == "values":
            # Stato completo dopo ogni nodo: si vede crescere.
            print(
                f"  chiavi={sorted(step)} "
                f"messaggi={len(step.get('messaggi', []))} "
                f"documenti={len(step.get('documenti', []))} "
                f"tentativi={step.get('tentativi')}"
            )
        elif modo == "updates":
            # Solo il delta prodotto da ciascun nodo: si vede CHI scrive cosa.
            for nodo, delta in step.items():
                print(f"  [{nodo}] ha scritto: {sorted(delta) if delta else '-'}")
        elif modo == "messages":
            # Token per token, come lo vedrebbe una UI in streaming.
            chunk = step[0] if isinstance(step, tuple) else step
            testo = getattr(chunk, "content", "")
            if testo:
                print(testo, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
