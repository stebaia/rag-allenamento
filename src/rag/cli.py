"""Entry point da riga di comando: indicizza la cartella 'documenti/' su Qdrant
(sotto uno user_id fisso 'cli') e apre il loop di Q&A. Utile per test manuali
rapidi dei moduli condivisi con l'API, senza passare da upload HTTP."""

import os
import uuid

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .chunking import chunk_documento
from .config import CARTELLA_DOCUMENTI, K, MODELLO_EMBED, MODELLO_LLM
from .graph import costruisci_grafo
from .loaders import carica_documenti
from .vectorstore import retriever_per_utente, upsert_documento

_CLI_USER_ID = "cli"


def main():
    load_dotenv()  # legge .env nella cwd, se presente

    if not os.environ.get("OPENAI_API_KEY"):
        print("[!] Manca OPENAI_API_KEY. Impostala e rilancia.")
        return

    # 1) componenti base
    embeddings = HuggingFaceEmbeddings(model_name=MODELLO_EMBED)

    print("Indicizzo la cartella 'documenti/' su Qdrant...")
    for doc in carica_documenti(CARTELLA_DOCUMENTI):
        chunks = chunk_documento(doc)
        n = upsert_documento(embeddings, _CLI_USER_ID, str(uuid.uuid4()), chunks)
        print(f"  {doc['fonte']}: {n} chunk")

    retriever = retriever_per_utente(embeddings, _CLI_USER_ID, K)
    llm = ChatOpenAI(model=MODELLO_LLM, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Sei un assistente su dieta, spesa e allenamento. Rispondi usando SOLO il "
        "contesto. Se l'informazione non c'è, dillo. Rispondi in italiano, conciso.\n\n"
        "CONTESTO:\n{context}\n\nDOMANDA: {question}"
    )

    # 2) grafo agentico
    grafo = costruisci_grafo(retriever, llm, prompt)

    # 3) ciclo domande
    print("\nPronto. Esempi: 'cosa mangio a pranzo martedì?', 'esercizi gambe?'\n")
    while True:
        domanda = input("Domanda (invio vuoto per uscire): ").strip()
        if not domanda:
            break
        risultato = grafo.invoke({"domanda": domanda, "tentativi": 0})
        print("\nRisposta:\n" + risultato["risposta"] + "\n")


if __name__ == "__main__":
    main()
