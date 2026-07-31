import os, glob, re
from typing import TypedDict, List
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# ---- CONFIG ----
CARTELLA_DOCUMENTI = "documenti"
CHROMA_PATH        = "chroma_lc"
MODELLO_EMBED      = "paraphrase-multilingual-MiniLM-L12-v2"   # locale, gratis
MODELLO_LLM        = "gpt-4o-mini"
K                  = 5                                          # blocchi recuperati
KEEP_COMPACT = ("spesa", "lista", "shopping", "grocery")
_GIORNO = re.compile(r'(?i)(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|domenica)')
_PASTO  = re.compile(r'(?=(?:Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b)')
_HEADER = re.compile(
    r'(?=(?:[A-E]\s*[\x7f•·]\s*(?:Luned|Marted|Mercoled|Gioved|Venerd))'  # giorni dieta
    r'|(?:SESSIONE\s*\d)'                                                 # sedute allenamento
    r'|(?:Note generali)|(?:Indicazioni\b)|(?:Note e terminologia))'
)
_MEAL = re.compile(r'^(Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b[^\d]*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$')


# In alternativa a export OPENAI_API_KEY=... puoi impostarla qui (ma NON
# condividere il file con la chiave dentro):
# os.environ["OPENAI_API_KEY"] = "sk-..."


class Stato(TypedDict):
    domanda: str
    documenti: List[Document]
    risposta: str
    tentativi: int


def _norm_giorno(s):
    return s.lower().replace("ì", "i")

def ottieni_vectorstore(embeddings) -> Chroma:
    """Costruisce l'indice usando il TUO chunking per-pasto, poi lo ricarica da disco."""
    if os.path.exists(CHROMA_PATH):
        print("Carico l'indice da disco...")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print("Costruisco l'indice con il chunking su misura...")
    docs_lc = []
    for doc in carica_documenti(CARTELLA_DOCUMENTI):        # le tue funzioni
        for c in chunk_documento(doc):
            docs_lc.append(Document(
                page_content=c["testo"],
                metadata={"fonte": c["fonte"], "giorno": c.get("giorno", "")},
            ))
    print(f"  {len(docs_lc)} chunk su misura")
    return Chroma.from_documents(docs_lc, embeddings, persist_directory=CHROMA_PATH)


def chunk_documento(doc):
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
                pezzi = split_ricorsivo(blocco, 800) if len(blocco) > 1200 else [re.sub(r'\s+', ' ', blocco)]
                for p in pezzi:
                    records.append({"testo": p, "fonte": doc["fonte"], "giorno": ""})
    return records

def split_pasti(blocco):
    m = _GIORNO.search(blocco[:20])
    giorno = m.group(0).capitalize() if m else "Giorno"
    out = []
    for p in _PASTO.split(blocco):
        p = re.sub(r'\s+', ' ', p).strip()
        if not p:
            continue
        mm = _MEAL.match(p)
        if mm:
            pasto, carb, prot, gr, kcal, cibi = mm.groups()
            out.append(f"{giorno}, {pasto.lower()}: {cibi.strip()} "
                       f"(macro: {carb}g carboidrati, {prot}g proteine, {gr}g grassi, {kcal} kcal)")
        else:
            out.append(f"{giorno} — {p}")     # es. riga dei totali giornalieri
    return out

def split_strutturato(testo):
    parti = [p.strip() for p in _HEADER.split(testo) if p.strip()]
    return parti if parti else [testo]

def split_ricorsivo(testo, max_caratteri = 600, separatori= None): 
    if separatori is None:
        separatori = ["\n\n", "\n", ". ", " ", ""]
    if len(testo) <= max_caratteri:
        return [testo.strip()] if testo.strip() else []
    sep, resto = separatori[0], separatori[1:]

    if sep == "":
        return [testo[i:i + max_caratteri] for i in range(0, len(testo), max_caratteri)]

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


def carica_documenti(cartella):
    documenti = []
    percorsi = sorted(
        glob.glob(os.path.join(cartella, "*.pdf")) +
        glob.glob(os.path.join(cartella, "*.txt"))
    )
    if not percorsi:
        print(f"[!] Nessun file in '{cartella}/'. Mettici i tuoi PDF e rilancia.")
        return documenti

    for percorso in percorsi:
        fonte = os.path.basename(percorso)
        if percorso.lower().endswith(".pdf"):
            reader = PdfReader(percorso)
            testo = "\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            with open(percorso, encoding="utf-8") as f:
                testo = f.read()

        if testo.strip():
            documenti.append({"fonte": fonte, "testo": testo})
        else:
            # PDF probabilmente scansionato (una "foto" del testo): servirebbe l'OCR
            print(f"[!] '{fonte}' non contiene testo estraibile (forse scansionato). Lo salto.")
    return documenti



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
    builder.add_conditional_edges("recupera", decidi,
                                  {"genera": "genera", "riformula": "riformula"})
    builder.add_edge("riformula", "recupera")   # ciclo: torna a recuperare
    builder.add_edge("genera", END)
    return builder.compile()



def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("[!] Manca OPENAI_API_KEY. Impostala e rilancia.")
        return

    # 1) componenti base
    embeddings  = HuggingFaceEmbeddings(model_name=MODELLO_EMBED)
    vectorstore = ottieni_vectorstore(embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": K})
    llm         = ChatOpenAI(model=MODELLO_LLM, temperature=0)
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