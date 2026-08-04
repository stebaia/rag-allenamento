"""Tool esposti all'LLM nel grafo agentico.

Differenza rispetto a nodo_recupera: lì il retrieval è una tappa OBBLIGATA del
flusso, qui è una capacità che il modello può decidere di usare — anche più
volte, con query diverse. La docstring di ogni tool non è documentazione per
noi: è il testo che l'LLM legge per decidere se e come chiamarlo, quindi va
scritta pensando a lui.
"""

import uuid
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import interrupt


def crea_tool_ricerca(retriever):
    """Fabbrica: il retriever è già filtrato per utente, quindi il tool va
    creato per-richiesta (non può essere una costante di modulo)."""

    @tool
    def cerca_documenti(query: str) -> str:
        """Cerca nell'archivio di documenti personali dell'utente.

        L'archivio contiene documenti di QUALSIASI genere caricati
        dall'utente: piani alimentari, liste della spesa e schede di
        allenamento, ma anche contratti, circolari, normativa e interi libri
        tecnici. Non sai in anticipo cosa contenga.

        USA SEMPRE questo tool prima di rispondere, anche quando la domanda
        sembra riguardare un argomento che non conosci o un libro che pensi di
        non avere: quel documento è probabilmente nell'archivio. Non dire mai
        all'utente di consultare altrove senza aver prima cercato qui.

        Usa query specifiche e ricche di parole chiave. Se la domanda riguarda
        due cose diverse (es. i pasti e cosa comprare, o due capitoli), fai
        ricerche separate.

        Args:
            query: cosa cercare, es. "colazione lunedì", "lista spesa verdura",
                "capitolo 2 regolamento riscossione coattiva"
        """
        print(f"   ↳ [tool] cerca_documenti({query!r})")
        docs = retriever.invoke(query)
        if not docs:
            return "Nessun documento trovato per questa ricerca."
        return "\n\n".join(d.page_content for d in docs)

    return cerca_documenti


def crea_tool_memoria(user_id: str, conferma: bool = False):
    """Fabbrica del tool di scrittura nello store.

    Il namespace ("memorie", user_id) è chiuso sull'utente: è il confine di
    sicurezza che impedisce a un utente di leggere le memorie di un altro,
    esattamente come il prefisso sul thread_id per il checkpointer.

    Con `conferma=True` la scrittura è sospesa da interrupt() finché l'utente
    non risponde: è la mitigazione al rischio del "fatto sbagliato" (§3.3
    Test D), un fatto errato che si auto-perpetua senza comparire in chat.
    """

    @tool
    def ricorda(fatto: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Salva un'informazione duratura e personale sull'utente: preferenze,
        intolleranze, obiettivi, abitudini di allenamento.

        NON usarlo per fatti già presenti nei documenti né per domande estemporanee.

        Args:
            fatto: il fatto da ricordare, in una frase, es. "è intollerante al lattosio"
        """
        print(f"   ↳ [tool] ricorda({fatto!r})")
        if conferma:
            # interrupt() SOSPENDE l'esecuzione: il valore torna al chiamante e
            # lo stato resta congelato nel checkpointer. Si riprende con
            # grafo.invoke(Command(resume="sì"), config).
            risposta = interrupt({"conferma_memoria": fatto})
            if str(risposta).strip().lower() not in ("sì", "si", "yes", "ok"):
                return "L'utente ha rifiutato: non ho salvato nulla."
        # InjectedStore: lo store arriva a runtime da LangGraph e NON è un
        # parametro che l'LLM deve riempire — non compare nello schema del tool.
        store.put(("memorie", user_id), str(uuid.uuid4()), {"fatto": fatto})
        return f"Ricordato: {fatto}"

    return ricorda
