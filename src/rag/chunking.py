"""Punto d'ingresso del chunking: sceglie come dividere un documento.

Due strade, decise da `chunk_documento`:

1. DOCUMENTI PERSONALI (dieta, spesa, allenamento) — riconosciuti dal nome del
   file o dalla presenza di giorni della settimana. Hanno formato noto e fisso,
   quindi usano gli splitter su misura definiti in questo modulo
   (split_pasti, split_spesa, split_sessione): sono tarati bene e non vanno
   toccati.

2. TUTTO IL RESTO (normativa, manuali, contratti, circolari) — formato ignoto.
   Passa a `chunking_generico`, che riconosce da solo come il documento è
   diviso: prima dai titoli che il parser individua per dimensione e grassetto
   (split_da_parsato), poi, se il PDF non espone quegli attributi, dalle
   espressioni regolari sul testo (split_generico).

La regola per il futuro: se serve un chunker nuovo per un documento nuovo,
qualcosa non va nel ramo 2. Il ramo 1 resta chiuso ai tre formati personali.
"""

import re

from .chunking_generico import split_da_parsato, split_generico
from .config import KEEP_COMPACT

_GIORNO = re.compile(r"(?i)(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|domenica)")
_PASTO = re.compile(r"(?=(?:Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b)")
_HEADER = re.compile(
    r"(?=(?:[A-E]\s*[\x7f•·]\s*(?:Luned|Marted|Mercoled|Gioved|Venerd))"  # giorni dieta
    r"|(?:SESSIONE\s*\d)"  # sedute allenamento
    r"|(?:Reverse\s*\d)"  # settimane di un piano "reverse diet" a target crescente
    r"|(?:Note generali)|(?:Indicazioni\b)|(?:Note e terminologia))"
)
_MEAL = re.compile(r"^(Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b[^\d]*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$")

# Piani con più settimane a target crescente, es. "Reverse 1 — dal 03/08/2026":
# il numero e la data segnano quale blocco di giorni appartiene a quale
# settimana. Senza tracciarli, un chunk "Lunedì" non si distinguerebbe da un
# "Lunedì" di un'altra settimana con macro diverse (vedi CONTEXTUAL_RETRIEVAL.md).
_REVERSE_HEADER = re.compile(r"(?i)^Reverse\s*(\d+)\s*[—-]\s*dal\s*(\d{2})/(\d{2})/(\d{4})")

# --- Lista della spesa (tabelle alimento/quantità/indicazioni per reparto) ---
# Il PDF elenca ogni alimento come "■ Nome", seguito dalla quantità su una
# riga e, opzionalmente, da un'indicazione ("1 confezione") che può
# continuare sulla riga dopo.
#
# Il bullet NON si può elencare per caratteri: è un glifo di un font simbolico,
# e ogni parser lo decodifica a modo suo — pypdf restituisce "■", PyMuPDF la
# lettera "I". Elencarli tutti significherebbe legare il chunking al parser in
# uso, e un cambio di parser romperebbe il riconoscimento in silenzio (è
# successo: la lista passava da 29 chunk a 1).
#
# Il criterio è quindi posizionale: un token isolato all'inizio della riga —
# un simbolo qualsiasi, o una singola lettera che non forma parola — seguito
# dal nome dell'alimento. Regge qualunque codifica del bullet.
# Il negative lookahead esclude le righe che iniziano con una quantità: "≈2
# tavolette da 100 g" è la continuazione dell'indicazione dell'alimento
# precedente andata a capo, non un alimento nuovo.
_SPESA_ALIMENTO = re.compile(
    r"^(?:[^\w\s]|[A-Za-z](?=\s))\s*(?![\d≈~])(\S.*)$"
)
# Reparti del supermercato: fanno da titolo di sezione, e vale la pena
# tenerli nel chunk (sapere che le patate sono in "Ortofrutta" è utile).
_SPESA_REPARTO = re.compile(r"(?i)^(Dispensa|Banco frigo(?:\s*/\s*freschi)?|Ortofrutta|Freschi|Surgelati)\s*$")
# Una quantità inizia sempre con una cifra (o un ≈/~ prima della cifra):
# "12 fette", "200 g sgocciolato", "2,1–2,8 kg", "3 pz", "550 ml".
_SPESA_QUANTITA = re.compile(r"^[≈~]?\s*\d")
_SPESA_NOTE = re.compile(r"(?i)^Note\s*$")

# --- Sedute di allenamento (tabelle esercizio/serie×rep/recupero/focus) ---
_SESSIONE_START = re.compile(r"^SESSIONE\s*\d")
_SESSIONE = re.compile(r"^SESSIONE\s*(\d+)\s*[—-]\s*(.+?)\s*$", re.MULTILINE)
_SOTTOTABELLA = re.compile(r"(?=(?:RISCALDAMENTO|ALLENAMENTO)\b)")
_HEADER_TABELLA = re.compile(r"^(ESERCIZIO|SERIE\s*[×xX]\s*REP|RECUPERO|FOCUS|PESO|FEELING)$")
_SERIE_REP = re.compile(r'^\d+\s*x\s*\d+(?:/\d+)?["\']?(?:\s*x\s*\w*)?$|^\d+["\']?$')
_INIZIO_RECUPERO = re.compile(r'^(?:/|\d+.*[\'"])')
_FOCUS_NOTI = re.compile(
    r"(?i)^(Petto|Dorsale|Bicipiti|Tricipiti|Spalle|Quadricipiti|Femorali|"
    r"Centro|Polpacci|Addome|Core|Glutei|Adduttori|Abduttori)$"
)


def _norm_giorno(s: str) -> str:
    return s.lower().replace("ì", "i")


def _record(testo: str, fonte: str, *, giorno: str = "", reverse: str = "", reverse_dal: str = "", tipo: str = "", capitolo: str = "", titolo: str = "") -> dict:
    """Un chunk con tutti i metadati usati dal retriever.

    Tenerlo in una funzione sola garantisce che ogni chunk abbia ESATTAMENTE
    le stesse chiavi: `upsert_documento` le legge con `c.get(...)`, quindi una
    chiave dimenticata in un ramo diventerebbe un metadato vuoto silenzioso.
    `tipo` marca la categoria del documento ("spesa") per i boost mirati del
    RetrieverIbrido.
    """
    return {
        "testo": testo,
        "fonte": fonte,
        "giorno": giorno,
        "reverse": reverse,
        "reverse_dal": reverse_dal,
        "tipo": tipo,
        # Riferimento della sezione per i documenti strutturati ("46",
        # "2.1"), "" per gli altri: permette al retriever di raccogliere tutti
        # i pezzi di un articolo o capitolo quando la domanda lo cita.
        "capitolo": capitolo,
        # Intestazione leggibile, es. "Articolo 46 Nozione di veicolo".
        "titolo": titolo,
    }


def chunk_documento(doc: dict) -> list[dict]:
    nome = doc["fonte"].lower()
    records = []

    # Documenti di struttura ignota (normativa, manuali, contratti, circolari):
    # split_generico riconosce da solo come sono divisi — per articoli, per
    # paragrafi numerati o per capi — senza che serva una regola nuova a ogni
    # documento caricato. Va tentato PRIMA dei rami dieta/spesa/allenamento,
    # che cercano giorni e pasti e qui non troverebbero nulla.
    #
    # I documenti personali (KEEP_COMPACT e quelli con giorni della settimana)
    # NON passano di qui: restano ai loro splitter su misura, che sono tarati
    # meglio di qualsiasi euristica generica.
    if not any(k in nome for k in KEEP_COMPACT) and not _GIORNO.search(
        doc["testo"][:3000]
    ):
        # Prima si prova con la struttura del PDF (titoli riconosciuti dal
        # font), poi con le regex sul testo: il primo criterio vale su
        # qualsiasi documento, il secondo solo su quelli che numerano le
        # sezioni in un modo previsto.
        generici = split_da_parsato(doc["parsato"]) if doc.get("parsato") else []
        if not generici:
            generici = split_generico(doc["testo"])
        if generici:
            return [
                _record(
                    g["testo"],
                    doc["fonte"],
                    tipo="documento",
                    capitolo=g["sezione"],
                    titolo=g["titolo"],
                )
                for g in generici
            ]

    if any(k in nome for k in KEEP_COMPACT):
        # Un chunk per alimento, così una domanda su un singolo prodotto
        # ("quante fette biscottate compro?") recupera la riga giusta invece
        # di un blocco con 40 alimenti insieme. Se il PDF non ha la forma
        # attesa (elenco con bullet), split_spesa non riconosce nulla e
        # ricadiamo sullo split generico di prima.
        pezzi = split_spesa(doc["testo"]) or split_ricorsivo(doc["testo"], max_caratteri=1500)
        for p in pezzi:
            records.append(_record(p, doc["fonte"], tipo="spesa"))
    else:
        # Numero e data d'inizio del blocco "Reverse N" attualmente in corso
        # di lettura: split_strutturato produce anche un blocco che INIZIA
        # con l'header "Reverse N — dal ...", quindi lo intercettiamo qui e
        # lo applichiamo a tutti i blocchi (giorni) successivi, finché non
        # incontriamo il prossimo header "Reverse".
        reverse_corrente, reverse_dal_corrente = "", ""
        for blocco in split_strutturato(doc["testo"]):
            r = _REVERSE_HEADER.match(blocco)
            if r:
                reverse_corrente = r.group(1)
                reverse_dal_corrente = f"{r.group(4)}-{r.group(3)}-{r.group(2)}"  # ISO: AAAA-MM-GG

            g = _GIORNO.search(blocco[:20])
            if g:
                giorno = _norm_giorno(g.group(0))
                for p in split_pasti(blocco):
                    # Anteponiamo "Reverse N" anche al TESTO del chunk (non
                    # solo al metadato): sia perché la ricerca lessicale del
                    # RetrieverIbrido cerca parole nel testo, sia perché
                    # l'LLM finale legge solo il testo, non i metadati.
                    testo = f"Reverse {reverse_corrente} — {p}" if reverse_corrente else p
                    records.append(
                        _record(
                            testo,
                            doc["fonte"],
                            giorno=giorno,
                            reverse=reverse_corrente,
                            reverse_dal=reverse_dal_corrente,
                            tipo="dieta",
                        )
                    )
            elif _SESSIONE_START.match(blocco):
                for p in split_sessione(blocco):
                    records.append(_record(p, doc["fonte"], tipo="allenamento"))
            else:
                pezzi = split_ricorsivo(blocco, 800) if len(blocco) > 1200 else [re.sub(r"\s+", " ", blocco)]
                for p in pezzi:
                    records.append(_record(p, doc["fonte"]))
    return records


def split_pasti(blocco: str) -> list[str]:
    m = _GIORNO.search(blocco[:20])
    giorno = m.group(0).capitalize() if m else "Giorno"
    out = []
    for p in _PASTO.split(blocco):
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        mm = _MEAL.match(p)
        if mm:
            pasto, carb, prot, gr, kcal, cibi = mm.groups()
            out.append(
                f"{giorno}, {pasto.lower()}: {cibi.strip()} "
                f"(macro: {carb}g carboidrati, {prot}g proteine, {gr}g grassi, {kcal} kcal)"
            )
        else:
            out.append(f"{giorno} — {p}")  # es. riga dei totali giornalieri
    return out


def split_spesa(testo: str) -> list[str]:
    """Un chunk per alimento della lista della spesa, più uno per le note.

    Trasforma la tabella "■ Fette biscottate / 12 fette / 1 confezione" nella
    frase "Lista della spesa, dispensa: Fette biscottate — 12 fette (1
    confezione)". Serve perché con un solo chunk-blob per tutta la lista
    l'LLM riceve 40 alimenti mescolati e fatica a riportare la quantità
    giusta del singolo alimento richiesto (vedi split_pasti per la stessa
    idea applicata ai pasti).
    """
    righe = [r.strip() for r in testo.splitlines() if r.strip()]
    out: list[str] = []
    reparto = ""
    corrente: dict | None = None
    note: list[str] = []
    in_note = False

    def chiudi():
        nonlocal corrente
        if corrente:
            out.append(_format_spesa(reparto, corrente))
            corrente = None

    for riga in righe:
        if _SPESA_NOTE.match(riga):
            chiudi()
            in_note = True
            continue
        if in_note:
            note.append(riga)
            continue
        if _SPESA_REPARTO.match(riga):
            chiudi()
            reparto = riga
            continue
        if _HEADER_TABELLA.match(riga):  # "Alimento"/"Quantità"/"Indicazioni"
            continue

        alimento = _SPESA_ALIMENTO.match(riga)
        if alimento:
            chiudi()
            corrente = {"nome": alimento.group(1).strip(), "quantita": "", "indicazioni": []}
        elif corrente is None:
            continue  # intestazione del documento, prima del primo alimento
        elif not corrente["quantita"] and _SPESA_QUANTITA.match(riga):
            corrente["quantita"] = riga
        else:
            corrente["indicazioni"].append(riga)
    chiudi()

    if note:
        out.append(f"Lista della spesa, note: {' '.join(note)}")
    return out


def _format_spesa(reparto: str, voce: dict) -> str:
    testa = f"Lista della spesa, {reparto.lower()}" if reparto else "Lista della spesa"
    riga = f"{testa}: {voce['nome']}"
    if voce["quantita"]:
        riga += f" — {voce['quantita']}"
    if voce["indicazioni"]:
        riga += f" ({' '.join(voce['indicazioni'])})"
    return riga


def split_sessione(blocco: str) -> list[str]:
    m = _SESSIONE.match(blocco)
    if not m:
        return [re.sub(r"\s+", " ", blocco).strip()]
    numero, nome_sessione = m.group(1), m.group(2)

    out = []
    for parte in _SOTTOTABELLA.split(blocco):
        parte = parte.strip()
        if not parte:
            continue
        prima_riga, _, resto = parte.partition("\n")
        etichetta = prima_riga.strip().lower()
        if etichetta not in ("riscaldamento", "allenamento"):
            continue
        # una riga vuota separa la tabella dall'intestazione ripetuta di pagina
        # (es. "Scheda di allenamento — ...") che precede la prossima sessione
        resto = resto.split("\n\n", 1)[0]
        righe = [r.strip() for r in resto.splitlines() if r.strip() and not _HEADER_TABELLA.match(r.strip())]
        for es in _parsa_esercizi(righe):
            out.append(_format_esercizio(numero, nome_sessione, etichetta, es))
    return out


def _parsa_esercizi(righe: list[str]) -> list[dict]:
    esercizi = []
    corrente = {"nome": [], "serie_rep": [], "recupero": [], "focus": []}
    stato = "NOME"

    def chiudi_esercizio():
        if corrente["nome"] or corrente["serie_rep"]:
            esercizi.append(corrente)

    def nuovo_esercizio():
        nonlocal corrente
        chiudi_esercizio()
        corrente = {"nome": [], "serie_rep": [], "recupero": [], "focus": []}

    for r in righe:
        if stato == "NOME":
            if _SERIE_REP.match(r):
                corrente["serie_rep"].append(r)
                stato = "SERIE_REP"
            else:
                corrente["nome"].append(r)
        elif stato == "SERIE_REP":
            if _INIZIO_RECUPERO.match(r):
                corrente["recupero"].append(r)
                stato = "RECUPERO"
            else:
                corrente["serie_rep"].append(r)
        else:  # stato == "RECUPERO"
            if _SERIE_REP.match(r):
                # innesco serie×rep senza passare per un nuovo nome: capita solo
                # se il nome del prossimo esercizio è vuoto (non osservato nei dati)
                nuovo_esercizio()
                corrente["serie_rep"].append(r)
                stato = "SERIE_REP"
            elif not corrente["focus"] and _FOCUS_NOTI.match(r):
                corrente["focus"].append(r)
            elif corrente["focus"] and r.count(" ") == 0 and r[:1].islower():
                # singola parola minuscola dopo il focus: sua continuazione
                # su due righe (es. "schiena" dopo "Centro")
                corrente["focus"].append(r)
            elif not corrente["focus"] and r[:1].islower():
                # riga minuscola prima del focus: continuazione del recupero
                # multi-riga (es. "braccio e l'altro" dopo "30-60\" tra un")
                corrente["recupero"].append(r)
            else:
                # non è continuazione: è il nome del prossimo esercizio
                nuovo_esercizio()
                corrente["nome"].append(r)
                stato = "NOME"
    chiudi_esercizio()
    return esercizi


def _unisci_nome(righe: list[str]) -> str:
    nome = righe[0]
    for riga in righe[1:]:
        if riga.startswith("("):
            nome += f" {riga}"
        elif riga.startswith("-"):
            nome += f", {riga.lstrip('- ').strip()}"
        else:
            nome += f", {riga}"
    return nome


def _format_esercizio(numero: str, nome_sessione: str, etichetta: str, es: dict) -> str:
    nome = _unisci_nome(es["nome"]) if es["nome"] else "?"
    serie_rep = re.sub(r"\s+", " ", " ".join(es["serie_rep"])).strip()
    recupero = re.sub(r"\s+", " ", " ".join(es["recupero"])).strip()
    riga = f"Sessione {numero} — {nome_sessione}, {etichetta}: {nome} — {serie_rep}, recupero {recupero}"
    if es["focus"]:
        focus = re.sub(r"\s+", " ", " ".join(es["focus"])).strip()
        riga += f", focus {focus}"
    return riga


def split_strutturato(testo: str) -> list[str]:
    parti = [p.strip() for p in _HEADER.split(testo) if p.strip()]
    return parti if parti else [testo]


def split_ricorsivo(testo: str, max_caratteri: int = 600, separatori: list[str] | None = None) -> list[str]:
    if separatori is None:
        separatori = ["\n\n", "\n", ". ", " ", ""]
    if len(testo) <= max_caratteri:
        return [testo.strip()] if testo.strip() else []
    sep, resto = separatori[0], separatori[1:]

    if sep == "":
        return [testo[i : i + max_caratteri] for i in range(0, len(testo), max_caratteri)]

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
