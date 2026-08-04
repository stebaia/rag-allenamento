"""Chunking per documenti di struttura ignota (normativa, manuali, circolari).

Il resto di chunking.py è tarato sui documenti personali dell'utente (dieta,
spesa, allenamento): formati noti, regex su misura, risultati precisi. Quel
modello però non regge quando l'utente carica un PDF qualsiasi, perché ogni
formato nuovo richiederebbe il suo splitter.

Qui l'approccio è rovesciato: invece di riconoscere OGNI formato, si
riconoscono i pochi modi in cui i documenti segnano le proprie divisioni —
"Articolo 46", "3.2.1. Titolo", "Capo II", un titolo in maiuscolo — e si
sceglie automaticamente quale usare in base a quanti confini produce sul
testo che si sta indicizzando.

Ogni chunk porta con sé due metadati che il retriever usa per i boost:
- `sezione`: il riferimento del blocco ("46", "3.2.1"), quello che l'utente
  cita nelle domande ("spiegami l'articolo 46");
- `titolo`: l'intestazione leggibile, ripetuta in testa ai pezzi di un blocco
  lungo perché non perdano il contesto.
"""

import re

# I livelli di divisione, dal più fine al più grosso. L'ordine conta: si sceglie
# il PRIMO che produce abbastanza confini, perché è quello che dà i blocchi più
# specifici — su un codice, "Articolo 46" è più utile di "Titolo III".
#
# Ogni voce: (nome, regex con gruppo 1 = riferimento, quanti confini servono).
# Le regex sono ancorate a inizio riga: un "articolo 46" citato dentro una
# frase non è una divisione, è un rimando.
_LIVELLI = [
    (
        "articolo",
        re.compile(r"(?m)^[ \t]*(?:Articolo|Art\.)\s*(\d+(?:[-/]?\w+)?)\b[ \t]*"),
        5,
    ),
    (
        "paragrafo",
        re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+){1,3})\.?\s+(?=\S)"),
        5,
    ),
    (
        "capo",
        re.compile(
            r"(?im)^[ \t]*(?:Capitolo|Capo|Titolo|Sezione|Parte)\s+"
            r"([IVXLC]+|\d+)\b[ \t]*"
        ),
        3,
    ),
]

# Righe che nei PDF impaginati sono rumore: voci di indice con i puntini o il
# rimando alla pagina, e i numeri di pagina isolati.
_RIGA_INDICE = re.compile(
    r"(?m)^.*?(?:\.{5,}\s*(?:Pag\.|pag\.|»)?\s*\d*|\s»\s*\d+)\s*$"
)
_NUMERO_PAGINA = re.compile(r"(?m)^\s*-?\s*\d{1,4}\s*-?\s*$")

# Un blocco più corto di così è quasi sempre un titolo rimasto orfano o una
# riga di indice sopravvissuta: non risponde a nessuna domanda.
_MINIMO_UTILE = 180

# Oltre questa soglia un blocco viene spezzato: sopra, l'informazione cercata
# rischia di annegare in mezzo ad altro nel contesto passato all'LLM.
_MAX_CHUNK = 1200


_VALORI_ROMANI = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _da_romano(s: str) -> int:
    """Converte un numero romano in intero.

    I documenti scrivono "Capitolo IV", l'utente chiede "il capitolo 4": il
    retriever usa questa conversione per far combaciare la domanda col
    metadato della sezione (vedi vectorstore.py).
    """
    totale = 0
    for i, c in enumerate(s):
        v = _VALORI_ROMANI[c]
        # Notazione sottrattiva: IV = 5-1. Se il valore successivo e' maggiore,
        # questo va sottratto invece che sommato.
        successivo = _VALORI_ROMANI.get(s[i + 1]) if i + 1 < len(s) else None
        totale += -v if successivo and successivo > v else v
    return totale


def _pulisci(testo: str) -> str:
    testo = _RIGA_INDICE.sub("", testo)
    testo = _NUMERO_PAGINA.sub("", testo)
    # Gli a capo dentro un paragrafo sono artefatti dell'impaginazione PDF;
    # le righe vuote invece separano davvero i blocchi, e vanno tenute.
    return re.sub(r"[ \t]+", " ", testo)


def _scegli_livello(testo: str):
    """Il livello di divisione che struttura meglio QUESTO documento.

    Non basta prendere il primo che supera una soglia: il libro di Mazzuti
    contiene una decina di "Articolo N" (citazioni di norme dentro il testo)
    ma è organizzato in 128 paragrafi "3.2.1", e dividerlo per articoli
    produrrebbe nove blocchi enormi invece di centoventotto sensati.

    Si sceglie quindi il livello con più riferimenti DISTINTI, cioè quello che
    il documento usa davvero come indice. A parità vince il più fine, per
    l'ordine in cui i livelli sono elencati. Se nessuno raggiunge la propria
    soglia, il documento non ha struttura riconoscibile e si torna [].
    """
    migliore = (None, [], 0)
    for nome, regex, minimo in _LIVELLI:
        confini = list(regex.finditer(testo))
        if len(confini) < minimo:
            continue
        distinti = len({m.group(1) for m in confini})
        if distinti > migliore[2]:
            migliore = (nome, confini, distinti)
    return migliore[0], migliore[1]


def _titolo_del_blocco(blocco: str, riferimento: str, nome_livello: str) -> str:
    """Un'etichetta leggibile per il blocco, es. "Articolo 46 Nozione di veicolo".

    Serve sia come metadato sia come intestazione dei pezzi quando il blocco
    va spezzato: senza, il secondo pezzo di un articolo lungo parla di "comma
    3" senza che si sappia più di quale articolo.
    """
    prima_riga = blocco.split("\n", 1)[0].strip()
    etichetta = {
        "articolo": f"Articolo {riferimento}",
        "paragrafo": riferimento,
        "capo": f"Capitolo {riferimento}",
    }[nome_livello]
    # Se la prima riga ripete già il riferimento, evitiamo di duplicarlo.
    if riferimento in prima_riga[:40]:
        return prima_riga[:110]
    return f"{etichetta} {prima_riga}"[:110]


def _spezza_lungo(testo: str, massimo: int) -> list[str]:
    """Divide un testo troppo lungo sui confini naturali più vicini."""
    if len(testo) <= massimo:
        return [testo]
    pezzi, corrente = [], ""
    # Prima si prova a rompere fra paragrafi, poi fra frasi: tagliare a metà
    # di una frase rende il chunk inutilizzabile da entrambi i lati.
    for parte in re.split(r"(?<=\n)\n+|(?<=\.)\s+(?=[A-Z0-9])", testo):
        if len(corrente) + len(parte) + 1 <= massimo:
            corrente = f"{corrente} {parte}".strip() if corrente else parte
        else:
            if corrente:
                pezzi.append(corrente)
            if len(parte) > massimo:
                # Frase singola più lunga del massimo: taglio netto, non c'è
                # confine naturale da rispettare.
                pezzi += [
                    parte[i : i + massimo] for i in range(0, len(parte), massimo)
                ]
                corrente = ""
            else:
                corrente = parte
    if corrente:
        pezzi.append(corrente)
    return pezzi


def split_da_parsato(parsato, max_caratteri: int = _MAX_CHUNK) -> list[dict]:
    """Divide un documento usando i titoli riconosciuti dal parser.

    È la via preferita quando il PDF è stato letto da un parser che espone gli
    attributi tipografici: un titolo si riconosce perché è in grassetto o più
    grande del corpo, criterio che vale su qualsiasi documento. Le espressioni
    regolari restano come ripiego per i PDF senza questi attributi.

    Il riferimento di sezione ("46", "2.1") viene comunque estratto dal testo
    del titolo, perché è quello che l'utente cita nelle domande.
    """
    righe = getattr(parsato, "righe", None)
    if not righe:
        return []

    # Una riga apre una sezione se il parser la riconosce come titolo OPPURE se
    # inizia con un riferimento esplicito ("Articolo 46"). Il solo criterio
    # tipografico non basta: nel Codice della Strada i titoli in grassetto sono
    # i "Capo I", mentre "Articolo 46" è composto come testo normale — e senza
    # questo secondo criterio l'articolo 46 finirebbe dentro il chunk del Capo,
    # perdendo il riferimento con cui l'utente lo cerca.
    grezzi = [
        i
        for i, r in enumerate(righe)
        if parsato.e_titolo(r) or _APRE_SEZIONE.match(r.testo.strip())
    ]
    if len(grezzi) < 3:
        return []

    # Righe di titolo consecutive sono UNA intestazione spezzata
    # dall'impaginazione, non sezioni diverse. Nel Codice della Strada
    # "Articolo 46", la fonte normativa e "Nozione di veicolo" stanno su tre
    # righe tutte in grassetto: trattarle come tre confini produce tre blocchi
    # troppo corti, che vengono accorpati al precedente facendo perdere il
    # riferimento "46" con cui l'utente cerca l'articolo.
    confini = [i for j, i in enumerate(grezzi) if j == 0 or i - grezzi[j - 1] > 1]
    if len(confini) < 3:
        return []

    chunk = []
    for n, inizio in enumerate(confini):
        fine = confini[n + 1] if n + 1 < len(confini) else len(righe)
        # L'intestazione può occupare più righe consecutive (numero, fonte,
        # rubrica): le uniamo tutte, così "Articolo 46" e "Nozione di veicolo"
        # stanno nello stesso titolo.
        pezzi_titolo = []
        for r in righe[inizio:fine]:
            if not (parsato.e_titolo(r) or _APRE_SEZIONE.match(r.testo.strip())):
                break
            pezzi_titolo.append(r.testo.strip())
        titolo = " ".join(pezzi_titolo) or righe[inizio].testo.strip()
        corpo = "\n".join(r.testo for r in righe[inizio:fine]).strip()
        if len(corpo) < _MINIMO_UTILE:
            # Titolo senza corpo (o quasi): lo si accorpa al blocco precedente
            # invece di perderlo, così non si crea un chunk vuoto.
            if chunk and len(chunk[-1]["testo"]) + len(corpo) <= max_caratteri:
                chunk[-1]["testo"] += f"\n{corpo}"
            continue

        riferimento = _riferimento_dal_titolo(titolo)
        for j, pezzo in enumerate(_spezza_lungo(corpo, max_caratteri)):
            chunk.append(
                {
                    "testo": pezzo if j == 0 else f"{titolo[:110]} (segue)\n{pezzo}",
                    "sezione": riferimento,
                    "titolo": titolo[:110],
                }
            )
    return chunk


# Una riga che apre una sezione numerata, anche se tipograficamente è testo
# normale: "Articolo 46", "Art. 12", "3.2.1. Titolo".
_APRE_SEZIONE = re.compile(
    r"(?i)^(?:Articolo|Art\.)\s*\d+\b|^\d+(?:\.\d+){1,3}\.?\s+\S"
)

# Il numero dentro un titolo: "Articolo 46", "3.2.1. Titolo", "Capo II".
# I riferimenti numerici hanno la precedenza su quelli romani: in un codice
# "Articolo 46" identifica il contenuto meglio del "Capo I" che lo contiene.
_RIFERIMENTO_NUMERICO = re.compile(
    r"(?i)^\W*(?:Articolo|Art\.)?\s*(\d+(?:\.\d+)*)\b"
)
_RIFERIMENTO = re.compile(
    r"(?i)^\W*(?:Articolo|Art\.|Capitolo|Capo|Titolo|Sezione|Parte)?\s*"
    r"(\d+(?:\.\d+)*|[IVXLC]+)\b"
)


# "Articolo 46" ovunque compaia nell'intestazione, non solo in testa: quando
# più livelli sono composti insieme ("TITOLO III ... Capo I ... Articolo 46")
# l'unità che l'utente cerca è l'articolo, non il titolo che lo contiene.
_ARTICOLO_OVUNQUE = re.compile(r"(?i)\b(?:Articolo|Art\.)\s*(\d+(?:[-/]?\w+)?)\b")


def _riferimento_dal_titolo(titolo: str) -> str:
    """Il riferimento più specifico contenuto nell'intestazione."""
    m = _ARTICOLO_OVUNQUE.search(titolo)
    if m:
        return m.group(1)
    m = _RIFERIMENTO_NUMERICO.match(titolo) or _RIFERIMENTO.match(titolo)
    return m.group(1) if m else ""


def split_generico(testo: str, max_caratteri: int = _MAX_CHUNK) -> list[dict]:
    """Divide un documento di struttura ignota.

    Ritorna una lista di dict {"testo", "sezione", "titolo"}, oppure [] se il
    documento non ha una struttura riconoscibile: in quel caso il chiamante
    ricade sullo split a lunghezza fissa, che funziona sempre ma senza
    metadati.
    """
    pulito = _pulisci(testo)
    nome_livello, confini = _scegli_livello(pulito)
    if not confini:
        return []

    chunk = []

    # Il testo che precede il primo confine (premessa, introduzione) non
    # appartiene a nessuna sezione ma va indicizzato lo stesso: su una
    # circolare discorsiva può essere metà del documento.
    testa = pulito[: confini[0].start()].strip()
    if len(testa) >= _MINIMO_UTILE:
        for pezzo in _spezza_lungo(testa, max_caratteri):
            chunk.append({"testo": pezzo, "sezione": "", "titolo": ""})

    for i, m in enumerate(confini):
        fine = confini[i + 1].start() if i + 1 < len(confini) else len(pulito)
        blocco = pulito[m.start() : fine].strip()
        if len(blocco) < _MINIMO_UTILE:
            # Blocco troppo corto per stare in piedi da solo (di solito un
            # titolo isolato): lo si attacca al precedente invece di buttarlo,
            # altrimenti si perde testo.
            if chunk and len(chunk[-1]["testo"]) + len(blocco) <= max_caratteri:
                chunk[-1]["testo"] += f"\n{blocco}"
            continue

        riferimento = m.group(1)
        titolo = _titolo_del_blocco(blocco, riferimento, nome_livello)

        for j, pezzo in enumerate(_spezza_lungo(blocco, max_caratteri)):
            chunk.append(
                {
                    "testo": pezzo if j == 0 else f"{titolo} (segue)\n{pezzo}",
                    "sezione": riferimento,
                    "titolo": titolo,
                }
            )
    return chunk
