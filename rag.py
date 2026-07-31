import os, glob, re
import numpy as np
import math
import chromadb, hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

os.environ["OPENAI_API_KEY"] = ""
CARTELLA = "documenti"
KEEP_COMPACT = ("spesa", "lista", "shopping", "grocery")

CHROMA_PATH = "chroma_db"
class Indice:
    def __init__(self, path=CHROMA_PATH, nome="documenti"):
        self.nome = nome
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(nome)
        self.vettori, self.meta, self.idf = None, [], {}
        self._carica_in_memoria()          # se c'è già roba su disco, la carico

    def _tok(self, testo):
        return set(w for w in re.findall(r'\w+', testo.lower()) if len(w) > 2)

    def _calcola_idf(self):
        N = len(self.meta) or 1
        df = {}
        for m in self.meta:
            for w in self._tok(m["testo"]):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def _carica_in_memoria(self):
        dati = self.collection.get(include=["embeddings", "documents", "metadatas"])
        if dati["ids"]:
            self.vettori = np.array(dati["embeddings"])
            self.meta = [{"testo": d, "fonte": m["fonte"], "giorno": m.get("giorno", "")}
                        for d, m in zip(dati["documents"], dati["metadatas"])]
            self._calcola_idf()

    def vuoto(self):
        return self.collection.count() == 0

    def reset(self):
        self.client.delete_collection(self.nome)
        self.collection = self.client.get_or_create_collection(self.nome)
        self.vettori, self.meta, self.idf = None, [], {}

    def aggiungi(self, chunk):
        emb = crea_embedding([c["testo"] for c in chunk])
        base = self.collection.count()
        self.collection.add(
            ids=[f"id_{base + i}" for i in range(len(chunk))],
            embeddings=emb.tolist(),
            documents=[c["testo"] for c in chunk],
            metadatas=[{"fonte": c["fonte"], "giorno": c.get("giorno", "")} for c in chunk],
        )
        self.vettori = emb if self.vettori is None else np.vstack([self.vettori, emb])
        self.meta.extend(chunk)
        self._calcola_idf()

    def cerca(self, domanda, k=8):
        q = crea_embedding([domanda])[0]
        sem = self.vettori @ q
        termini = self._tok(domanda)
        lex = np.array([
            sum(self.idf.get(w, 0) for w in termini if w in self._tok(m["testo"]))
            for m in self.meta
        ])
        if lex.max() > 0:
            lex = lex / lex.max()
        punteggi = sem + 0.4 * lex
        ordine = list(np.argsort(punteggi)[::-1])

        # se la domanda nomina un giorno, includo TUTTI i pasti di quel giorno
        g = _GIORNO.search(domanda.lower())
        if g:
            giorno = _norm_giorno(g.group(0))
            del_giorno = [i for i in ordine if self.meta[i].get("giorno") == giorno]
            altri = [i for i in ordine if i not in set(del_giorno)]
            scelti = (del_giorno + altri)[:max(k, len(del_giorno) + 2)]
        else:
            scelti = ordine[:k]

        return [(self.meta[i], float(punteggi[i])) for i in scelti]

print("Carico il modello di embedding (la prima volta lo scarica, ~120 MB)...")
modello = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")



def firma_documenti(cartella):
    parti = []
    for p in sorted(glob.glob(os.path.join(cartella, "*.pdf")) +
                    glob.glob(os.path.join(cartella, "*.txt"))):
        st = os.stat(p)
        parti.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.md5("|".join(parti).encode()).hexdigest()

def _leggi_firma():
    f = os.path.join(CHROMA_PATH, ".firma")
    return open(f).read().strip() if os.path.exists(f) else None

def _scrivi_firma(firma):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(os.path.join(CHROMA_PATH, ".firma"), "w") as f:
        f.write(firma)



def genera(domanda, contesti):
    blocchi = "\n\n---\n\n".join(f"[Da: {m['fonte']}]\n{m['testo']}" for m, _ in contesti)
    prompt = f"""Sei un assistente personale che risponde su dieta, lista della spesa
e allenamento, usando ESCLUSIVAMENTE i documenti nel CONTESTO qui sotto. Se
l'informazione non c'è, dillo invece di inventare. Indica da quale file arriva
l'informazione. Rispondi in italiano, in modo conciso.

CONTESTO:
{blocchi}

DOMANDA: {domanda}"""

    from openai import OpenAI
    client = OpenAI()                                # legge OPENAI_API_KEY
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                         # economico e adatto qui
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

def crea_embedding(testi):
    # normalize_embeddings=True -> la similarità coseno diventa un prodotto scalare
    return modello.encode(testi, normalize_embeddings=True)


def chunk_documento_old(doc):
    nome = doc["fonte"].lower()
    max_car = 1500 if any(k in nome for k in KEEP_COMPACT) else 600
    pezzi = split_ricorsivo(doc["testo"], max_caratteri=max_car)
    return [{"testo": p, "fonte": doc["fonte"]} for p in pezzi]

def _norm_giorno(s):
    return s.lower().replace("ì", "i")

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


_HEADER = re.compile(
    r'(?=(?:[A-E]\s*[\x7f•·]\s*(?:Luned|Marted|Mercoled|Gioved|Venerd))'  # giorni dieta
    r'|(?:SESSIONE\s*\d)'                                                 # sedute allenamento
    r'|(?:Note generali)|(?:Indicazioni\b)|(?:Note e terminologia))'
)

_GIORNO = re.compile(r'(?i)(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|domenica)')
_PASTO  = re.compile(r'(?=(?:Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b)')

_MEAL = re.compile(r'^(Colazione|Pranzo|Spuntino serale|Spuntino|Cena)\b[^\d]*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$')

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

if __name__ == "__main__":
    documenti = carica_documenti(CARTELLA)
    if documenti:
        firma = firma_documenti(CARTELLA)
        indice = Indice()

        if not indice.vuoto() and _leggi_firma() == firma:
            print(f"Indice caricato da disco: {len(indice.meta)} chunk (nessuna modifica ai file).")
        else:
            print("File nuovi o modificati: (ri)costruisco l'indice...")
            indice.reset()
            for doc in carica_documenti(CARTELLA):
                chunk = chunk_documento(doc)
                indice.aggiungi(chunk)
                print(f"  - {doc['fonte']}: {len(chunk)} chunk")
            _scrivi_firma(firma)
            print(f"Totale: {len(indice.meta)} chunk.\n")
        esempi = [
            "Cosa devo mangiare a pranzo?",
            "Quali esercizi faccio nel giorno gambe?",
            "Gli ingredienti della cena sono nella lista della spesa?",
        ]
        print("Domande da provare:")
        for e in esempi:
            print("  ·", e)

        ha_chiave = bool(os.environ.get("OPENAI_API_KEY"))
        while True:
            domanda = input("\nDomanda (invio vuoto per uscire): ").strip()
            if not domanda:
                break
            contesti = indice.cerca(domanda, k=8)
            print("\nBlocchi recuperati:")
            for m, p in contesti:
                print(f"  [{p:.3f}] ({m['fonte']}) {m['testo'][:70].strip()}...")
            print("\n--- testo completo del blocco migliore ---")
            print(contesti[0][0]["testo"][:400])   # riga di debug, poi togli
            if ha_chiave:
                print("\nRisposta:\n" + genera(domanda, contesti))


