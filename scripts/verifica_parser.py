"""Confronta i chunk dei documenti personali fra i due parser PDF.

Gli splitter di dieta, spesa e allenamento dipendono da come il parser
decodifica il testo: il bullet della lista della spesa e' un glifo di un
font simbolico, che pypdf legge "■" e PyMuPDF "I". Un cambio di parser
puo' quindi far smettere di funzionare uno splitter SENZA errori — la
lista era passata da 29 chunk a 1 in silenzio.

Questo script rende visibile quel tipo di regressione:

    .venv/bin/python scripts/verifica_parser.py

Atteso: tutte le righe OK. Una riga DIVERGE va guardata — non e'
automaticamente un problema (differenze di poche unita' su chunk di
intestazione sono normali), ma un cambio di tipo o di numero di giorni
riconosciuti lo e'.
"""

import sys, glob, os
sys.path.insert(0, 'src')
from rag.parsing import _parse_pymupdf, _parse_pypdf
from rag.chunking import chunk_documento

CHIAVI = ('spesa', 'piano', 'allen', 'reverse', 'alimentare')

def riassunto(f, testo):
    rec = chunk_documento({'fonte': f, 'testo': testo})
    tipi = {}
    for r in rec:
        tipi[r.get('tipo') or '-'] = tipi.get(r.get('tipo') or '-', 0) + 1
    return len(rec), tipi, len({r['giorno'] for r in rec if r.get('giorno')})

print(f"{'documento':40s} {'pypdf':>22s}   {'pymupdf':>22s}")
for p in sorted(glob.glob('documenti/*.pdf')):
    f = os.path.basename(p)
    if not any(k in f.lower() for k in CHIAVI):
        continue
    na, ta, ga = riassunto(f, _parse_pypdf(p).testo)
    nb, tb, gb = riassunto(f, _parse_pymupdf(p).testo)
    esito = 'OK ' if (ta == tb and ga == gb) else 'DIVERGE'
    print(f'{esito} {f[:36]:38s} chunk={na:4d} {str(ta)[:20]:22s} chunk={nb:4d} {str(tb)[:20]:22s}')
