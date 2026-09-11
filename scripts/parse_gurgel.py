# -*- coding: utf-8 -*-
"""Extrai pares yrl-PT do corpus Gurgel (github.com/juliana-gurgel/yrl).
Formatos: intercalado (yrl / PT / linha em branco) e textos paralelos linha a linha."""
import sys, io, os, json, re, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")
G = os.path.join(BASE, "gurgel")
out = []

def add(yrl, pt, fonte):
    yrl, pt = yrl.strip(), pt.strip()
    if len(yrl) < 2 or len(pt) < 2: return
    if yrl.lower() == pt.lower(): return
    out.append({"yrl": yrl, "pt": pt, "fonte": fonte, "grupo": "gurgel"})

def parse_interleaved(path, fonte):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="replace")]
    blocks, cur = [], []
    for l in lines:
        if l.strip(): cur.append(l.strip())
        elif cur: blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    n = 0
    for b in blocks:
        if len(b) == 2:
            add(b[0], b[1], fonte); n += 1
        elif len(b) > 2 and len(b) % 2 == 0:
            for i in range(0, len(b), 2):
                add(b[i], b[i+1], fonte); n += 1
    return n

def parse_linealigned(fy, fp, fonte):
    ly = [l.strip() for l in open(fy, encoding="utf-8", errors="replace")]
    lp = [l.strip() for l in open(fp, encoding="utf-8", errors="replace")]
    n = 0
    for a, b in zip(ly, lp):
        if a and b:
            add(a, b, fonte); n += 1
    return n

# lições: p-* e e-* intercalados; t-yrl ↔ t-por alinhados
for d in sorted(glob.glob(os.path.join(G, "pibic-2020-2021/corpus/licoes/licao-*"))):
    lic = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "p-*.txt")) + glob.glob(os.path.join(d, "e-*.txt")):
        n = parse_interleaved(f, f"licoes/{lic}/{os.path.basename(f)}")
    ty = glob.glob(os.path.join(d, "t-yrl-*.txt"))
    tp = glob.glob(os.path.join(d, "t-por-*.txt"))
    if ty and tp:
        parse_linealigned(ty[0], tp[0], f"licoes/{lic}/texto")

# casasnovas gramática (intercalado)
cg = os.path.join(G, "nheenga-tagger/corpora/casasnovas-2006/gramatica/gram-casasnovas.txt")
if os.path.exists(cg):
    parse_interleaved(cg, "casasnovas-2006/gramatica")

# quaisquer outros txt bilíngues nos corpora
for f in glob.glob(os.path.join(G, "nheenga-tagger/corpora/**/*.txt"), recursive=True):
    if "gram-casasnovas" in f or os.path.getsize(f) < 200: continue
    parse_interleaved(f, os.path.relpath(f, G).replace("\\", "/"))

# dedupe interno exato
seen, uniq = set(), []
for r in out:
    k = (r["yrl"].lower(), r["pt"].lower())
    if k in seen: continue
    seen.add(k); uniq.append(r)

dest = os.path.join(BASE, "candidatos_gurgel.jsonl")
with open(dest, "w", encoding="utf-8") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(f"pares extraídos: {len(out)} | únicos: {len(uniq)} → {dest}")
print("por fonte (top):", Counter(r["fonte"].split("/")[0] for r in uniq).most_common())

