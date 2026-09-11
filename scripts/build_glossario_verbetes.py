# -*- coding: utf-8 -*-
"""Converte os verbetes do Ávila em camada de glossário PT→yrl.

- Glosas viram chaves PT (divididas em ';' e ','), filtrando descrições
  enciclopédicas ("certo tipo de...", "nome comum a...") e glosas longas.
- Headwords com notação relacional "(t, r, s/x)" têm o parêntese removido.
- A rede de remissivas (var_de / ver) anexa variantes yrl à entrada canônica.
- Nível: "verbete_avila" (fonte autoritativa — dispensa auditoria por
  evidência; filtragem programática apenas).

Saída: work/glossario_verbetes.jsonl (mesmo esquema dos demais glossários).
"""
import json, re, sys, io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

DESCARTA_INICIO = ("certo tipo", "certa ", "nome comum", "nome genérico",
                   "nome de", "espécie de", "designação", "prefixo", "sufixo",
                   "partícula", "variante", "o mesmo que", "diz-se", "espécie")
PT_STOP = set("a o e de da do em um uma para com que se por é são".split())

def norm(s): return re.sub(r"\s+", " ", s.strip().lower())

def clean_head(h):
    h = re.sub(r"\s*\([^)]*\)\s*", "", h).strip()      # notação relacional
    h = re.sub(r"\d+$", "", h).strip()                  # homônimos numerados
    return h

rows = [json.loads(l) for l in open(ROOT / "work" / "verbetes_avila.jsonl", encoding="utf-8")]

# rede de variantes: variante -> canônico
canon = {}
for e in rows:
    alvo = e.get("var_de") or e.get("ver")
    if alvo:
        canon[clean_head(e["yrl"])] = clean_head(alvo)

variantes_de = defaultdict(set)
for var, can in canon.items():
    if var and can and var != can:
        variantes_de[can].add(var)

by_pt = defaultdict(set)
for e in rows:
    head = clean_head(e["yrl"])
    if not head or len(head) < 2:
        continue
    formas = {head} | variantes_de.get(head, set())
    for g in e["glosas"]:
        for termo in re.split(r"\s*[;,]\s*", g):
            t = termo.strip(" .()")
            nt = norm(t)
            if (not t or len(t) < 3 or len(t.split()) > 4
                    or any(nt.startswith(d) for d in DESCARTA_INICIO)
                    or nt in PT_STOP or len(t) > 45):
                continue
            for f in formas:
                by_pt[nt].add((t, f))

out = ROOT / "work" / "glossario_verbetes.jsonl"
n = 0
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for nt in sorted(by_pt):
        pares = sorted(by_pt[nt])
        pt_orig = pares[0][0]
        yrls = []
        for _, f in pares:
            if norm(f) not in {norm(x) for x in yrls}:
                yrls.append(f)
        fh.write(json.dumps({"pt": pt_orig, "yrl": yrls[:8],
                             "nivel": "verbete_avila"}, ensure_ascii=False) + "\n")
        n += 1
print(f"[glossario_verbetes] {n} entradas PT (de {len(rows)} verbetes; "
      f"{len(variantes_de)} redes de variantes) -> {out}")
