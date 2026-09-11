# -*- coding: utf-8 -*-
"""Variantes de token pairs para o pack de fechamento da Fase 2.

- train.lexesp-yrl.jsonl = lex (1.027) + PARES ESPACIAIS minerados dos
  verbetes do Ávila (classe de erro `wirupi`→"em cima" da análise do Exp. 3:
  polaridade espacial trocada; conserto barato via micro-pares).
- train.lex2-yrl.jsonl = lex + glossário de verbetes (~5k): testa se
  vocabulário geral em massa ajuda ou dilui (hipótese aberta do Exp. 2).

Saída em cluster/data/norm3/ (mesmo esquema do train.lex-yrl.jsonl).
"""
import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
N3 = ROOT / "cluster" / "data" / "norm3"

ESPACIAL = re.compile(
    r"\b(?:embaixo|debaixo|em cima|acima|abaixo|por cima|dentro|fora|"
    r"atrás|detrás|frente|diante|perto|próximo|junto|longe|meio|entre|"
    r"lado|direita|esquerda|margem|beira|alto|fundo|cabeceira|"
    r"rio acima|rio abaixo|para cima|para baixo)\b", re.IGNORECASE)

def clean_head(h):
    h = re.sub(r"\s*\([^)]*\)\s*", "", h).strip()
    return re.sub(r"\d+$", "", h).strip()

lex = [json.loads(l) for l in open(N3 / "train.lex-yrl.jsonl", encoding="utf-8")]
vistos = {(r["src"].lower(), r["tgt"].lower()) for r in lex}

# ---------------- pares espaciais ----------------
# lições da auditoria: fragmentos de glosa enciclopédica ("entre as quais",
# "do lado externo" em descrição de espécie) viram pares FALSOS se o filtro
# olhar só o fragmento — a glosa INTEIRA precisa ser limpa; e a forma do
# termo precisa ser válida (parênteses balanceados, sem palavra duplicada).
ENCICLO = re.compile(r"certo|espécie|tipo de|nome |ave |árvore|planta|peixe|"
                     r"rato|inseto|fruto|utilidade|costuma", re.I)
RUIM = re.compile(r"\betc\b|^entre (?:os|as) qua|^entre outr|^-|\berro\b|[A-ZÀ-Ú]{4,}")

def forma_ok(t):
    if t.count("(") != t.count(")"):
        return False
    toks = t.lower().split()
    if any(a == b for a, b in zip(toks, toks[1:])):   # "esquerda esquerda"
        return False
    return bool(re.fullmatch(r"[a-zà-ÿA-Z][a-zà-ÿA-Z()' -]+", t))

# falsos/malformados confirmados na inspeção manual dos 70 candidatos
BLOCK = {("do lado externo", "kurumĩ"), ("com a beira recortada", "xirí-kaá"),
         ("meio", "xinga"), ("tornar alto erguer", "muiwaté")}
MANUAIS = [("erguer", "muiwaté"), ("tornar alto", "muiwaté")]

esp, n = [], 0
for e in map(json.loads, open(ROOT / "work" / "verbetes_avila.jsonl", encoding="utf-8")):
    head = clean_head(e["yrl"])
    if not head or len(head) < 2 or len(head.split()) > 2:
        continue
    for g in e.get("glosas", []):
        if ENCICLO.search(g):                 # glosa INTEIRA enciclopédica: fora
            continue
        for termo in re.split(r"\s*[;,]\s*", g):
            t = termo.strip(" .()")
            if not (2 < len(t) <= 40 and len(t.split()) <= 4):
                continue
            if not ESPACIAL.search(t) or RUIM.search(t) or not forma_ok(t):
                continue
            if t.lower() == "meio" and "pouco" in g.lower():
                continue                      # sentido adverbial ("meio, um pouco")
            k = (t.lower(), head.lower())
            if k in vistos or k in BLOCK:
                continue
            vistos.add(k)
            esp.append({"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                        "src": t, "tgt": head,
                        "grupo": "lexico_espacial", "id": f"esp{n:04d}"})
            n += 1
for src_m, tgt_m in MANUAIS:
    if (src_m, tgt_m) not in vistos:
        vistos.add((src_m, tgt_m))
        esp.append({"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                    "src": src_m, "tgt": tgt_m,
                    "grupo": "lexico_espacial", "id": f"esp{n:04d}"})
        n += 1

# ---------------- guard anti-vazamento (lado inteiro vs conjuntos de eval) ----------------
# ANTES de escrever qualquer arquivo: um par de 1-2 palavras pode coincidir
# com uma fala curta inteira dos conjuntos de avaliação
eval_src, eval_tgt = set(), set()
for f in ("dev_const", "dev_extra", "test_const", "test_extra", "dev_fala", "test_fala"):
    for r in map(json.loads, open(N3 / f"{f}.por-yrl.jsonl", encoding="utf-8")):
        eval_src.add(re.sub(r"\s+", " ", r["src"].strip().lower()))
        eval_tgt.add(re.sub(r"\s+", " ", r["tgt"].strip().lower()))

def colide_eval(src, tgt):
    return src.strip().lower() in eval_src or tgt.strip().lower() in eval_tgt

antes = len(esp)
esp = [r for r in esp if not colide_eval(r["src"], r["tgt"])]
if antes != len(esp):
    print(f"[guard] espaciais removidos por colisão com eval: {antes - len(esp)}")

with open(N3 / "train.lexesp-yrl.jsonl", "w", encoding="utf-8", newline="\n") as fh:
    for r in lex + esp:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# sanidade: a classe do Exp. 3 está coberta? (deacc — Ávila usa wírupi/uakí)
import unicodedata
def _d(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if not unicodedata.combining(c))
chaves = [r for r in esp if re.search(r"wirup|iwat|uak|piterup|suaxar", _d(r["tgt"]))]
print(f"[lexesp] {len(lex)} lex + {len(esp)} espaciais = {len(lex)+len(esp)}")
print(f"  cobertura da classe wirupi/iwate/ruaki/piterupi/suaxara: {len(chaves)} pares")
for r in chaves[:8]:
    print(f"    {r['src']} -> {r['tgt']}")

# ---------------- lex2 (~5k) ----------------
FRAGMENTO = re.compile(r"^(?:cuja?s?\b|que\b|como\b|usad[oa]|serve\b|o que faz)")
vistos2 = {(r["src"].lower(), r["tgt"].lower()) for r in lex}
verb, n2, n_forma, n_vaz = [], 0, 0, 0
for e in map(json.loads, open(ROOT / "work" / "glossario_verbetes.jsonl", encoding="utf-8")):
    pt = e["pt"].strip()
    if not (2 < len(pt) <= 45 and len(pt.split()) <= 3):
        continue
    if RUIM.search(pt) or not forma_ok(pt) or FRAGMENTO.search(pt.lower()):
        n_forma += 1
        continue
    for y in e["yrl"][:1]:
        y = y.strip()
        if not y or len(y.split()) > 3:
            continue
        if colide_eval(pt, y):
            n_vaz += 1
            continue
        k = (pt.lower(), y.lower())
        if k in vistos2:
            continue
        vistos2.add(k)
        verb.append({"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                     "src": pt, "tgt": y,
                     "grupo": "lexico_verbetes", "id": f"vrb{n2:05d}"})
        n2 += 1
print(f"[lex2] descartes: forma/fragmento {n_forma} | colisão c/ eval {n_vaz}")

with open(N3 / "train.lex2-yrl.jsonl", "w", encoding="utf-8", newline="\n") as fh:
    for r in lex + verb:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[lex2] {len(lex)} lex + {len(verb)} verbetes = {len(lex)+len(verb)}")
