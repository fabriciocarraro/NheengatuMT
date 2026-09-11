# -*- coding: utf-8 -*-
"""Kit léxico v0 da Fase 2 (terminologia).

Gera, só com fontes locais:
  work/lemas_yrl.tsv          — tabela FORMA→LEMA extraída do treebank UD
                                ("Yauti-lite": lematização por tabela; upgrade
                                futuro = Yauti de verdade)
  work/lexico_pt_yrl_v0.jsonl — léxico PT→yrl colhido dos pares CURTOS do
                                TREINO (norm2; dev/test nunca entram) +
                                candidatos Gurgel curtos

Upgrades previstos (v1): verbetes do dicionário de Ávila (extração do PDF por
layout), glossário CompLin (não existe como arquivo no repo — confirmar com
Alencar), léxico da Incubadora Wikimedia.
"""
import json, re, sys, io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

PT_STOP = set("""a o e de da do em um uma para com que se por as os das dos nas
nos no na ao à às aos é são foi ser ter seu sua seus suas ele ela eles elas
não mais muito já também""".split())

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def strip_punct(s):
    return re.sub(r'^[\s\W]+|[\s\W]+$', "", s, flags=re.UNICODE)

# ---------- 1. FORMA -> LEMA do treebank (lema mais frequente vence) ----------
from collections import Counter
lemma_votes = defaultdict(Counter)
for f in ("yrl_complin-ud-train.conllu", "yrl_complin-ud-test.conllu"):
    p = ROOT / "work" / "ud_yrl" / f
    if not p.exists():
        continue
    for line in open(p, encoding="utf-8"):
        if not line or line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 4 or not cols[0].isdigit():
            continue
        form, lemma, upos = cols[1], cols[2], cols[3]
        if upos == "PUNCT" or lemma == "_" or not re.search(r"[^\W\d_]", form):
            continue
        lemma_votes[norm(form)][norm(lemma)] += 1
lemmas = {k: v.most_common(1)[0][0] for k, v in lemma_votes.items()}
out = ROOT / "work" / "lemas_yrl.tsv"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for k, v in sorted(lemmas.items()):
        fh.write(f"{k}\t{v}\n")
print(f"[lemas] {len(lemmas)} pares forma->lema -> {out}")

# ---------- 2. Léxico PT->yrl (colheita de pares curtos do TREINO) ----------
def harvest(rows, fonte, get_pt, get_yrl, grupo=None):
    got = []
    for r in rows:
        pt, yrl = strip_punct(get_pt(r)), strip_punct(get_yrl(r))
        if not pt or not yrl:
            continue
        if any(ch.isdigit() for ch in pt + yrl):
            continue
        npt, nyrl = len(pt.split()), len(yrl.split())
        if npt > 4 or nyrl > 4:
            continue
        if npt == 1 and (norm(pt) in PT_STOP or len(pt) < 4):
            continue
        # descarta interrogativas/imperativas de lição ("Responda.", "Quem é você?")
        if re.search(r"[?!]", get_pt(r)) or pt.lower().startswith(("responda", "traduza", "complete")):
            continue
        got.append({"pt": pt, "yrl": yrl, "fonte": fonte,
                    **({"grupo": grupo(r)} if grupo else {})})
    return got

DATA2 = ROOT / "cluster" / "data" / "norm2"
entries = []
train = [json.loads(l) for l in open(DATA2 / "train.por-yrl.jsonl", encoding="utf-8")]
entries += harvest(train, "treino_norm2", lambda r: r["src"], lambda r: r["tgt"], grupo=lambda r: r.get("grupo"))
gurgel = [json.loads(l) for l in open(ROOT / "work" / "candidatos_gurgel.jsonl", encoding="utf-8")]
gurgel_entries = harvest(gurgel, "gurgel_candidatos", lambda r: r["pt"], lambda r: r["yrl"])

# anti-vazamento: candidatos_gurgel é fonte PRÉ-SPLIT — descarta qualquer
# entrada que coincida com um par (ou fonte inteira) de dev/test
eval_src, eval_pair = set(), set()
for f in ("dev_const", "dev_extra", "test_const", "test_extra"):
    for r in map(json.loads, open(DATA2 / f"{f}.por-yrl.jsonl", encoding="utf-8")):
        eval_src.add(norm(r["src"]))
        eval_pair.add((norm(r["src"]), norm(r["tgt"])))
before = len(gurgel_entries)
gurgel_entries = [e for e in gurgel_entries
                  if norm(e["pt"]) not in eval_src
                  and (norm(e["pt"]), norm(e["yrl"])) not in eval_pair]
print(f"[anti-vazamento] gurgel_candidatos: {before} -> {len(gurgel_entries)} "
      f"({before - len(gurgel_entries)} removidas por coincidir com dev/test)")
entries += gurgel_entries

# lixo: multiword só-stopword e cópias PT==yrl
entries = [e for e in entries
           if not (len(e["pt"].split()) > 1 and
                   all(norm(w) in PT_STOP or len(w) < 3 for w in e["pt"].split()))
           and norm(e["pt"]) != norm(e["yrl"])]

# dedupe e agrupamento: mesma entrada PT -> variantes yrl (sinônimos/grafias)
by_pt = defaultdict(dict)
for e in entries:
    by_pt[norm(e["pt"])].setdefault(norm(e["yrl"]), e)
out = ROOT / "work" / "lexico_pt_yrl_v0.jsonl"
n_terms = n_var = 0
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for pt_key in sorted(by_pt):
        variants = list(by_pt[pt_key].values())
        rec = {"pt": variants[0]["pt"],
               "yrl": [v["yrl"] for v in variants],
               "fontes": sorted({v["fonte"] for v in variants} |
                                {v.get("grupo") for v in variants if v.get("grupo")})}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_terms += 1; n_var += len(variants)
print(f"[lexico] {n_terms} entradas PT ({n_var} variantes yrl) -> {out}")
print("[nota] fonte = SÓ treino + candidatos; dev/test jamais entram no léxico")
