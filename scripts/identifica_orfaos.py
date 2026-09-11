# -*- coding: utf-8 -*-
"""A4 (destilação de órfãos), passo 1: identifica os termos do glossário
com ≤2 ocorrências no lado PT do treino norm3 — a classe "terminologia
órfã" diagnosticada na Fase 1 (~1 exemplo/termo = modelo não aprende).

Saída: work/orfaos_a4.jsonl {pt, yrl (variantes), freq_train}
"""
import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

rows = [json.loads(l) for l in open(ROOT / "cluster" / "data" / "norm3" / "train.por-yrl.jsonl", encoding="utf-8")]
srcs = [r["src"].lower() for r in rows if r["src_lang"] == "por_Latn"]
blob = "\n".join(srcs)

orfaos, freqs = [], []
for line in open(ROOT / "work" / "glossario_fase2.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("nivel") not in ("aprovado", "corrigido"):
        continue
    pt = e["pt"].strip()
    if not pt:
        continue
    pat = re.compile(r"(?<![\w])" + re.escape(pt.lower()) + r"(?![\w])")
    f = len(pat.findall(blob))
    freqs.append(f)
    if f <= 2:
        yrl = e["yrl"] if isinstance(e["yrl"], list) else [e["yrl"]]
        orfaos.append({"pt": pt, "yrl": [y.strip() for y in yrl if y.strip()][:3],
                       "freq_train": f})

orfaos.sort(key=lambda x: (x["freq_train"], x["pt"]))
out = ROOT / "work" / "orfaos_a4.jsonl"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for o in orfaos:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")

import statistics as st
print(f"glossário avaliado: {len(freqs)} termos | mediana de freq no treino: {st.median(freqs)}")
print(f"ÓRFÃOS (freq ≤2): {len(orfaos)}  -> {out}")
print(f"  freq 0: {sum(1 for o in orfaos if o['freq_train']==0)}")
print(f"  freq 1: {sum(1 for o in orfaos if o['freq_train']==1)}")
print(f"  freq 2: {sum(1 for o in orfaos if o['freq_train']==2)}")
print("amostra:", [o["pt"] for o in orfaos[:12]])
