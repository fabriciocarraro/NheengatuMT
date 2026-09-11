# -*- coding: utf-8 -*-
"""Token pairs (Bilex Rx): o glossário auditado vira micro-pares de treino.

Cada entrada aprovado/corrigido do glossario_fase2 gera 1 par por variante yrl
(variantes múltiplas = mesma fonte com traduções alternativas — classe que já
existe no corpus). Saída: cluster/data/norm2/train.lex-yrl.jsonl, para uso via
--extra_train (o build_bi bidirecionaliza — desejado no Bilex Rx).
"""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
out_path = ROOT / "cluster" / "data" / "norm2" / "train.lex-yrl.jsonl"

n = 0
with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
    for line in open(ROOT / "work" / "glossario_fase2.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("nivel") not in ("aprovado", "corrigido"):
            continue
        for v in e["yrl"]:
            fh.write(json.dumps({"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                                 "src": e["pt"], "tgt": v,
                                 "grupo": "lexico_fase2", "id": f"lex{n:05d}"},
                                ensure_ascii=False) + "\n")
            n += 1
print(f"[token pairs] {n} micro-pares -> {out_path}")
