# -*- coding: utf-8 -*-
"""Baixa o paralelo espanhol-guarani do AmericasNLP e converte para o formato do pipeline.
Rodar no nó de login (internet). Saída: data/gn/train.spa-grn.jsonl"""
import json, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "data", "gn"), exist_ok=True)
BASES = [
    "https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/guarani-spanish/",
    "https://raw.githubusercontent.com/AmericasNLP/americasnlp2023/master/ST1_MachineTranslation/data/guarani-spanish/",
]
def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace").splitlines()

es = gn = None
for base in BASES:
    for pref in ["train"]:
        try:
            es = get(base + pref + ".es")
            gn = get(base + pref + ".gn")
            print("fonte:", base)
            break
        except Exception as e:
            print("tentativa falhou:", base, type(e).__name__)
    if es: break
assert es and gn and len(es) == len(gn), "não foi possível baixar o paralelo es-gn"

out = os.path.join(ROOT, "data", "gn", "train.spa-grn.jsonl")
n = 0
with open(out, "w", encoding="utf-8") as f:
    for s, g in zip(es, gn):
        s, g = s.strip(), g.strip()
        if len(s) > 3 and len(g) > 3:
            f.write(json.dumps({"src_lang": "spa_Latn", "tgt_lang": "grn_Latn",
                                "src": s, "tgt": g}, ensure_ascii=False) + "\n")
            n += 1
print(f"ok: {n} pares es-gn -> {out}")
