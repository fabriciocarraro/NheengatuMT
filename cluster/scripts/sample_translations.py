# -*- coding: utf-8 -*-
"""Gera amostras de tradução para inspeção humana (o que a métrica não mostra).
Uso: python scripts/sample_translations.py --model runs/TAG/best --data_dir data/norm \
        --set dev_const.por-yrl.jsonl --n 25 --out runs/TAG/amostras.txt"""
import argparse, json, os, random, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--data_dir", required=True)
p.add_argument("--set", default="dev_const.por-yrl.jsonl"); p.add_argument("--n", type=int, default=25)
p.add_argument("--out", default=None); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rows = [json.loads(l) for l in open(os.path.join(a.data_dir, a.set), encoding="utf-8")]
random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()

def tr(texts, sl, tl):
    tok.src_lang = sl; out = []
    bos = tok.convert_tokens_to_ids(tl)
    for i in range(0, len(texts), 16):
        b = tok(texts[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            g = model.generate(**b, forced_bos_token_id=bos, num_beams=4, max_length=192, repetition_penalty=1.5)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out

hyp = tr([r["src"] for r in rows], "por_Latn", "yrl_Latn")
back = tr([r["tgt"] for r in rows], "yrl_Latn", "por_Latn")
lines = []
for r, h, b in zip(rows, hyp, back):
    lines += [f"[{r.get('id','')} | {r.get('grupo','')}]",
              f"  PT (fonte)     : {r['src']}",
              f"  YRL (referência): {r['tgt']}",
              f"  YRL (modelo)   : {h}",
              f"  PT (volta do modelo, a partir da referência): {b}", ""]
txt = "\n".join(lines)
print(txt)
if a.out:
    open(a.out, "w", encoding="utf-8").write(txt)
    print("salvo em", a.out)
