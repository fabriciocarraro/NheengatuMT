# -*- coding: utf-8 -*-
"""APOSENTADO — ESTE SCRIPT NUNCA RODOU. NÃO produziu os dados do repositório.

Os `train.porMT-yrl.jsonl` que existem em data/*/ foram traduzidos por um LLM
(claude-fable-5), em 03/08/2026, e carregam essa proveniência no próprio campo
`mt_engine` de cada linha — confira antes de supor qualquer coisa a partir deste
arquivo. A decisão de trocar o NLLB pelo LLM está registrada na conversa daquele
dia: FR->PT é alto-recurso para alto-recurso, onde o LLM ganha do NLLB-1.3B com
folga, e o alvo yrl permanece humano de todo modo.

Fica aqui só como caminho alternativo documentado (útil se um dia for preciso
refazer a fonte sintética sem depender de LLM). Se for rodar, saiba que vai
SOBRESCREVER os dados de proveniência LLM.

--- documentação original abaixo ---

Converte os pares FR<->yrl (Pequeno Príncipe) em pares PT_sintético<->yrl,
traduzindo o lado francês com o NLLB-1.3B local (fra_Latn -> por_Latn).

O yrl (alvo) permanece humano; só a FONTE é sintética — configuração segura
(idêntica à back-translation). Rodar 1x em GPU (~minutos):

  python scripts/translate_fr_to_pt.py
Gera data/raw/train.porMT-yrl.jsonl e data/norm/train.porMT-yrl.jsonl
(idênticos; o norm não altera fontes fora da Constituição).
Usar no treino via --extra_train $DATA/$var/train.porMT-yrl.jsonl
"""
import json, os, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "nllb-200-distilled-1.3B")
SRC = os.path.join(ROOT, "data", "raw", "train.fra-yrl.jsonl")

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
tok.src_lang = "fra_Latn"
bos = tok.convert_tokens_to_ids("por_Latn")

pt = []
B = 32
for i in range(0, len(rows), B):
    batch = tok([r["src"] for r in rows[i:i+B]], return_tensors="pt", padding=True,
                truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        gen = model.generate(**batch, forced_bos_token_id=bos, num_beams=5, max_length=256)
    pt += tok.batch_decode(gen, skip_special_tokens=True)
    print(f"{min(i+B, len(rows))}/{len(rows)}", end="\r")

for var in ("raw", "norm", "uni", "raw2", "norm2", "uni2"):
    if not os.path.isdir(os.path.join(ROOT, "data", var)):
        continue
    out = os.path.join(ROOT, "data", var, "train.porMT-yrl.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r, p in zip(rows, pt):
            f.write(json.dumps({"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                                "src": p.strip(), "tgt": r["tgt"],
                                "grupo": "trevisan_ptMT", "id": r.get("id", "")},
                               ensure_ascii=False) + "\n")
    print(f"\nok: {out} ({len(rows)} pares, fonte PT sintética)")
