# -*- coding: utf-8 -*-
"""Checkpoint averaging: média aritmética dos pesos de 2+ checkpoints do
mesmo treino (clássico +0,2-0,6 chrF). Salva um novo diretório de modelo
compatível com evaluate_yrl/evaluate_mbr.

Uso:
  python avg_checkpoints.py --ckpts runs/TAG/checkpoint-19000 \
      runs/TAG/checkpoint-20000 runs/TAG/best --out runs/TAG/avg
(o tokenizer/config vêm do primeiro; pesos = média float32 → bf16)
"""
import argparse, sys, io, shutil, os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert len(a.ckpts) >= 2, "precisa de 2+ checkpoints"

    print(f"[avg] carregando {a.ckpts[0]}")
    modelo = AutoModelForSeq2SeqLM.from_pretrained(a.ckpts[0], torch_dtype=torch.float32)
    soma = {k: v.clone() for k, v in modelo.state_dict().items()}
    for c in a.ckpts[1:]:
        print(f"[avg] somando {c}")
        m2 = AutoModelForSeq2SeqLM.from_pretrained(c, torch_dtype=torch.float32)
        sd2 = m2.state_dict()
        assert sd2.keys() == soma.keys(), f"chaves diferentes em {c}"
        for k in soma:
            if soma[k].dtype.is_floating_point:
                soma[k] += sd2[k]
        del m2, sd2
    n = len(a.ckpts)
    for k in soma:
        if soma[k].dtype.is_floating_point:
            soma[k] /= n
    modelo.load_state_dict(soma)
    os.makedirs(a.out, exist_ok=True)
    modelo.save_pretrained(a.out, safe_serialization=True)
    try:
        AutoTokenizer.from_pretrained(a.ckpts[0]).save_pretrained(a.out)
    except Exception:
        # checkpoints intermediários às vezes não têm tokenizer: usa o do best
        alt = os.path.join(os.path.dirname(a.ckpts[0].rstrip("/\\")), "best")
        AutoTokenizer.from_pretrained(alt).save_pretrained(a.out)
        print(f"[avg] tokenizer copiado de {alt}")
    print(f"[avg] média de {n} checkpoints salva em {a.out}")

if __name__ == "__main__":
    main()
