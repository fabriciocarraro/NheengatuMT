# -*- coding: utf-8 -*-
"""A4: gera pares sintéticos PT→yrl com o termo-alvo GARANTIDO por
decodificação restrita (constrained beam largo — receita validada no C1:
feixe 16, custo ~-0,5 chrF, cumprimento 100%).

Entrada: JSONL {src, alvos:[...], termo_pt}
Saída:   JSONL no esquema de treino {src_lang, tgt_lang, src, tgt, grupo, id}
         (uso via --extra_train, como o léxico)

Uso:
  python gen_forced.py --model runs/TAG/best --in data/a4_sintetico_pt.jsonl \
      --out data/norm3/train.orf-yrl.jsonl [--num_beams 16]
"""
import argparse, json, sys, io
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--infile", "--in", dest="infile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_beams", type=int, default=16)
    p.add_argument("--max_len", type=int, default=192)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).cuda().eval()
    tok.src_lang = "por_Latn"
    bos = tok.convert_tokens_to_ids("yrl_Latn")

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    n_ok = n_falha = n_cumprido = n_alvos = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            alvos = list(dict.fromkeys(t.strip() for t in r.get("alvos", []) if t.strip()))[:2]
            force = []
            for alvo in alvos:
                ids = tok(alvo, add_special_tokens=False)["input_ids"]
                if ids and tok.unk_token_id not in ids:
                    force.append(ids)
            enc = tok(r["src"], return_tensors="pt", truncation=True,
                      max_length=a.max_len).to("cuda")
            kw = dict(forced_bos_token_id=bos, num_beams=a.num_beams,
                      max_length=a.max_len, repetition_penalty=1.5)
            try:
                with torch.no_grad():
                    gen = (model.generate(**enc, force_words_ids=force, **kw)
                           if force else model.generate(**enc, **kw))
            except Exception:
                with torch.no_grad():
                    gen = model.generate(**enc, **kw)
                n_falha += 1
            tgt = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
            hits = sum(1 for alvo in alvos if alvo.lower() in tgt.lower())
            n_alvos += len(alvos)
            n_cumprido += hits
            if tgt:
                fh.write(json.dumps({
                    "src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
                    "src": r["src"], "tgt": tgt,
                    "grupo": "sintetico_orfaos", "id": f"orf{i:04d}",
                    "alvos_ok": bool(alvos) and hits == len(alvos)},
                    ensure_ascii=False) + "\n")
                n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rows)} ({n_cumprido}/{n_alvos} alvos ok)")
    print(f"[gen_forced] {n_ok} pares -> {a.out} | falhas de restrição {n_falha} | "
          f"alvos cumpridos {n_cumprido}/{n_alvos}")

if __name__ == "__main__":
    main()
