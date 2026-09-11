# -*- coding: utf-8 -*-
"""Avaliação com DECODIFICAÇÃO RESTRITA (constrained beam) no modo com-dica:
os alvos yrl das dicas viram restrições duras (force_words_ids) — o termo
TEM que aparecer na saída. Objetivo: TSR com dica ~90+ por construção,
medindo o custo em chrF.

Só direção por→yrl (dicas não existem na outra). bsz=1 (restrições são
por frase). Frases sem dica disparada: beam normal.

Uso:
  python evaluate_forced.py --model runs/TAG/best --data_dir data/norm3 \
      --sets dev_const.por-yrl.jsonl --annotate data/glossario_fase2.jsonl \
      --out forced.json --save_hyp DIR
"""
import argparse, json, os, re, sys, io
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ALVO = re.compile(r"\| ([^|]+) \|")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--annotate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_terms", type=int, default=2,
                   help="máx. de dicas/restrições por frase (resíduo do TSR "
                        "é cobertura — subir isto ataca direto)")
    p.add_argument("--save_hyp", default=None)
    a = p.parse_args()

    from term_annotate import Annotator
    annot = Annotator(a.annotate, rate=1.0, seed=0, max_terms=a.max_terms)
    print(f"[annot] {len(annot.terms)} termos; restrições duras; "
          f"max_terms={a.max_terms}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).cuda().eval()

    results = {}
    for s in a.sets:
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, s), encoding="utf-8")]
        sl, tl = rows[0]["src_lang"], rows[0]["tgt_lang"]
        assert sl == "por_Latn", "restrições só na direção por→yrl"
        tok.src_lang = sl
        bos = tok.convert_tokens_to_ids(tl)
        hyp, n_com, n_sem, n_falha = [], 0, 0, 0
        n_cumprida = n_alvos = 0
        for r in rows:
            x_ann = annot(r["src"])
            # dedupe preservando ordem (termo repetido = restrição duplicada quebra o HF)
            alvos = list(dict.fromkeys(
                t.strip() for t in ALVO.findall(x_ann) if t.strip()))[:a.max_terms]
            enc = tok(x_ann, return_tensors="pt", truncation=True,
                      max_length=a.max_len).to("cuda")
            kw = dict(forced_bos_token_id=bos, num_beams=a.num_beams,
                      max_length=a.max_len, repetition_penalty=1.5)
            force = []
            for alvo in alvos:
                ids = tok(alvo, add_special_tokens=False)["input_ids"]
                # alvo com <unk> forçaria lixo na saída: pula a restrição
                if ids and tok.unk_token_id not in ids:
                    force.append(ids)
            try:
                with torch.no_grad():
                    if force:
                        gen = model.generate(**enc, force_words_ids=force, **kw)
                        n_com += 1
                    else:
                        gen = model.generate(**enc, **kw)
                        n_sem += 1
            except Exception:
                # restrição patológica: cai para beam normal, conta a falha
                with torch.no_grad():
                    gen = model.generate(**enc, **kw)
                n_falha += 1
            saida = tok.batch_decode(gen, skip_special_tokens=True)[0]
            # diagnóstico: a restrição foi mesmo cumprida na superfície?
            for alvo in alvos:
                n_alvos += 1
                if alvo.lower() in saida.lower():
                    n_cumprida += 1
            hyp.append(saida)
        Y = [r["tgt"] for r in rows]
        if a.save_hyp:
            os.makedirs(a.save_hyp, exist_ok=True)
            nome = f"{s.replace('.jsonl','')}_{sl[:3]}2{tl[:3]}"
            with open(os.path.join(a.save_hyp, f"hyp_{nome}.jsonl"), "w",
                      encoding="utf-8") as fh:
                for r_, h_ in zip(rows, hyp):
                    fh.write(json.dumps({"id": r_.get("id"), "src": r_["src"],
                                         "ref": r_["tgt"], "hyp": h_},
                                        ensure_ascii=False) + "\n")
        chrf = sacrebleu.corpus_chrf(hyp, [Y], word_order=2).score
        try:
            bleu = sacrebleu.corpus_bleu(hyp, [Y], tokenize="flores200").score
        except Exception:
            bleu = sacrebleu.corpus_bleu(hyp, [Y]).score
        results[f"{s}|{sl}->{tl}"] = {"chrF++": round(chrf, 2),
                                      "spBLEU": round(bleu, 2), "n": len(rows),
                                      "com_restricao": n_com, "sem_dica": n_sem,
                                      "falhas_restricao": n_falha,
                                      "alvos_cumpridos": f"{n_cumprida}/{n_alvos}"}
        print(f"{s}: chrF++ {chrf:.2f} | spBLEU {bleu:.2f} | "
              f"restritas {n_com} | sem dica {n_sem} | falhas {n_falha} | "
              f"alvos cumpridos na superfície: {n_cumprida}/{n_alvos}")
    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    main()
