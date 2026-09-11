# -*- coding: utf-8 -*-
"""Avaliação final: 2 direções x conjuntos, chrF++ e spBLEU.
Uso: python evaluate_yrl.py --model best/ --data_dir $DATA/raw \
        --sets test_const.por-yrl.jsonl test_extra.por-yrl.jsonl --out results.json"""
import argparse, json, os
import torch
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--no_repeat", type=int, default=0,
                   help="no_repeat_ngram_size na geração (0 = desligado, comportamento original)")
    p.add_argument("--save_hyp", default=None,
                   help="diretório para salvar hipóteses em JSONL (p/ term_success.py)")
    p.add_argument("--annotate", default=None,
                   help="glossario.jsonl — anota termos no lado PT (por→yrl) com rate=1.0, "
                        "espelhando o regime de treino anotado")
    p.add_argument("--annotate_max_terms", type=int, default=2,
                   help="máx. de termos anotados por frase. O default 2 é o regime "
                        "de treino e reproduz as rodadas publicadas — não mudar para "
                        "comparar com elas. Existe porque com glossário estendido por "
                        "dicionário as 2 vagas são o gargalo: medido no dev_const, os "
                        "termos-alvo só recebiam dica em 8/19 oportunidades com 2, e em "
                        "17/19 com 4 (sem perder nenhuma anotação auditada).")
    p.add_argument("--tag", default=None,
                   help="prefixo de registro na FONTE (ex. '[fala]'), ambas as "
                        "direções — espelha o treino com --reg_tags")
    a = p.parse_args()

    # Caminho local inexistente vira tentativa de baixar do Hugging Face Hub e
    # explode 30 linhas abaixo com um HFValidationError sobre `repo_id`, que não
    # diz nada sobre a causa. Aconteceu em 06/08/2026 com um $T obsoleto no
    # ambiente do job, apontando para um run que não existia mais.
    for caminho, rot in ((a.model, "--model"), (a.data_dir, "--data_dir")):
        if not os.path.isdir(caminho):
            raise SystemExit(f"[erro] {rot} {caminho!r} não existe (ou não é diretório).")
    faltam = [s for s in a.sets if not os.path.isfile(os.path.join(a.data_dir, s))]
    if faltam:
        raise SystemExit(f"[erro] --sets não encontrados em {a.data_dir}: {faltam}")

    annot = None
    if a.annotate:
        from term_annotate import Annotator
        annot = Annotator(a.annotate, rate=1.0, seed=0,
                          max_terms=a.annotate_max_terms)
        print(f"[annot] inferência com terminologia inline: {len(annot.terms)} termos")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()

    def translate(texts, src_lang, tgt_lang):
        tok.src_lang = src_lang
        out = []
        bos = tok.convert_tokens_to_ids(tgt_lang)
        for i in range(0, len(texts), a.bsz):
            batch = tok(texts[i:i+a.bsz], return_tensors="pt", padding=True,
                        truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                kw = dict(forced_bos_token_id=bos, num_beams=4,
                          max_length=a.max_len, repetition_penalty=1.5)
                if a.no_repeat:
                    kw["no_repeat_ngram_size"] = a.no_repeat
                gen = model.generate(**batch, **kw)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    results = {}
    for s in a.sets:
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, s), encoding="utf-8")]
        src = [r["src"] for r in rows]; tgt = [r["tgt"] for r in rows]
        sl, tl = rows[0]["src_lang"], rows[0]["tgt_lang"]
        for direction, (X, Y, s_l, t_l) in {
                f"{sl}->{tl}": (src, tgt, sl, tl),
                f"{tl}->{sl}": (tgt, src, tl, sl)}.items():
            X_orig = X
            if annot is not None and s_l == "por_Latn":
                X = [annot(x) for x in X]
            if a.tag:
                X = [f"{a.tag} {x}" for x in X]
            hyp = translate(X, s_l, t_l)
            if a.save_hyp:
                os.makedirs(a.save_hyp, exist_ok=True)
                tag = f"{s.replace('.jsonl','')}_{s_l[:3]}2{t_l[:3]}"
                with open(os.path.join(a.save_hyp, f"hyp_{tag}.jsonl"), "w",
                          encoding="utf-8") as fh:
                    for r_, x_, y_, h_ in zip(rows, X_orig, Y, hyp):
                        fh.write(json.dumps({"id": r_.get("id"), "src": x_,
                                             "ref": y_, "hyp": h_},
                                            ensure_ascii=False) + "\n")
            chrf = sacrebleu.corpus_chrf(hyp, [Y], word_order=2).score
            try:
                bleu = sacrebleu.corpus_bleu(hyp, [Y], tokenize="flores200").score
            except Exception:
                bleu = sacrebleu.corpus_bleu(hyp, [Y]).score
            results[f"{s}|{direction}"] = {"chrF++": round(chrf, 2), "spBLEU": round(bleu, 2), "n": len(X)}
            print(f"{s} {direction}: chrF++ {chrf:.2f} | spBLEU {bleu:.2f}")
    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    main()
