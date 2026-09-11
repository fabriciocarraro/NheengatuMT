# -*- coding: utf-8 -*-
"""Avaliação com decodificação MBR (Minimum Bayes Risk) por chrF++.

Para cada fonte: gera N candidatos por amostragem (+ opcionalmente a
hipótese beam-4 no pool), e escolhe o candidato de maior utilidade média
chrF++ contra os demais. Sem métrica neural — exatamente o decode de
utilidade que uma língua fora do COMET permite.

Pool multi-modelo: --models aceita vários checkpoints; cada um contribui
n/k candidatos (modelos carregados um por vez para caber na GPU).

Uso (exemplos):
  python evaluate_mbr.py --models runs/TAG/best --data_dir data/norm3 \
      --sets dev_extra.por-yrl.jsonl --nsamples 32 --temp 0.7 \
      --include_beam --out mbr_results.json [--save_hyp DIR] [--annotate G]
Direções: por padrão só por→yrl (--dirs fwd); --dirs both p/ as duas.
"""
import argparse, json, os, sys, io
import torch
import sacrebleu
from sacrebleu.metrics import CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
CHRF_SENT = CHRF(word_order=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="1+ checkpoints; pool de candidatos é a união")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nsamples", type=int, default=32,
                   help="candidatos por fonte (divididos entre os modelos)")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help=">0 usa epsilon-sampling em vez de top-p")
    p.add_argument("--include_beam", action="store_true",
                   help="inclui a hipótese beam-4 de cada modelo no pool")
    p.add_argument("--beam_nbest", type=int, default=0,
                   help=">0: pool = n-best do beam (sem amostragem) — "
                        "rerank MBR de candidatos de alta qualidade")
    p.add_argument("--prefer_terms", action="store_true",
                   help="com --annotate: seleção SOFT — escolhe do pool o "
                        "candidato (em ordem de beam) que contém mais alvos "
                        "das dicas; empate = melhor beam. Sem MBR.")
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--dirs", choices=["fwd", "both"], default="fwd")
    p.add_argument("--annotate", default=None,
                   help="glossário p/ dica inline na fonte (rate 1.0, só por→yrl)")
    p.add_argument("--tag", default=None)
    p.add_argument("--save_hyp", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    annot = None
    if a.annotate:
        from term_annotate import Annotator
        annot = Annotator(a.annotate, rate=1.0, seed=0)
        print(f"[annot] dica inline: {len(annot.terms)} termos")

    tok = AutoTokenizer.from_pretrained(a.models[0])

    def gerar(model, texts, src_lang, tgt_lang, n_amostra):
        """n_amostra candidatos por amostragem (+beam se pedido) por fonte."""
        tok.src_lang = src_lang
        bos = tok.convert_tokens_to_ids(tgt_lang)
        pools = [[] for _ in texts]
        for i in range(0, len(texts), a.bsz):
            lote = texts[i:i + a.bsz]
            enc = tok(lote, return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                if a.beam_nbest > 0:
                    gen = model.generate(**enc, forced_bos_token_id=bos,
                                         num_beams=a.beam_nbest,
                                         num_return_sequences=a.beam_nbest,
                                         max_length=a.max_len,
                                         repetition_penalty=1.5)
                    dec = tok.batch_decode(gen, skip_special_tokens=True)
                    for j in range(len(lote)):
                        pools[i + j] += dec[j * a.beam_nbest:(j + 1) * a.beam_nbest]
                    continue
                if n_amostra > 0:
                    kw = dict(forced_bos_token_id=bos, do_sample=True,
                              max_length=a.max_len, temperature=a.temp,
                              num_return_sequences=n_amostra)
                    if a.epsilon > 0:
                        kw["epsilon_cutoff"] = a.epsilon
                    else:
                        kw["top_p"] = a.top_p
                    gen = model.generate(**enc, **kw)
                    dec = tok.batch_decode(gen, skip_special_tokens=True)
                    for j in range(len(lote)):
                        pools[i + j] += dec[j * n_amostra:(j + 1) * n_amostra]
                if a.include_beam:
                    gen_b = model.generate(**enc, forced_bos_token_id=bos,
                                           num_beams=4, max_length=a.max_len,
                                           repetition_penalty=1.5)
                    dec_b = tok.batch_decode(gen_b, skip_special_tokens=True)
                    for j in range(len(lote)):
                        pools[i + j].append(dec_b[j])
        return pools

    def mbr_escolhe(cands):
        """argmax da utilidade média chrF++ contra os demais candidatos."""
        unicos = list(dict.fromkeys(c.strip() for c in cands if c.strip()))
        if not unicos:
            return cands[0] if cands else ""
        if len(unicos) == 1:
            return unicos[0]
        melhor, melhor_u = unicos[0], -1.0
        for c in unicos:
            resto = [c2 for c2 in unicos if c2 is not c]
            u = sum(CHRF_SENT.sentence_score(c, [c2]).score for c2 in resto) / len(resto)
            if u > melhor_u:
                melhor, melhor_u = c, u
        return melhor

    n_por_modelo = max(1, a.nsamples // len(a.models))
    results = {}
    for s in a.sets:
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, s), encoding="utf-8")]
        src = [r["src"] for r in rows]; tgt = [r["tgt"] for r in rows]
        sl, tl = rows[0]["src_lang"], rows[0]["tgt_lang"]
        direcoes = {f"{sl}->{tl}": (src, tgt, sl, tl)}
        if a.dirs == "both":
            direcoes[f"{tl}->{sl}"] = (tgt, src, tl, sl)
        for direction, (X, Y, s_l, t_l) in direcoes.items():
            X_orig = X
            if annot is not None and s_l == "por_Latn":
                X = [annot(x) for x in X]
            if a.tag:
                X = [f"{a.tag} {x}" for x in X]
            pools = [[] for _ in X]
            for mpath in a.models:
                print(f"[mbr] {s} {direction}: amostrando {n_por_modelo}/fonte de {mpath}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    mpath, torch_dtype=torch.bfloat16).cuda().eval()
                mp = gerar(model, X, s_l, t_l, n_por_modelo)
                for i in range(len(X)):
                    pools[i] += mp[i]
                del model
                torch.cuda.empty_cache()
            if a.beam_nbest > 0 and len(a.models) == 1:
                # diagnóstico: o argmax do beam (1º do n-best) separa
                # "beam grande piorou" de "a seleção MBR piorou"
                top1 = [pool[0] if pool else "" for pool in pools]
                chrf_top1 = sacrebleu.corpus_chrf(top1, [Y], word_order=2).score
                print(f"[diag] argmax do beam-{a.beam_nbest}: chrF++ {chrf_top1:.2f}")
            if a.prefer_terms:
                assert annot is not None, "--prefer_terms exige --annotate"
                import re as _re
                RE_ALVO = _re.compile(r"\| ([^|]+) \|")
                hyp, n_sat, n_alv, n_troca = [], 0, 0, 0
                for x_in, pool in zip(X, pools):
                    alvos = list(dict.fromkeys(
                        t.strip() for t in RE_ALVO.findall(x_in) if t.strip()))[:2]
                    n_alv += len(alvos)
                    if not alvos or not pool:
                        hyp.append(pool[0] if pool else "")
                        continue
                    melhor, melhor_hits = pool[0], sum(
                        1 for t in alvos if t.lower() in pool[0].lower())
                    for c in pool[1:]:
                        hits = sum(1 for t in alvos if t.lower() in c.lower())
                        if hits > melhor_hits:
                            melhor, melhor_hits = c, hits
                    if melhor is not pool[0]:
                        n_troca += 1
                    n_sat += melhor_hits
                    hyp.append(melhor)
                print(f"[prefer] alvos satisfeitos: {n_sat}/{n_alv} | "
                      f"trocas de argmax: {n_troca}")
            else:
                hyp = [mbr_escolhe(pool) for pool in pools]
            if a.save_hyp:
                os.makedirs(a.save_hyp, exist_ok=True)
                nome = f"{s.replace('.jsonl','')}_{s_l[:3]}2{t_l[:3]}"
                with open(os.path.join(a.save_hyp, f"hyp_{nome}.jsonl"), "w",
                          encoding="utf-8") as fh:
                    for r_, x_, y_, h_, pool in zip(rows, X_orig, Y, hyp, pools):
                        fh.write(json.dumps({"id": r_.get("id"), "src": x_,
                                             "ref": y_, "hyp": h_,
                                             "n_pool": len(set(pool))},
                                            ensure_ascii=False) + "\n")
            chrf = sacrebleu.corpus_chrf(hyp, [Y], word_order=2).score
            try:
                bleu = sacrebleu.corpus_bleu(hyp, [Y], tokenize="flores200").score
            except Exception:
                bleu = sacrebleu.corpus_bleu(hyp, [Y]).score
            results[f"{s}|{direction}"] = {"chrF++": round(chrf, 2),
                                           "spBLEU": round(bleu, 2), "n": len(X)}
            if a.beam_nbest > 0:
                modo = f"rerank beam-{a.beam_nbest}"
            else:
                modo = (f"amostra n={a.nsamples}, temp={a.temp}, "
                        f"{'eps' if a.epsilon>0 else 'top_p'}, "
                        f"beam={'sim' if a.include_beam else 'nao'}")
            print(f"{s} {direction}: chrF++ {chrf:.2f} | spBLEU {bleu:.2f} "
                  f"(MBR {modo}, modelos={len(a.models)})")
    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    main()
