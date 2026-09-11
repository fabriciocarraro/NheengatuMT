# -*- coding: utf-8 -*-
"""Avaliação do comparador causal, nas DUAS direções.

Espelha o evaluate_yrl.py em tudo o que é comparável: mesmos conjuntos, beam 4,
mesmo nome de arquivo de hipóteses (hyp_<set>_<sl>2<tl>.jsonl) e mesma chave de
resultado ("<set>|<sl>-><tl>"). Assim term_success.py e as auditorias de chrF
rodam sobre a saída do LLM sem uma linha de adaptação.

TRÊS DIFERENÇAS DELIBERADAS em relação ao baseline, todas para NÃO enfraquecer
o comparador de graça (declaradas em docs/PREREGISTRO_ARQUITETURA.md §5b):

 1. repetition_penalty aplicada SÓ à continuação. Num decoder-only o processor
    padrão do HF enxerga `input_ids` inteiro — prompt incluído — então penalizaria
    o modelo por reusar qualquer token da frase-fonte. O baseline seq2seq só
    enxerga o lado alvo. Medido no dev_const: 8,4% dos tokens do alvo yrl e 14,0%
    do alvo pt também ocorrem na fonte; com o processor padrão levariam 1,5x de
    castigo sem motivo. Ver RepPenSoNaContinuacao.
 2. Orçamento de comprimento convertido de tokenizador. O baseline corta em 192
    tokens NLLB; a fertilidade medida no yrl é 2,74 (Qwen) contra 2,68 (NLLB),
    logo o equivalente é ~196 tokens Qwen. Usar os 160 originais cortaria 8
    referências do dev_const, contra 3 do baseline.
 3. Extração da 1ª linha não vazia. Um decoder-only pode emendar o "próximo
    exemplo" depois da tradução; o seq2seq não tem como. Quantas vezes isso
    dispara é reportado (campo `emendas`) — se for alto, o modelo está divagando
    e o número tem de ser lido com isso em mente.

Uso:
  python evaluate_causal.py --model runs/TAG/best --data_dir $DATA/norm3 \
      --sets dev_const.por-yrl.jsonl dev_extra.por-yrl.jsonl dev_fala.por-yrl.jsonl \
      --save_hyp runs/TAG/hyp_dev --out runs/TAG/eval_dev.json
"""
import argparse, json, os, re, sys
import torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor

sys.stdout.reconfigure(encoding="utf-8")

# DUPLICADO DE PROPÓSITO do train_causal.py: se os moldes divergirem, o modelo
# é avaliado num prompt que nunca viu. O teste test_causal_logic.py compara os
# dois arquivos e falha se saírem do lugar.
NOMES = {"por_Latn": "português", "yrl_Latn": "nheengatu", "fra_Latn": "francês"}

def molde(sl, tl):
    a, b = NOMES[sl], NOMES[tl]
    return f"Traduza do {a} para o {b}.\n{a.capitalize()}: {{src}}\n{b.capitalize()}:"

PROMPT = {(sl, tl): molde(sl, tl) for sl in NOMES for tl in NOMES if sl != tl}


class RepPenSoNaContinuacao(LogitsProcessor):
    """repetition_penalty restrita ao que o modelo gerou (nota 1 do cabeçalho).

    Com padding à esquerda todos os prompts do lote têm o mesmo comprimento,
    então basta fatiar em n_prompt. Vale também sob beam search, onde as linhas
    são (batch x beams) mas o prompt continua do mesmo tamanho."""
    def __init__(self, penalty, n_prompt):
        self.penalty, self.n_prompt = penalty, n_prompt
    def __call__(self, input_ids, scores):
        ger = input_ids[:, self.n_prompt:]
        if ger.shape[1] == 0:
            return scores
        s = torch.gather(scores, 1, ger)
        s = torch.where(s < 0, s * self.penalty, s / self.penalty)
        return scores.scatter(1, ger, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_class", default="AutoModelForCausalLM",
                   help="mesma classe usada no treino (ex. Qwen3_5ForCausalLM em "
                        "checkpoint multimodal)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--rep_pen", type=float, default=1.5)   # igual ao baseline, mas ver nota 1
    p.add_argument("--max_len", type=int, default=196,
                   help="teto de tokens Qwen p/ fonte e p/ geração; 196 = equivalente "
                        "medido dos 192 tokens NLLB do baseline (nota 2)")
    p.add_argument("--save_hyp", default=None)
    p.add_argument("--annotate", default=None,
                   help="glossario.jsonl — anota termos no lado PT (por→yrl) com rate=1.0")
    a = p.parse_args()

    annot = None
    if a.annotate:
        from term_annotate import Annotator
        annot = Annotator(a.annotate, rate=1.0, seed=0)
        print(f"[annot] inferência com terminologia inline: {len(annot.terms)} termos")

    import inspect, transformers
    def kw(f, nome, valor):
        """`torch_dtype`->`dtype` e a saída do `use_fast` no transformers 5.x:
        detecta o nome real em vez de apostar (ver kw_dtype no train_causal.py)."""
        return {nome: valor} if nome in inspect.signature(f).parameters else {}

    tok = AutoTokenizer.from_pretrained(
        a.model, **kw(AutoTokenizer.from_pretrained, "use_fast", True))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # obrigatório para geração em lote em decoder-only
    Cls = getattr(transformers, a.model_class, None)
    if Cls is None:
        sys.exit(f"[erro] transformers {transformers.__version__} não expõe "
                 f"'{a.model_class}'")
    dt = kw(Cls.from_pretrained, "dtype", torch.bfloat16) or {"torch_dtype": torch.bfloat16}
    model = Cls.from_pretrained(a.model, **dt).cuda().eval()
    print(f"[modelo] {a.model_class} | "
          f"{sum(p_.numel() for p_ in model.parameters())/1e9:.3f} B parâmetros")
    # Qwen embarca temperature/top_p/top_k no generation_config. Com do_sample=False
    # eles só rendem warnings, mas deixá-los é convite a comportamento surpresa.
    for k in ("temperature", "top_p", "top_k"):
        if getattr(model.generation_config, k, None) is not None:
            setattr(model.generation_config, k, None)

    emendas = [0]      # quantas gerações traziam texto depois da 1ª linha
    pensou = [0]       # quantas trouxeram bloco de raciocínio

    # O Qwen3.5 é um modelo de RACIOCÍNIO: sem isso ele emite "<think> ... </think>"
    # antes da resposta, e qualquer recorte ingênuo devolve o marcador em vez da
    # tradução (medido: 40/40 hipóteses viraram literalmente "<think>").
    # Duas defesas: proibir o token na geração, e limpar o que escapar.
    ids_think = [i for i in (tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>"))
                 if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]
    if ids_think:
        print(f"[think] suprimindo tokens de raciocínio: {ids_think}")

    RE_THINK = re.compile(r"<think>.*?</think>", re.S)

    def limpa(t):
        """Remove bloco de raciocínio (fechado ou não) e devolve a 1ª linha útil."""
        if "<think>" in t:
            pensou[0] += 1
            t = RE_THINK.sub(" ", t)
            t = t.split("</think>")[-1]        # bloco aberto sem fechar
            t = t.replace("<think>", " ")
        linhas = [x.strip() for x in t.strip().split("\n") if x.strip()]
        if len(linhas) > 1:
            emendas[0] += 1
        return linhas[0] if linhas else ""

    def corta(txt):
        """Trunca a FONTE no mesmo orçamento do baseline (nota 2)."""
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        return txt if len(ids) <= a.max_len else tok.decode(ids[:a.max_len],
                                                            skip_special_tokens=True)

    def translate(texts, src_lang, tgt_lang):
        out = []
        for i in range(0, len(texts), a.bsz):
            lote = [PROMPT[(src_lang, tgt_lang)].format(src=corta(t))
                    for t in texts[i:i+a.bsz]]
            enc = tok(lote, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            n_prompt = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **enc, num_beams=a.beam, do_sample=False,
                    max_new_tokens=a.max_len,
                    logits_processor=[RepPenSoNaContinuacao(a.rep_pen, n_prompt)],
                    suppress_tokens=ids_think or None,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            # decodifica SEM pular especiais: é preciso enxergar o <think> para
            # limpá-lo; os demais marcadores saem no split de linha/strip
            for t in tok.batch_decode(gen[:, n_prompt:], skip_special_tokens=False):
                for m in (tok.eos_token, tok.pad_token, "<|im_start|>", "<|im_end|>",
                          "<|endoftext|>"):
                    if m:
                        t = t.replace(m, "\n")
                out.append(limpa(t))
            print(f"   {min(i+a.bsz, len(texts))}/{len(texts)}", end="\r", flush=True)
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
            emendas[0] = 0; pensou[0] = 0
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
            vazias = sum(1 for h in hyp if not h.strip())
            results[f"{s}|{direction}"] = {"chrF++": round(chrf, 2),
                                           "spBLEU": round(bleu, 2), "n": len(X),
                                           "vazias": vazias, "emendas": emendas[0],
                                           "pensou": pensou[0]}
            print(f"{s} {direction}: chrF++ {chrf:.2f} | spBLEU {bleu:.2f} "
                  f"| vazias {vazias} | emendas {emendas[0]} | pensou {pensou[0]}")
    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    main()
