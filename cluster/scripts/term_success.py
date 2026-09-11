# -*- coding: utf-8 -*-
"""Term Success Rate (TSR) — métrica de conformidade terminológica (padrão WMT23).

Para cada frase avaliada: encontra no léxico os termos PT presentes na FONTE e
verifica se alguma tradução yrl esperada aparece na HIPÓTESE (match exato por
fronteira de palavra, match por lema via tabela forma->lema, ou fuzzy >= limiar).
chrF sozinho NÃO detecta ganho terminológico (findings WMT23) — esta métrica é
pré-requisito da Fase 2.

Uso:
  python term_success.py --lexico work/lexico_pt_yrl_v0.jsonl \
      --lemmas work/lemas_yrl.tsv \
      --hyp runs/TAG/hyp_dev_const.jsonl [--min_pt_len 4] [--fuzzy 0.85] [--show_misses 15]

Formato de --hyp: JSONL com {"src": ..., "hyp": ...} (campos extras ignorados),
como o produzido por evaluate_yrl.py --save_hyp.
"""
import argparse, json, re, sys, io, unicodedata
from difflib import SequenceMatcher
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def deacc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))

def toks(s):
    return re.findall(r"[^\W\d_]+", s.lower(), re.UNICODE)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lexico", required=True, nargs="+",
                   help="um ou mais léxicos JSONL (v0 + bitermos minerados)")
    p.add_argument("--lemmas", default=None)
    p.add_argument("--hyp", required=True)
    p.add_argument("--min_pt_len", type=int, default=4,
                   help="comprimento mínimo (chars) de termo PT de 1 palavra")
    p.add_argument("--fuzzy", type=float, default=0.85)
    p.add_argument("--show_misses", type=int, default=15)
    p.add_argument("--incluir_suspeitos", action="store_true",
                   help="inclui entradas marcadas suspeito_formula pela mineração")
    a = p.parse_args()

    # carrega e DEDUPLICA por PT normalizado (léxico v0 + bitermos se sobrepõem):
    # variantes yrl são unidas; suspeitos de fórmula ficam de fora por padrão
    merged = {}
    for path in a.lexico:
        for l in open(path, encoding="utf-8"):
            e = json.loads(l)
            if e.get("suspeito_formula") and not a.incluir_suspeitos:
                continue
            key = norm(e["pt"])
            variants = e["yrl"] if isinstance(e["yrl"], list) else [e["yrl"]]
            if key in merged:
                for v in variants:
                    if norm(v) not in {norm(x) for x in merged[key]["yrl"]}:
                        merged[key]["yrl"].append(v)
            else:
                merged[key] = {"pt": e["pt"], "yrl": list(variants)}
    lex = list(merged.values())
    lemma = {}
    if a.lemmas:
        for line in open(a.lemmas, encoding="utf-8"):
            k, v = line.rstrip("\n").split("\t")
            lemma[k] = v

    def lemmatize(t):
        return lemma.get(t, t)

    def term_in_hyp(term, hyp_toks, hyp_lemmas, hyp_norm):
        tt = toks(term)
        if not tt:
            return False
        if len(tt) == 1:
            t = tt[0]; tl = lemmatize(t); td = deacc(t)
            for h, hl in zip(hyp_toks, hyp_lemmas):
                if h == t or hl == tl or deacc(h) == td:
                    return True
                if len(t) >= 5 and SequenceMatcher(None, h, t).ratio() >= a.fuzzy:
                    return True
            return False
        # multi-palavra: substring normalizada OU todos os tokens de conteúdo presentes
        if norm(term) in hyp_norm:
            return True
        content = [t for t in tt if len(t) >= 3]
        return bool(content) and all(
            any(h == t or lemmatize(h) == lemmatize(t) or
                (len(t) >= 5 and SequenceMatcher(None, h, t).ratio() >= a.fuzzy)
                for h in hyp_toks)
            for t in content)

    def matchable(s):
        """Texto de casamento: minúsculo, SEM pontuação (termo adjacente a
        vírgula era perdido — subcontagem de 31% medida no dev_const)."""
        s = re.sub(r"[^\w\s]", " ", s.lower())
        return " " + re.sub(r"\s+", " ", s).strip() + " "

    total = hits = 0
    misses = []
    per_len = Counter(); per_len_hit = Counter()
    for line in open(a.hyp, encoding="utf-8"):
        r = json.loads(line)
        src_match = matchable(r["src"])
        hyp_toks = toks(r["hyp"])
        hyp_lemmas = [lemmatize(t) for t in hyp_toks]
        hyp_norm = matchable(r["hyp"]).strip()
        # termos presentes na fonte; termo contido em termo maior JÁ CASADO na
        # mesma frase é suprimido (evita contar 'tribunal federal' dentro de
        # 'supremo tribunal federal')
        found = []
        for e in lex:
            pt = norm(e["pt"])
            if len(pt.split()) == 1 and len(pt) < a.min_pt_len:
                continue
            if f" {pt} " in src_match:
                found.append(e)
        found.sort(key=lambda e: -len(norm(e["pt"]).split()))
        counted_pts = []
        for e in found:
            pt = norm(e["pt"])
            if any(f" {pt} " in f" {big} " for big in counted_pts):
                continue
            counted_pts.append(pt)
            total += 1
            klen = "multi" if len(pt.split()) > 1 else "single"
            per_len[klen] += 1
            if any(term_in_hyp(v, hyp_toks, hyp_lemmas, hyp_norm) for v in e["yrl"]):
                hits += 1; per_len_hit[klen] += 1
            elif len(misses) < 1000:
                misses.append((e["pt"], e["yrl"][:3], r["src"][:70], r["hyp"][:70]))

    if not total:
        print("Nenhum termo do léxico encontrado nas fontes — verifique os arquivos.")
        return
    print(f"TSR: {100*hits/total:.1f}%  ({hits}/{total} ocorrências de termo)")
    for k in ("single", "multi"):
        if per_len[k]:
            print(f"  {k}: {100*per_len_hit[k]/per_len[k]:.1f}% ({per_len_hit[k]}/{per_len[k]})")
    if misses and a.show_misses:
        print(f"\nPrimeiros misses (de {len(misses)}):")
        for pt, yrl, src, hyp in misses[:a.show_misses]:
            print(f"  [{pt}] esperado {yrl}")
            print(f"     src: {src}")
            print(f"     hyp: {hyp}")

if __name__ == "__main__":
    main()
