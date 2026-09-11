# -*- coding: utf-8 -*-
"""Por que o TSR erra: classifica cada miss em 4 modos de falha.

O TSR diz QUANTO o modelo erra em terminologia. Isto diz POR QUE, separando
quatro causas que o agregado mistura:

  1 alvo_ausente  — nenhuma variante yrl esperada ocorre no lado ALVO do treino
  2 fonte_inedita — o termo PT nao ocorre no lado FONTE do treino
  3 colisao       — o termo PT ocorre no treino, mas NUNCA alinhado ao alvo
                    esperado (foi visto sistematicamente sob outra glosa)
  4 falha_modelo  — o par PT->yrl ocorre no treino >=1x e o modelo errou assim mesmo

E mede a ZONA ORFA: tokens de conteudo da fonte que nao ocorrem no treino e nao
tem entrada no lexico de scoring. Essa zona e invisivel ao TSR por construcao
(o lexico foi minerado do proprio corpus de treino) e inalcancavel pela dica de
glossario, que so injeta entradas do lexico.

O matcher e reimplementado VERBATIM a partir de cluster/scripts/term_success.py.
O script VALIDA a reimplementacao reproduzindo o TSR do arquivo de hipoteses
antes de classificar; se o total de ocorrencias nao bater com o esperado, o
resultado nao vale (foi essa checagem que faltou na auditoria de 05/08).

Uso:
  python scripts/classifica_misses_tsr.py \
      --hyp resultados/cluster_final/runs/<TAG>/hyp_TESTE/hyp_test_const.por-yrl_por2yrl.jsonl \
      --train cluster/data/norm3/train.por-yrl.jsonl \
      --eval-src release/benchmark/test_const.por-yrl.jsonl \
      --lexico work/glossario_fase2.jsonl work/lexico_pt_yrl_v0.jsonl \
      --lemmas work/lemas_yrl.tsv \
      [--dicionario work/glossario_verbetes.jsonl] [--esperado-total 232]

Nota: --train aponta para dados cuja redistribuicao depende de permissoes
pendentes (ver docs/PROVENIENCIA.md). Quem so tem release/train_aberto/ pode
rodar sobre ele; os numeros mudam porque a cobertura muda.
"""
import argparse, json, re, sys, unicodedata
from difflib import SequenceMatcher
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

# --- helpers copiados verbatim de term_success.py (nao alterar sem revalidar) ---
def norm(s):  return re.sub(r"\s+", " ", s.strip().lower())
def deacc(s): return "".join(c for c in unicodedata.normalize("NFD", s)
                             if not unicodedata.combining(c))
def toks(s):  return re.findall(r"[^\W\d_]+", s.lower(), re.UNICODE)
def matchable(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " " + re.sub(r"\s+", " ", s).strip() + " "

# tokens funcionais do PT: proxy grosseiro para "palavra de conteudo", junto
# com o corte de comprimento. A contagem e por forma de superficie, entao
# 'geologia' conta como orfa mesmo que 'geologico' ocorresse — a taxa e um
# limite superior frouxo. A disjuncao com o lexico, essa, nao depende do proxy.
STOP = set("""de da do das dos a o as os e ou em no na nos nas um uma umas uns para por
com que se ao aos pelo pela pelos pelas nem sua seu suas seus este esta estes estas esse
essa esses essas isso aquele aquela mais menos como quando onde entre sobre sem sob ate
apos ser ter haver seja sejam sera serao foi foram tem tera terao""".split())


def carrega_lexico(paths, incluir_suspeitos=False):
    merged = {}
    for path in paths:
        for l in open(path, encoding="utf-8"):
            e = json.loads(l)
            if e.get("suspeito_formula") and not incluir_suspeitos:
                continue
            key = norm(e["pt"])
            variants = e["yrl"] if isinstance(e["yrl"], list) else [e["yrl"]]
            if key in merged:
                for v in variants:
                    if norm(v) not in {norm(x) for x in merged[key]["yrl"]}:
                        merged[key]["yrl"].append(v)
            else:
                merged[key] = {"pt": e["pt"], "yrl": list(variants)}
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-src", required=True,
                    help="JSONL do conjunto de avaliacao, para medir a zona orfa")
    ap.add_argument("--lexico", required=True, nargs="+")
    ap.add_argument("--lemmas", required=True)
    ap.add_argument("--dicionario", default=None,
                    help="fonte externa (ex. verbetes do Avila) p/ medir cobertura da zona orfa")
    ap.add_argument("--esperado-total", type=int, default=None,
                    help="n de ocorrencias de termo esperado; aborta se divergir")
    ap.add_argument("--min-pt-len", type=int, default=4)
    ap.add_argument("--fuzzy", type=float, default=0.85)
    ap.add_argument("--min-tok-len", type=int, default=5,
                    help="corte de comprimento para 'palavra de conteudo'")
    a = ap.parse_args()

    lex = carrega_lexico(a.lexico)
    lemma = {}
    for line in open(a.lemmas, encoding="utf-8"):
        k, v = line.rstrip("\n").split("\t")
        lemma[k] = v
    lemmatize = lambda t: lemma.get(t, t)

    def term_in(term, tk, tl_, tn):
        tt = toks(term)
        if not tt:
            return False
        if len(tt) == 1:
            t = tt[0]; tl = lemmatize(t); td = deacc(t)
            for h, hl in zip(tk, tl_):
                if h == t or hl == tl or deacc(h) == td:
                    return True
                if len(t) >= 5 and SequenceMatcher(None, h, t).ratio() >= a.fuzzy:
                    return True
            return False
        if norm(term) in tn:
            return True
        content = [t for t in tt if len(t) >= 3]
        return bool(content) and all(
            any(h == t or lemmatize(h) == lemmatize(t) or
                (len(t) >= 5 and SequenceMatcher(None, h, t).ratio() >= a.fuzzy)
                for h in tk)
            for t in content)

    # ---------- 1. pontua e valida ----------
    total = hits = 0
    misses = []
    for line in open(a.hyp, encoding="utf-8"):
        r = json.loads(line)
        src_match = matchable(r["src"])
        tk = toks(r["hyp"]); tl_ = [lemmatize(t) for t in tk]
        tn = matchable(r["hyp"]).strip()
        found = [e for e in lex
                 if not (len(norm(e["pt"]).split()) == 1
                         and len(norm(e["pt"])) < a.min_pt_len)
                 and f" {norm(e['pt'])} " in src_match]
        found.sort(key=lambda e: -len(norm(e["pt"]).split()))
        counted = []
        for e in found:
            pt = norm(e["pt"])
            if any(f" {pt} " in f" {big} " for big in counted):
                continue
            counted.append(pt); total += 1
            if any(term_in(v, tk, tl_, tn) for v in e["yrl"]):
                hits += 1
            else:
                misses.append(e)

    print("=" * 68)
    print(f"TSR: {100*hits/total:.1f}%  ({hits}/{total} ocorrencias de termo)")
    if a.esperado_total is not None:
        ok = total == a.esperado_total
        print(f"validacao do matcher: esperado {a.esperado_total} ocorrencias — "
              f"{'OK' if ok else 'DIVERGIU'}")
        if not ok:
            sys.exit("ABORTADO: reimplementacao do matcher nao reproduz o total "
                     "publicado; qualquer classificacao daqui seria invalida.")
    print("=" * 68)

    # ---------- 2. indices do treino ----------
    TR = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    tr_src = [matchable(d["src"]) for d in TR]
    tr_tok = [toks(d["tgt"]) for d in TR]
    tr_lem = [[lemmatize(t) for t in tt] for tt in tr_tok]
    tr_nrm = [matchable(d["tgt"]).strip() for d in TR]

    def no_alvo(variantes, idxs):
        return [i for i in idxs
                if any(term_in(v, tr_tok[i], tr_lem[i], tr_nrm[i]) for v in variantes)]

    # ---------- 3. classifica ----------
    uniq, cont = {}, Counter()
    for e in misses:
        uniq.setdefault(norm(e["pt"]), e); cont[norm(e["pt"])] += 1

    det = []
    todos = range(len(TR))
    for k, e in sorted(uniq.items()):
        pt_idx = [i for i, s in enumerate(tr_src) if f" {k} " in s]
        n_par = len(no_alvo(e["yrl"], pt_idx))
        n_alvo = n_par if n_par else len(no_alvo(e["yrl"], todos))
        modo = ("1 alvo_ausente"  if n_alvo == 0 else
                "2 fonte_inedita" if not pt_idx else
                "3 colisao"       if n_par == 0 else
                "4 falha_modelo")
        det.append((modo, cont[k], k, e["yrl"][:2], len(pt_idx), n_par, n_alvo))

    agg, aggo = Counter(), Counter()
    for modo, n, *_ in det:
        agg[modo] += 1; aggo[modo] += n
    tm = sum(aggo.values()) or 1
    print(f"\nPOR QUE ERROU — {sum(aggo.values())} misses, {len(uniq)} termos distintos\n")
    print(f"{'modo':17s} {'termos':>6s} {'ocorr':>6s} {'%':>7s}")
    print("-" * 40)
    for modo in sorted(set(agg) | {"1 alvo_ausente", "2 fonte_inedita",
                                   "3 colisao", "4 falha_modelo"}):
        print(f"{modo:17s} {agg[modo]:6d} {aggo[modo]:6d} {100*aggo[modo]/tm:6.1f}%")

    print(f"\n{'modo':17s} {'n':>3s}  {'termo PT':26s} {'yrl esperado':22s} "
          f"{'src':>5s} {'par':>4s} {'alvo':>5s}")
    print("-" * 92)
    for modo, n, k, y, ns, npar, nal in sorted(det, key=lambda r: (r[0], -r[1])):
        print(f"{modo:17s} {n:3d}  {k[:26]:26s} {', '.join(y)[:21]:22s} "
              f"{ns:5d} {npar:4d} {nal:5d}")

    # ---------- 4. zona orfa ----------
    V = Counter()
    for d in TR:
        V.update(toks(d["src"]))
    lex_pt = {norm(e["pt"]) for e in lex}
    orf, tot_cont, com_lex = Counter(), 0, Counter()
    for d in (json.loads(l) for l in open(a.eval_src, encoding="utf-8")):
        for t in toks(d["src"]):
            if len(t) < a.min_tok_len or t in STOP:
                continue
            tot_cont += 1
            if V[t] == 0:
                orf[t] += 1
                if t in lex_pt:
                    com_lex[t] += 1

    print(f"\n\nZONA ORFA — o que o TSR nao pode ver\n")
    print(f"  tokens de conteudo na fonte      : {tot_cont}")
    print(f"  com 0 ocorrencias no treino      : {sum(orf.values())} "
          f"({100*sum(orf.values())/max(tot_cont,1):.1f}%) — {len(orf)} tipos")
    print(f"    destes, com entrada no lexico  : {sum(com_lex.values())} "
          f"({len(com_lex)} tipos)   <- a dica poderia agir")
    print(f"    destes, sem entrada no lexico  : {sum(orf.values())-sum(com_lex.values())} "
          f"({len(orf)-len(com_lex)} tipos)   <- invisivel ao TSR E a dica")

    if a.dicionario:
        dic = {}
        for l in open(a.dicionario, encoding="utf-8"):
            e = json.loads(l); dic.setdefault(norm(e["pt"]), e)
        cob = [t for t in orf if t in dic]
        print(f"\n  cobertura da zona orfa por {a.dicionario}:")
        print(f"    resolviveis      : {len(cob)} tipos / {sum(orf[t] for t in cob)} ocorr")
        print(f"    sem cobertura    : {len(orf)-len(cob)} tipos / "
              f"{sum(orf[t] for t in orf if t not in cob)} ocorr")
        for t in sorted(cob, key=lambda x: -orf[x])[:15]:
            y = dic[t]["yrl"]
            y = y if isinstance(y, str) else ", ".join(y[:2])
            print(f"      {t:22s} {orf[t]}x -> {y[:40]}")


if __name__ == "__main__":
    main()
