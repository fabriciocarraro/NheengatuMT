# -*- coding: utf-8 -*-
"""Glossario de inferencia estendido pelos verbetes do dicionario (Avila).

Motivacao (RESULTADOS_FASE2 §"Quantificacao do mecanismo"): o glossario da Fase 2
foi minerado do proprio corpus de treino, entao ele e uma PROJECAO do corpus e
tem exatamente a mesma fronteira de cobertura. A zona orfa — palavras da fonte
com 0 ocorrencia no treino — tem cobertura ZERO no lexico, e por isso a dica de
terminologia nao tem como agir ali. So uma fonte EXTERNA move a fronteira, e a
unica que temos sao os verbetes do dicionario.

Este script monta o arquivo que `evaluate_yrl.py --annotate` consome.

Tres armadilhas que ele resolve, e que quebram o experimento em silencio:

 1. `term_annotate.Annotator` FILTRA por nivel: so passam "aprovado" e
    "corrigido". Os verbetes tem nivel "verbete_avila" e seriam TODOS
    descartados — a rodada terminaria sem erro e sem anotar nada. Aqui eles
    recebem nivel "aprovado" e um campo `origem` que preserva a proveniencia
    (o Annotator ignora campos que nao conhece).

 2. O Annotator ordena candidatos por MAIS PALAVRAS primeiro e corta em
    `max_terms=2`. Verbetes de dicionario sao definicoes ("a banda de alem",
    "atravessar horizontalmente a ripa"), entao entradas longas DESLOCARIAM as
    do glossario auditado. Por isso so entram verbetes de ate 2 palavras, e
    nunca iniciados por artigo/preposicao.

 3. Em colisao de `pt`, o glossario da Fase 2 VENCE: ele foi auditado contra o
    corpus, o verbete nao.

Uso:
  python scripts/build_glossario_estendido.py \
      --base work/glossario_fase2.jsonl \
      --verbetes work/glossario_verbetes.jsonl \
      --out work/glossario_fase2_mais_avila.jsonl
"""
import argparse, json, re, sys

sys.stdout.reconfigure(encoding="utf-8")

# artigos, preposicoes e contracoes: um verbete iniciado por eles e frase de
# dicionario, nao termo, e vira gatilho de anotacao ruidoso
INICIO_RUIM = set("""a o as os um uma uns umas de da do das dos em no na nos nas
por pelo pela para com sem sob sobre ao aos e ou que se""".split())

LIMPO = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--verbetes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-palavras", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=4)
    a = ap.parse_args()

    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    vistos = {e["pt"].strip().lower() for e in base}
    print(f"base            : {len(base)} entradas ({a.base})")

    verb = [json.loads(l) for l in open(a.verbetes, encoding="utf-8")]
    print(f"verbetes brutos : {len(verb)}")

    novos, desc = [], {"sujo": 0, "longo": 0, "curto": 0, "inicio": 0,
                       "colide": 0, "sem_yrl": 0}
    for e in verb:
        pt = e["pt"].strip()
        if not LIMPO.fullmatch(pt):
            desc["sujo"] += 1; continue
        if len(pt) < a.min_chars:
            desc["curto"] += 1; continue
        pal = pt.split()
        if len(pal) > a.max_palavras:
            desc["longo"] += 1; continue
        if pal[0].lower() in INICIO_RUIM:
            desc["inicio"] += 1; continue
        if pt.lower() in vistos:
            desc["colide"] += 1; continue
        yrl = e["yrl"] if isinstance(e["yrl"], list) else [e["yrl"]]
        yrl = [y.strip() for y in yrl if y and y.strip()]
        if not yrl:
            desc["sem_yrl"] += 1; continue
        vistos.add(pt.lower())
        # nivel "aprovado" para PASSAR pelo filtro do Annotator; `origem`
        # preserva a proveniencia e nao e lido por ele
        novos.append({"pt": pt, "yrl": yrl, "nivel": "aprovado",
                      "origem": "verbete_avila"})

    print("descartados     : " + ", ".join(f"{k}={v}" for k, v in desc.items()))
    print(f"verbetes aceitos: {len(novos)}")

    with open(a.out, "w", encoding="utf-8") as f:
        for e in base:
            e = dict(e); e.setdefault("origem", "glossario_fase2")
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for e in novos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nescrito         : {a.out}  ({len(base) + len(novos)} entradas)")
    print("ATENCAO: este arquivo e so para --annotate (inferencia). O lexico de "
          "scoring do TSR fica CONGELADO em glossario_fase2 + lexico_pt_yrl_v0, "
          "senao os numeros deixam de comparar com o publicado.")


if __name__ == "__main__":
    main()
