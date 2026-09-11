# -*- coding: utf-8 -*-
"""Gera release/ — a camada do corpus que pode ser redistribuída hoje.

Regra: um par só entra se a fonte dele estiver liberada por licença aberta ou
por ser ato oficial. Fontes com permissão pendente NÃO entram; para elas o
repositório oferece os scripts de extração, não o texto.

Base legal por fonte em docs/PROVENIENCIA.md. Rodar da raiz do projeto:
    python scripts/build_public_release.py
"""
import json, os, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTE = os.path.join(RAIZ, "cluster", "data", "norm3")
SAIDA = os.path.join(RAIZ, "release")

# Fontes redistribuíveis. Ver docs/PROVENIENCIA.md para a base de cada uma.
LIBERADAS = {
    "constituicao":  "Ato oficial (art. 8º, IV, Lei 9.610/1998) — sem proteção autoral",
    "treebank_ud":   "CC BY-NC-SA 4.0 (UD_Nheengatu-CompLin)",
    "refubium_fala": "CC BY-NC-SA 4.0 (DOI 10.17169/refubium-39406)",
    "rm_tycho":      "Uso livre declarado (Plataforma Tycho Brahe / FAPESP)",
    "catecismo1944": "Domínio público provável (idade)",
    # Autorização ESCRITA do autor por e-mail, 10/08/2026: "autorizo sim a
    # publicação dos pares utilizados, acompanhados dos devidos créditos".
    # Não nomeia licença — o que temos é permissão do titular com crédito
    # obrigatório, e é assim que NOTICE e PROVENIENCIA a declaram. Cobre os
    # 2.745 pares usados no corpus; NÃO cobre `rm_navarro` (titular distinto).
    "avila2021":     "Autorização escrita do autor (10/08/2026) — crédito obrigatório",
}
# Autorizadas para USO, mas com escopo de redistribuição a confirmar.
PENDENTE_ESCOPO = {"leetra_kariama", "leetra_tapajoara", "leetra_leitura", "leetra_kabari"}

BENCHMARK = ["dev_const", "test_const", "dev_fala", "test_fala"]
RETIDOS = ["dev_extra", "test_extra"]


def carrega(nome):
    p = os.path.join(FONTE, f"{nome}.por-yrl.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def grava(rows, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("=== BENCHMARK (conjuntos 100% redistribuíveis) ===")
    total_bench = 0
    for nome in BENCHMARK:
        rows = carrega(nome)
        grupos = collections.Counter(r.get("grupo", "?") for r in rows)
        alheias = [g for g in grupos if g not in LIBERADAS]
        if alheias:
            sys.exit(f"ABORTADO: {nome} contém fonte não liberada: {alheias}")
        grava(rows, os.path.join(SAIDA, "benchmark", f"{nome}.por-yrl.jsonl"))
        total_bench += len(rows)
        print(f"  {nome:12s} {len(rows):>5d}  {dict(grupos)}")

    print("\n=== TREINO (só as fontes liberadas) ===")
    treino = carrega("train")
    aberto = [r for r in treino if r.get("grupo") in LIBERADAS]
    grava(aberto, os.path.join(SAIDA, "train_aberto", "train.por-yrl.jsonl"))
    c = collections.Counter(r["grupo"] for r in aberto)
    for g, n in c.most_common():
        print(f"  {g:16s} {n:>5d}")
    print(f"  {'TOTAL':16s} {len(aberto):>5d} de {len(treino)} "
          f"({100*len(aberto)/len(treino):.1f}%)")

    print("\n=== RETIDO (permissão pendente — só o script de extração vai no repo) ===")
    retido = collections.Counter(r.get("grupo") for r in treino
                                 if r.get("grupo") not in LIBERADAS)
    for g, n in retido.most_common():
        marca = "escopo a confirmar" if g in PENDENTE_ESCOPO else "permissão pendente"
        print(f"  {g:20s} {n:>5d}  ({marca})")
    for nome in RETIDOS:
        rows = carrega(nome)
        fora = sum(1 for r in rows if r.get("grupo") not in LIBERADAS)
        print(f"  {nome:20s} {len(rows):>5d}  ({fora} de fonte restrita — conjunto inteiro retido)")

    print(f"\n=== escrito em release/ ===")
    print(f"  benchmark : {total_bench} segmentos em 4 arquivos")
    print(f"  treino    : {len(aberto)} pares")


if __name__ == "__main__":
    main()
