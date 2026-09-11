# -*- coding: utf-8 -*-
"""Gera os dados v2 do pacote de treino: cluster/data/{raw,norm,uni} -> cluster/data/{raw2,norm2,uni2}.

Três limpezas cirúrgicas, motivadas pelo diagnóstico das amostras (RESULTADOS
Adendo 4) — o 1.3B reproduziu "{o mesmo que:" e SESEWÁRA na saída:

  R1. Metatexto de dicionário (Ávila): remove "{...}" e fragmentos truncados
      "{o mesmo que:", "{v. tb.:" (214 pares medidos); remove "[sic]" e
      anotações "[lit. ...]". Casos ambíguos NÃO são tocados — vão para o log
      de revisão.
  R2. Marcadores de elipse "[...]"/"[…]" (1.334 pares medidos): removidos dos
      dois lados, com conserto de pontuação órfã (", ." -> ".", vírgula
      inicial etc.).
  R3. CAPS de diagramação no lado yrl (SESEWÁRA 67x etc.): rebaixados, EXCETO
      numerais romanos e pares de cabeçalho (fonte PT majoritariamente em
      caixa alta). Token inicial do segmento vira Capitalizado.

Princípio herdado do reparo dirigido: transformação só quando a regra é
inequívoca; todo o resto é logado, nunca alterado. Saída auditável em
work/relatorio_dados_v2.json.
"""
import json, re, sys, io
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "cluster" / "data"
VARIANTS = {"raw": "raw2", "norm": "norm2", "uni": "uni2"}
FILES = ["train.por-yrl.jsonl", "dev_const.por-yrl.jsonl", "dev_extra.por-yrl.jsonl",
         "test_const.por-yrl.jsonl", "test_extra.por-yrl.jsonl", "train.fra-yrl.jsonl"]

ROMAN = re.compile(r"^[IVXLCDM]+$")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# metatexto conhecido do dicionário de Ávila (dentro de chaves, fechadas ou truncadas)
META_BRACE = re.compile(r"\s*\{[^{}]*\}")          # {…} fechado, em qualquer posição
META_TRUNC = re.compile(r"\s*\{[^{}]*$")           # { aberto até o fim (truncado)
SIC = re.compile(r"\s*\[sic\.?\]", re.IGNORECASE)
LIT = re.compile(r"\s*\[lit\.[^\]]*\]")            # [lit. …] fechado
LIT_TRUNC = re.compile(r"\s*\[lit\.[^\]]*$")       # [lit. … truncado
ELLIPSIS = re.compile(r"\[(?:\.\.\.|…)\]")

def tidy(t):
    """Conserta pontuação órfã após remoções."""
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r",\s*([.;])", r"\1", t)
    t = re.sub(r"([:;!?])\s*\.", r"\1", t)          # "lontra:." -> "lontra:"
    t = re.sub(r"(?<!\.)\.\s*\.(?!\.)", ".", t)     # ".." isolado -> "." (protege "...")
    t = re.sub(r"^[\s,;:]+", "", t)
    t = re.sub(r"[,;]\s*$", "", t)                  # vírgula/; órfã no fim
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

META_MARKER = re.compile(
    r"\s*\{\s*(?:o mesmo que|v\.\s?tb\.?|var\.|sin[ôo]n\.?)[^{}]*\}?\s*$",
    re.IGNORECASE)

def clean_metatext(t, log):
    # pré-passe: marcadores inequívocos do dicionário dispensam a trava de
    # tamanho (frases curtas como "Vou fachear. {o mesmo que:" são corte certo)
    pre = t
    t = META_MARKER.sub("", t)
    marker_changed = (t != pre)
    orig = t                               # a trava avalia só o que vem a seguir
    t2 = META_BRACE.sub("", t)
    t2 = META_TRUNC.sub("", t2)
    t2 = SIC.sub("", t2)
    t2 = LIT.sub("", t2)
    t2 = LIT_TRUNC.sub("", t2)
    if ("{" in t2) or ("}" in t2):        # sobrou chave que as regras não explicam
        log.append(("revisar_chave", pre))
        return (tidy(t) if marker_changed else t), marker_changed
    if len(t2.strip()) < max(3, len(orig.strip()) // 2) and t2 != orig:
        log.append(("revisar_corte_grande", pre))
        return (tidy(t) if marker_changed else t), marker_changed
    changed = (t2 != orig) or marker_changed
    return (tidy(t2) if changed else t), changed

def clean_ellipsis(t):
    t2 = ELLIPSIS.sub("", t)
    if t2 == t:
        return t, False
    return tidy(t2), True

def fix_caps(tgt, src, counter):
    letters = [c for c in src if c.isalpha()]
    src_upper_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0
    if len(letters) >= 4 and src_upper_ratio > 0.5:
        for m in WORD.finditer(tgt):
            w = m.group(0)
            if len(w) >= 2 and w == w.upper() and w != w.lower() and not ROMAN.match(w):
                counter["cabecalho_preservado"] += 1
        return tgt, False
    first = WORD.search(tgt)
    out, last, changed = [], 0, False
    for m in WORD.finditer(tgt):
        w = m.group(0)
        if len(w) >= 2 and w == w.upper() and w != w.lower() and not ROMAN.match(w):
            new = w.capitalize() if (first and m.start() == first.start()) else w.lower()
            out.append(tgt[last:m.start()]); out.append(new)
            last = m.end(); changed = True
            counter[w] += 1
    out.append(tgt[last:])
    return ("".join(out) if changed else tgt), changed

def main():
    report = {"variantes": {}, "revisao": [], "caps_tokens": {}, "resumo": {}}
    caps_counter = Counter()
    tot = Counter()
    for var, var2 in VARIANTS.items():
        (DATA / var2).mkdir(exist_ok=True)
        report["variantes"][var2] = {}
        for fname in FILES:
            src_path = DATA / var / fname
            rows, out_rows = [], []
            with open(src_path, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f]
            stats = Counter()
            review_log = []
            for r in rows:
                r = dict(r)
                for side in ("src", "tgt"):
                    t, ch = clean_metatext(r[side], review_log)
                    if ch: stats[f"metatexto_{side}"] += 1
                    t, ch = clean_ellipsis(t)
                    if ch: stats[f"elipse_{side}"] += 1
                    r[side] = t
                t, ch = fix_caps(r["tgt"], r["src"], caps_counter)
                if ch: stats["caps_tgt"] += 1
                r["tgt"] = t
                out_rows.append(r)
            assert len(out_rows) == len(rows)
            with open(DATA / var2 / fname, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            changed = sum(1 for a, b in zip(rows_orig_iter(src_path), out_rows) if a != b)
            stats["linhas_alteradas"] = changed
            report["variantes"][var2][fname] = dict(stats)
            for k, v in stats.items():
                tot[k] += v
            for item in review_log:
                report["revisao"].append({"variante": var, "arquivo": fname,
                                          "tipo": item[0], "texto": item[1][:200]})
    report["caps_tokens"] = dict(caps_counter.most_common())
    report["resumo"] = dict(tot)
    out = ROOT / "work" / "relatorio_dados_v2.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("== RESUMO v2 ==")
    for k, v in sorted(tot.items()):
        print(f"  {k}: {v}")
    print(f"  casos para revisão manual (não tocados): {len(report['revisao'])}")
    print(f"  tipos de CAPS rebaixados: {len([k for k in caps_counter if k != 'cabecalho_preservado'])}")
    print(f"  relatório completo: {out}")

def rows_orig_iter(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)

if __name__ == "__main__":
    main()
