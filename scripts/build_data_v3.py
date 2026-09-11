# -*- coding: utf-8 -*-
"""Gera a série v3: cluster/data/norm3 = norm2 + backlog auditado da Fase 2.

Novos blocos de treino (todos auditados por agentes com evidência; ver
RESULTADOS_FASE2):
  - Refubium 1.533 pares de fala espontânea (typos do PT corrigidos por
    revisão mínima; registro coloquial preservado). Um GRUPO DE SESSÃO
    inteiro (falantes disjuntos) sai do treino e vira dev_fala/test_fala.
  - Melgueiro 1.728 exemplos (pt_completo & !descartar).
  - LEETRA 527 (kariama/tapajoara realinhados + leitura/kabari auditados);
    linhas em CAPS de diagramação viram sentence-case (lição da v2).

Convenções herdadas: normalização = charmap de glifos do align.py (norm);
dev/test const/extra IDÊNTICOS ao norm2 (chrF comparável com Exp. 1-2);
esquema {src_lang, tgt_lang, src, tgt, grupo, id}.

Uso: python scripts/build_data_v3.py
"""
import json, os, re, shutil, sys, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "cluster" / "data" / "norm2"
N3 = ROOT / "cluster" / "data" / "norm3"
# Intermediários da construção. Sobrescreva com NHE_SCRATCH se quiser outro lugar.
SCRATCH = Path(os.environ.get("NHE_SCRATCH", ROOT / "work" / "scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)

# --------- normalização yrl (charmap do align.py, convenção 'norm') ---------
NG_CHARMAP = {
    "ū": "ũ", "ē": "ẽ", "ī": "ĩ", "ῖ": "ĩ", "Ē": "Ẽ", "Ū": "Ũ",
    "ữ": "ũ", "û": "ũ",
    "È": "É", "Ì": "Í", "ì": "í", "À": "Á", "à": "á", "Â": "Ã",
    "ŕ": "r", "ķ": "k", "ă": "ã", "ā": "ã",
    "´": "",
}
charmap_counts = Counter()

def norm_yrl(t):
    for a, b in NG_CHARMAP.items():
        if a in t:
            charmap_counts[a] += t.count(a)
            t = t.replace(a, b)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" ([,;.:!?])", r"\1", t)
    return t.strip()

def norm_pt(t):
    return re.sub(r" {2,}", " ", t).strip()

def frac_caps(s):
    letras = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letras) / max(1, len(letras))

def sentence_case(s):
    s = s.lower()
    for i, c in enumerate(s):
        if c.isalpha():
            return s[:i] + c.upper() + s[i + 1:]
    return s

def chave(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower())
                if not unicodedata.combining(c))
    return re.sub(r"[^\w ]", "", re.sub(r"\s+", " ", s)).strip()

def linha(src, tgt, grupo, id_):
    return {"src_lang": "por_Latn", "tgt_lang": "yrl_Latn",
            "src": src, "tgt": tgt, "grupo": grupo, "id": id_}

# ---------------- refubium: typos + holdout de sessão ----------------
ref = [json.loads(l) for l in open(ROOT / "work" / "refubium_pares.jsonl", encoding="utf-8")]
fixes, fix_rej = {}, 0
for c in range(3):
    for f in map(json.loads, open(SCRATCH / f"typo_fix_{c}.jsonl", encoding="utf-8")):
        antes = ref[f["id"]]["pt"]
        ratio = SequenceMatcher(None, antes, f["pt"]).ratio()
        if ratio >= 0.7 and abs(len(antes.split()) - len(f["pt"].split())) <= 2:
            fixes[f["id"]] = f["pt"]
        else:
            fix_rej += 1
for i, e in enumerate(ref):
    if i in fixes:
        e["pt"] = fixes[i]

def grupo_sessao(arquivo):
    s = re.sub(r"^(?:Concurso|Conto|Maptask|Quem)(?:_\d+)?_", "", arquivo)
    return re.sub(r"_NHEEN\.eaf$", "", s)

por_grupo = Counter(grupo_sessao(e["arquivo"]) for e in ref)
holdout = min(por_grupo, key=lambda g: abs(por_grupo[g] - 280))
ref_train = [e for e in ref if grupo_sessao(e["arquivo"]) != holdout]
ref_hold = sorted((e for e in ref if grupo_sessao(e["arquivo"]) == holdout),
                  key=lambda e: (e["tarefa"], e["arquivo"], e["t0_ms"]))
# dedupe por conteúdo DENTRO do holdout: backchannels repetidos ("ta", "aham")
# cairiam nos dois lados do split alternado, correlacionando dev e teste
vist_h, hold_unico = set(), []
for e in ref_hold:
    k = (chave(e["pt"]), chave(e["yrl"]))
    if k in vist_h:
        continue
    vist_h.add(k)
    hold_unico.append(e)
dev_fala = hold_unico[0::2]
test_fala = hold_unico[1::2]

def rows_refubium(lst):
    out = []
    for e in lst:
        aid = re.sub(r"\.eaf$", "", e["arquivo"])
        out.append(linha(norm_pt(e["pt"]), norm_yrl(e["yrl"]), "refubium_fala",
                         f"ref_{aid}_{e['falante']}_{e['t0_ms']}"))
    return out

# ---------------- melgueiro ----------------
mel = [json.loads(l) for l in open(ROOT / "work" / "melgueiro_pares.jsonl", encoding="utf-8")]
mel_rows, seq = [], Counter()
for e in mel:
    if not e.get("pt_completo") or e.get("descartar"):
        continue
    seq[e["pagina"]] += 1
    # "melgueiro2022" já existe no agregado (144 linhas, ids x0xxxx) —
    # bloco novo da extração própria leva sufixo _ex para preservar proveniência
    mel_rows.append(linha(norm_pt(e["pt"]), norm_yrl(e["yrl"]),
                          "melgueiro2022_ex", f"mel_p{e['pagina']}_{seq[e['pagina']]}"))

# ---------------- leetra ----------------
lee = [json.loads(l) for l in open(ROOT / "work" / "leetra_pares.jsonl", encoding="utf-8")]
lee_rows, seq2, n_caps = [], Counter(), 0
for e in lee:
    y, p = e["yrl"], e["pt"]
    if frac_caps(y) > 0.6:
        y, p = sentence_case(y), sentence_case(p)
        n_caps += 1
    seq2[e["livro"]] += 1
    lee_rows.append(linha(norm_pt(p), norm_yrl(y),
                          f"leetra_{e['livro']}",
                          f"lee_{e['livro']}_p{e['pag_yrl']}_{seq2[e['livro']]}"))

# ---------------- dedupe + guarda de vazamento ----------------
n2_train = [json.loads(l) for l in open(N2 / "train.por-yrl.jsonl", encoding="utf-8")]
eval_sets = {}
for nome in ("dev_const", "dev_extra", "test_const", "test_extra"):
    eval_sets[nome] = [json.loads(l) for l in open(N2 / f"{nome}.por-yrl.jsonl", encoding="utf-8")]

vistos = {(chave(r["src"]), chave(r["tgt"])) for r in n2_train}
tgt_eval = {chave(r["tgt"]) for rows in eval_sets.values() for r in rows}

novos, st = [], Counter()
for r in rows_refubium(ref_train) + mel_rows + lee_rows:
    k = (chave(r["src"]), chave(r["tgt"]))
    if not r["src"] or not r["tgt"]:
        st["vazio"] += 1
        continue
    if k in vistos:
        st["dup"] += 1
        continue
    if chave(r["tgt"]) in tgt_eval:
        st["vazamento_eval_excluido"] += 1
        continue
    vistos.add(k)
    novos.append(r)

# dev/test_fala não podem vazar do treino (falantes já são disjuntos; checa mesmo assim)
tgt_train_novo = {chave(r["tgt"]) for r in novos}
df_rows = [r for r in rows_refubium(dev_fala) if chave(r["tgt"]) not in tgt_train_novo]
tf_rows = [r for r in rows_refubium(test_fala) if chave(r["tgt"]) not in tgt_train_novo]

# ---------------- escrita norm3 ----------------
N3.mkdir(parents=True, exist_ok=True)
for f in N2.glob("*.jsonl"):
    shutil.copy2(f, N3 / f.name)

def escreve(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

escreve(N3 / "train.por-yrl.jsonl", n2_train + novos)
escreve(N3 / "dev_fala.por-yrl.jsonl", df_rows)
escreve(N3 / "test_fala.por-yrl.jsonl", tf_rows)

# ---------------- relatório ----------------
por_grupo_novo = Counter(r["grupo"] for r in novos)
print("=== build v3 ===")
print(f"typo fixes aplicados: {len(fixes)} (rejeitados na validação: {fix_rej})")
print(f"holdout de sessão refubium: {holdout} ({por_grupo[holdout]} pares -> "
      f"dev_fala {len(df_rows)} / test_fala {len(tf_rows)})")
print(f"caps->sentence-case (leetra): {n_caps}")
print(f"charmap aplicado: {dict(charmap_counts) or 'nada a trocar'}")
print(f"descartes: {dict(st)}")
print(f"novos no treino: {len(novos)} {dict(por_grupo_novo)}")
print(f"train.por-yrl: {len(n2_train)} (norm2) + {len(novos)} = {len(n2_train) + len(novos)}")
print(f"norm3 completo em {N3}")
