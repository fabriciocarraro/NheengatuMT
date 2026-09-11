# -*- coding: utf-8 -*-
"""Mineração de bitermos PT↔yrl por coocorrência (Dice) na TM de treino.

Fonte da terminologia jurídica que os pares curtos não cobrem: pares LONGOS da
Constituição. Para cada n-grama PT recorrente, procura o n-grama yrl com maior
Dice nos mesmos pares. Só treino (norm2); dev/test jamais entram.

Saída: work/bitermos_minerados.jsonl (mesmo esquema do léxico v0) — candidatos
com dice >= limiar, para uso no TSR e na anotação inline. Termos ÓRFÃOS (1
ocorrência) por definição não são mineráveis — a fonte deles será o dicionário
de Ávila (extração de verbetes, próximo incremento).
"""
import json, re, sys, io
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

MIN_DF = 3          # frequência mínima do n-grama PT
MIN_DICE = 0.35
MAX_N = 3

PT_STOP = set("""a o e de da do das dos em um uma uns umas para com que se por
as os no na nos nas ao à às aos é são foi ser ter seu sua seus suas ele ela
não mais muito já também sobre entre como quando mediante sem sob pela pelo
pelas pelos lei leis nos termos forma casos caso qualquer outra outro cada""".split())

ROMAN = re.compile(r"^[ivxlcdm]+$")

def chunks(s):
    """Tokens em janelas delimitadas por pontuação — n-grama não cruza vírgula.
    Hífen ASCII NÃO é delimitador: compostos yrl (mira-itá, 3.270 ocorrências
    no treino) viram tokens adjacentes no MESMO chunk."""
    out = []
    for seg in re.split(r"[.,;:!?()\[\]—–/]", s.lower()):
        t = [w for w in re.findall(r"[^\W\d_]+", seg, re.UNICODE)
             if w != "º" and not ROMAN.match(w)]
        if t:
            out.append(t)
    return out

def ngrams_chunked(chunk_list, nmax):
    for tokens in chunk_list:
        for n in range(1, nmax + 1):
            for i in range(len(tokens) - n + 1):
                yield " ".join(tokens[i:i + n])

def content_pt(g):
    ws = g.split()
    if all(w in PT_STOP for w in ws):
        return False
    if ws[0] in PT_STOP or ws[-1] in PT_STOP:   # não começar/terminar em stopword
        return False
    if len(ws) == 1 and len(ws[0]) < 5:
        return False
    return True

def content_yrl(g):
    ws = g.split()
    return sum(len(w) for w in ws) >= 4 and any(len(w) >= 3 for w in ws)

rows = [json.loads(l) for l in
        open(ROOT / "cluster" / "data" / "norm2" / "train.por-yrl.jsonl", encoding="utf-8")]
pairs = [(chunks(r["src"]), chunks(r["tgt"]), r.get("grupo")) for r in rows]

pt_df, yrl_df = Counter(), Counter()
pt_pairs = defaultdict(set)
for idx, (pt, yrl, g) in enumerate(pairs):
    for gm in set(ngrams_chunked(pt, MAX_N)):
        if content_pt(gm):
            pt_df[gm] += 1; pt_pairs[gm].add(idx)
    for gm in set(ngrams_chunked(yrl, MAX_N)):
        if content_yrl(gm):
            yrl_df[gm] += 1

out = []
for gm, df in pt_df.items():
    if df < MIN_DF:
        continue
    best, best_dice = None, 0.0
    cand = Counter()
    for idx in pt_pairs[gm]:
        for ym in set(ngrams_chunked(pairs[idx][1], MAX_N)):
            if content_yrl(ym):
                cand[ym] += 1
    for ym, co in cand.items():
        if co < MIN_DF:
            continue
        dice = 2 * co / (df + yrl_df[ym])
        # prefere dice maior; empate -> n-grama yrl mais longo (mais específico)
        if dice > best_dice or (dice == best_dice and best and len(ym) > len(best)):
            best, best_dice = ym, dice
    if best and best_dice >= MIN_DICE:
        grupos = Counter(pairs[i][2] for i in pt_pairs[gm])
        out.append({"pt": gm, "yrl": [best], "dice": round(best_dice, 3),
                    "df": df, "grupo_dominante": grupos.most_common(1)[0][0],
                    "fontes": ["mineracao_dice"]})

# poda: remove n-gramas PT contidos em n-grama maior com dice >= igual - 0.05
out.sort(key=lambda e: (-len(e["pt"].split()), -e["dice"]))
kept, covered = [], []
for e in out:
    if any(e["pt"] in big and e["dice"] <= d + 0.05 for big, d in covered):
        continue
    kept.append(e); covered.append((e["pt"], e["dice"]))

# assinatura de fórmula repetida: >=3 PT grams apontando para o MESMO yrl —
# mantém sem flag só o de maior dice/df; os demais ficam marcados p/ auditoria
by_yrl = defaultdict(list)
for e in kept:
    by_yrl[e["yrl"][0]].append(e)
for ym, group in by_yrl.items():
    if len(group) >= 3:
        group.sort(key=lambda e: (-e["dice"], -e["df"]))
        for e in group[1:]:
            e["suspeito_formula"] = True
kept.sort(key=lambda e: -e["dice"])

path = ROOT / "work" / "bitermos_minerados.jsonl"
with open(path, "w", encoding="utf-8", newline="\n") as fh:
    for e in kept:
        fh.write(json.dumps(e, ensure_ascii=False) + "\n")
const = [e for e in kept if e["grupo_dominante"] == "constituicao"]
susp = sum(1 for e in kept if e.get("suspeito_formula"))
print(f"[bitermos] {len(kept)} minerados (dice>={MIN_DICE}, df>={MIN_DF}) -> {path}")
print(f"[bitermos] {len(const)} jurídicos | {susp} marcados suspeito_formula (auditoria)")
print("\nTop 12 jurídicos limpos (dice):")
for e in [e for e in const if not e.get("suspeito_formula")][:12]:
    print(f"  {e['dice']:.2f} df={e['df']:3d}  {e['pt']!r} -> {e['yrl'][0]!r}")
