# -*- coding: utf-8 -*-
"""Extrai pares exemplo-yrl ↔ tradução-PT da tese de Melgueiro (2022, UnB) —
vocabulário de Stradelli atualizado para o Nheengatu contemporâneo.

Padrão-alvo: `<frase yrl> – <tradução pt>.` — mas o texto é denso (headwords,
glosas e exemplos colados, pontuação final ausente às vezes), então a fronteira
de cada lado é decidida por CAMINHADA TOKEN A TOKEN a partir do separador:
para trás enquanto o token parecer yrl, para frente enquanto parecer PT.
O classificador usa o vocabulário do nosso próprio corpus (lado yrl vs lado PT
de cluster/data/norm2) + diacríticos nasais + morfologia yrl.

Saída: work/melgueiro_pares.jsonl {"yrl", "pt", "pagina"}
Fonte: MELGUEIRO, E. M. C. Tese (UnB, 2022) — citação obrigatória.
"""
import fitz, json, re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "work" / "novas_fontes" / "melgueiro_2022_stradelli_atual.pdf"
OUT = ROOT / "work" / "melgueiro_pares.jsonl"

TILDE = re.compile(r"[ãẽĩũỹ]")
YRL_MORF = re.compile(r"^(?:u|re|ya|pe|ta|ai)[a-zà-ÿ]{4,}$")
PT_STOP = set("""o a os as de da do das dos que não nao ele ela eles elas você
voce nós para pra com em no na nos nas um uma uns umas é foi ser está estava
meu minha seu sua teu tua nosso nossa quando onde como porque mas ou já mais
muito pouco tudo nada alguém ninguém gente coisa dia casa água homem mulher""".split())

def strip_p(t):
    return re.sub(r"^[^\wãẽĩũỹà-ÿ]+|[^\wãẽĩũỹà-ÿ]+$", "", t.lower())

def load_vocab():
    yv, pv = set(), set()
    d = ROOT / "cluster" / "data" / "norm2"
    for f in d.glob("*.por-yrl.jsonl"):
        for r in map(json.loads, open(f, encoding="utf-8")):
            for t in r["tgt"].split():
                t = strip_p(t)
                if t: yv.add(t)
            for t in r["src"].split():
                t = strip_p(t)
                if t: pv.add(t)
    return yv, pv

YV, PV = load_vocab()
PV |= PT_STOP

def cls(tok):
    t = strip_p(tok)
    if not t:
        return "punct"
    if t.isdigit():
        return "digit"
    if TILDE.search(t):
        return "yrl"
    iny, inp = t in YV, t in PV
    if iny and inp:
        return "amb"
    if iny:
        return "yrl"
    if inp:
        return "pt"
    if YRL_MORF.match(t):
        return "yrl?"
    return "unk"

FIM = re.compile(r"[.!?;]$")

def walk_left(toks):
    """Caminha para trás a partir do separador coletando a frase yrl."""
    out = []
    for tok in reversed(toks[-40:]):
        c = cls(tok)
        if c in ("pt", "digit", "punct"):
            break
        if out and FIM.search(tok):        # pontuação final = frase anterior
            break
        out.append(tok)
        if len(out) >= 25:
            break
    out.reverse()
    n_yrl = sum(1 for t in out if cls(t) in ("yrl", "yrl?"))
    if len(out) >= 2 and n_yrl >= 1 and n_yrl / len(out) >= 0.4:
        return " ".join(out)
    return None

def walk_right(toks):
    """Caminha para frente a partir do separador coletando a tradução PT."""
    out, unk_run = [], 0
    for tok in toks[:30]:
        c = cls(tok)
        if c in ("yrl", "digit") or (not out and c in ("yrl?", "unk")):
            break
        unk_run = unk_run + 1 if c in ("unk", "yrl?") else 0
        if unk_run >= 2:                    # 2 tokens não-PT seguidos: verbete novo
            out = out[:-1]
            break
        out.append(tok)
        if FIM.search(tok):
            break
    n_pt = sum(1 for t in out if cls(t) in ("pt", "amb"))
    if len(out) >= 2 and n_pt >= 2:
        return " ".join(out)
    return None

def page_text_columns(page):
    blocks = page.get_text("blocks")
    mid = page.rect.width / 2
    return "\n".join(b[4] for b in sorted(
        blocks, key=lambda b: (0 if b[0] < mid * 0.75 else 1, round(b[1]))))

SEP = re.compile(r"\s+[–—-]\s+|(?<=[a-zà-ÿãẽĩũỹ])[–—-]\s+(?=[A-ZÀ-Ú])")

def main():
    doc = fitz.open(PDF)
    pares, seen, n_sep = [], set(), 0
    for p in range(len(doc)):
        txt = page_text_columns(doc[p])
        txt = re.sub(r"-\n(?=[a-zà-ÿ])", "", txt)
        txt = re.sub(r"\s*\n\s*", " ", txt)
        for m in SEP.finditer(txt):
            n_sep += 1
            l = walk_left(txt[:m.start()].split())
            r = walk_right(txt[m.end():].split())
            if not l or not r:
                continue
            r = r.rstrip(";").strip()
            key = (l.lower(), r.lower())
            if key in seen:
                continue
            seen.add(key)
            pares.append({"yrl": l, "pt": r, "pagina": p + 1,
                          "pt_completo": bool(FIM.search(r))})
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for e in pares:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    tok = sum(len(e["yrl"].split()) for e in pares)
    fim_ok = sum(1 for e in pares if FIM.search(e["pt"]))
    print(f"[melgueiro] {len(pares)} pares únicos (de {n_sep} separadores) -> {OUT}")
    print(f"  ~{tok} tokens yrl | PT terminando em pontuação: {fim_ok}/{len(pares)}")

if __name__ == "__main__":
    main()
