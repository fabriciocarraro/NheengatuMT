# -*- coding: utf-8 -*-
"""Extrai pares yrl↔PT dos 4 livros bilíngues do LEETRA/UFSCar (autorização
04/08/2026: "citar partes com crédito à Revista Leetra Indígena e aos autores").

Estrutura dos livros: texto yrl seguido da tradução PT (página alternada ou
parágrafos na mesma página). Pipeline:
  1. blocos de texto por página (PyMuPDF), sem cabeçalhos/rodapés repetidos;
  2. classificação de língua por bloco (vocabulário do corpus norm2 +
     diacríticos nasais + morfologia yrl);
  3. blocos consecutivos da mesma língua viram SEGMENTOS; cada segmento yrl
     casa com o segmento PT seguinte (guarda de razão de tamanho);
  4. alinhamento sentença a sentença por DP de comprimentos (Gale-Church
     simplificado; beads 1:1, 1:2, 2:1, 2:2; fallback segmento inteiro
     quando curto).

Saída: work/leetra_pares.jsonl
  {"livro", "pag_yrl", "pag_pt", "yrl", "pt", "bead", "caps"}
"""
import fitz, json, re, sys, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "LEETRA"
OUT = ROOT / "work" / "leetra_pares.jsonl"

LIVROS = {
    "kariama": "Veiga_org_2015_EscolaKariamaContaUmbuesa_Baniwa.pdf",
    "tapajoara": "Tapajoaracorrigido2016final.pdf",
    "leitura": "livro de leitura e escrita em yegatu.pdf",
    "kabari": "Kabari Teepa 2015.pdf",
}

# ---------- classificador de língua (mesmo esquema do Melgueiro) ----------
TILDE = re.compile(r"[ãẽĩũỹ]")
YRL_MORF = re.compile(r"^(?:u|re|ya|pe|ta|ai)[a-zà-ÿ]{4,}$")
PT_STOP = set("""o a os as de da do das dos que não nao ele ela eles elas você
voce nós para pra com em no na nos nas um uma uns umas é foi ser está estava
meu minha seu sua nosso nossa quando onde como porque mas ou já mais muito
pouco tudo nada gente coisa dia casa água homem mulher escola professor
história vamos nossos suas seus pelo pela""".split())

def strip_p(t):
    return re.sub(r"^[^\wãẽĩũỹà-ÿ]+|[^\wãẽĩũỹà-ÿ]+$", "", t.lower())

def load_vocab():
    yv, pv = set(), set()
    for f in (ROOT / "cluster" / "data" / "norm2").glob("*.por-yrl.jsonl"):
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

def lang_bloco(txt):
    y = p = 0
    for tok in txt.split():
        t = strip_p(tok)
        if not t or t.isdigit():
            continue
        if TILDE.search(t):
            y += 1; continue
        iny, inp = t in YV, t in PV
        if iny and inp:
            continue
        if iny or YRL_MORF.match(t):
            y += 1
        elif inp:
            p += 1
    if y >= 3 and y >= 2 * p: return "yrl"
    if p >= 3 and p >= 2 * y: return "pt"
    return "outro"

# ---------- limpeza de página ----------
CREDITO = re.compile(r"^(fonte[.:]|desenho de|foto[.:]|ilustra|rev\. leetra)", re.I)
LEADER = re.compile(r"\.{4,}")                      # linhas de sumário

def legenda_pt(t):
    """Créditos de autor/legendas de foto: PT curto que NÃO é tradução."""
    if len(t) >= 150:
        return False
    toks = [x for x in t.split() if x.isalpha()]
    if toks and sum(x.isupper() for x in toks) / len(toks) >= 0.5:
        return False                       # texto em CAPS = conteúdo, não legenda
    if re.search(r"\(\d|\bprofe?s|\bsrª?\b|\bd\.|confeccion|encontr[oa]\b|"
                 r"lembra|universidade|laborat[óo]rio", t, re.I):
        return True
    tc = sum(1 for x in toks if x[0].isupper())
    return bool(toks) and tc / len(toks) >= 0.5

def blocos_livro(doc):
    """[(pagina, texto_bloco)] sem cabeçalhos/rodapés repetidos nem créditos."""
    brutos, freq = [], Counter()
    for p in range(len(doc)):
        for b in sorted(doc[p].get_text("blocks"), key=lambda b: (round(b[1]), b[0])):
            t = re.sub(r"\s+", " ", b[4]).strip()
            if not t:
                continue
            brutos.append((p + 1, t))
            freq[re.sub(r"\d+", "#", t.lower())[:40]] += 1
    limiar = max(3, int(0.2 * len(doc)))
    out = []
    for pag, t in brutos:
        chave = re.sub(r"\d+", "#", t.lower())[:40]
        if len(t) < 80 and freq[chave] >= limiar:      # cabeçalho/rodapé
            continue
        if re.fullmatch(r"[\divxlc .\-–—]+", t.lower()):  # número de página
            continue
        if CREDITO.match(t):
            continue
        out.append((pag, t))
    return out

# ---------- segmentos ----------
def segmentos(blocos):
    segs = []
    for pag, t in blocos:
        if LEADER.search(t):
            continue
        lg = lang_bloco(t)
        if lg == "pt" and legenda_pt(t):
            continue                                  # crédito/legenda, não tradução
        if lg == "outro":
            continue                                  # transparente p/ a fusão
        if segs and segs[-1]["lang"] == lg and pag - segs[-1]["pag_fim"] <= 2:
            segs[-1]["texto"] += " " + t
            segs[-1]["pag_fim"] = pag
        else:
            segs.append({"lang": lg, "texto": t, "pag_ini": pag, "pag_fim": pag})
    return segs

# ---------- sentenças e alinhamento ----------
def sentencas(txt):
    partes = re.split(r"(?<=[.!?…])\s+", txt)
    return [s.strip() for s in partes if len(s.strip()) >= 3]

def alinhar(ys, ps):
    """Gale-Church simplificado; retorna [(yrl, pt, bead)]."""
    ny, np_ = len(ys), len(ps)
    INF = 1e18
    D = [[INF] * (np_ + 1) for _ in range(ny + 1)]
    B = [[None] * (np_ + 1) for _ in range(ny + 1)]
    D[0][0] = 0.0
    beads = [(1, 1, 0.0), (1, 2, 0.45), (2, 1, 0.45), (2, 2, 0.7),
             (1, 0, 1.7), (0, 1, 1.7)]
    for i in range(ny + 1):
        for j in range(np_ + 1):
            if D[i][j] >= INF:
                continue
            for di, dj, pen in beads:
                ni, nj = i + di, j + dj
                if ni > ny or nj > np_:
                    continue
                if di and dj:
                    ly = sum(len(s) for s in ys[i:ni])
                    lp = sum(len(s) for s in ps[j:nj])
                    c = 2.0 * abs(math.log((ly + 1) / (lp + 1))) + pen
                else:
                    c = pen
                if D[i][j] + c < D[ni][nj]:
                    D[ni][nj] = D[i][j] + c
                    B[ni][nj] = (i, j, di, dj)
    out, i, j = [], ny, np_
    while (i, j) != (0, 0):
        if B[i][j] is None:
            return []                                   # sem caminho
        pi, pj, di, dj = B[i][j]
        if di and dj:
            y = " ".join(ys[pi:pi + di])
            p = " ".join(ps[pj:pj + dj])
            if 0.33 <= (len(y) + 1) / (len(p) + 1) <= 3.0:
                out.append((y, p, f"{di}:{dj}"))
        i, j = pi, pj
    out.reverse()
    return out

def frac_caps(s):
    letras = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letras) / max(1, len(letras))

# ---------- âncoras lexicais (guarda contra deriva do DP em tradução livre) ----------
import unicodedata

def deacc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if not unicodedata.combining(c))

def _carrega_glosas():
    g = defaultdict(set)
    arq = ROOT / "work" / "verbetes_avila.jsonl"
    for e in map(json.loads, open(arq, encoding="utf-8")):
        k = deacc(re.sub(r"\s*\([^)]*\)\s*", "", e["yrl"]).strip())
        for gl in e.get("glosas", [])[:3]:
            for w in re.findall(r"[a-zà-ÿ]{4,}", gl.lower()):
                g[k].add(deacc(w))
    return g

GLOSAS = _carrega_glosas()

def ancoras(y, p):
    """Conteúdo compartilhado verificável entre os lados: empréstimos/nomes,
    dígitos, ou glosa do Ávila de um token yrl presente no pt."""
    ty = {deacc(strip_p(t)) for t in y.split() if len(strip_p(t)) >= 4}
    tp = {deacc(strip_p(t)) for t in p.split() if len(strip_p(t)) >= 4}
    n = len(ty & tp)
    n += len(set(re.findall(r"\d+", y)) & set(re.findall(r"\d+", p)))
    for t in ty:
        if GLOSAS.get(t) and GLOSAS[t] & tp:
            n += 1
    return n

def main():
    pares, st = [], defaultdict(Counter)
    for livro, arq in LIVROS.items():
        doc = fitz.open(DIR / arq)
        segs = segmentos(blocos_livro(doc))
        i = 0
        while i < len(segs):
            s = segs[i]
            if s["lang"] != "yrl" or len(s["texto"]) < 40:
                i += 1
                continue
            j = i + 1
            if j >= len(segs) or segs[j]["lang"] != "pt":
                st[livro]["seg_yrl_sem_pt"] += 1
                i += 1
                continue
            p = segs[j]
            razao = len(p["texto"]) / max(1, len(s["texto"]))
            if razao < 0.35:                  # ruído PT residual: descarta e
                st[livro]["pt_ruido"] += 1    # tenta o próximo p/ o MESMO yrl
                del segs[j]
                continue
            if razao > 3.2:
                st[livro]["seg_razao_ruim"] += 1
                i = j
                continue
            ys, ps = sentencas(s["texto"]), sentencas(p["texto"])
            if min(len(ys), len(ps)) <= 2:
                al = [(s["texto"], p["texto"], "seg")] \
                    if len(s["texto"]) <= 420 else []
                if not al:
                    st[livro]["seg_longo_sem_sentencas"] += 1
            else:
                al = alinhar(ys, ps)
            seg_curto = min(len(ys), len(ps)) <= 4
            for y, pt, bead in al:
                anc = ancoras(y, pt)
                # tradução livre + segmento longo: exige âncora lexical
                if not seg_curto and bead != "seg" and anc == 0:
                    st[livro]["descartado_sem_ancora"] += 1
                    continue
                pares.append({"livro": livro, "pag_yrl": s["pag_ini"],
                              "pag_pt": p["pag_ini"], "yrl": y, "pt": pt,
                              "bead": bead, "anc": anc,
                              "caps": frac_caps(y) > 0.6})
                st[livro][f"bead_{bead}"] += 1
            st[livro]["segmentos_pareados"] += 1
            i = j + 1
    # dedupe exato
    seen, finais = set(), []
    for e in pares:
        k = (e["yrl"].lower(), e["pt"].lower())
        if k in seen:
            continue
        seen.add(k)
        finais.append(e)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for e in finais:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[leetra] {len(finais)} pares únicos -> {OUT}")
    for livro in LIVROS:
        c = st[livro]
        n = sum(v for k, v in c.items() if k.startswith("bead_"))
        print(f"  {livro}: {n} pares | segmentos pareados {c['segmentos_pareados']} | "
              f"yrl sem PT {c['seg_yrl_sem_pt']} | razão ruim {c['seg_razao_ruim']} | "
              f"sem âncora {c['descartado_sem_ancora']} | "
              f"beads {{'1:1': {c['bead_1:1']}, '1:2': {c['bead_1:2']}, "
              f"'2:1': {c['bead_2:1']}, '2:2': {c['bead_2:2']}, 'seg': {c['bead_seg']}}}")
    tok = sum(len(e["yrl"].split()) for e in finais)
    ncaps = sum(1 for e in finais if e["caps"])
    print(f"  tokens yrl ~{tok} | pares em CAPS: {ncaps}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
