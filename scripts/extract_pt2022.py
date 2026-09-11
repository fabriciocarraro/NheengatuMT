# -*- coding: utf-8 -*-
"""
Extrai linhas lógicas do work/CF88_EC128_livro.pdf (Senado, até a EC 128/2022).

Convenções desta edição (inspecionadas):
 - cabeçalho no TOPO (MinionPro-It 8.0): nº de página + seção corrente → remover;
 - notas de rodapé "NE:" em fonte ≤9.5 marcadas com asterisco(s) → remover blocos
   e retirar os asteriscos remanescentes no corpo (com contagem);
 - ordinais em expoente (~6.4): o/a → º/ª; sem referências "(EC nº ...)" inline;
 - soft hyphen (U+00AD) na hifenização; fileiras visuais com fragmentos → fundir.

Saída: work/pt2022_lines.jsonl (mesmo formato de pt_lines.jsonl)
"""
import sys, io, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import fitz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")
LOG = open(os.path.join(WORK, "extract2022_log.txt"), "w", encoding="utf-8")
def log(*a): print(*a, file=LOG)

SPACE_MAP = {0x2002: " ", 0x2003: " ", 0x2004: " ", 0x2005: " ", 0x2009: " ",
             0x00A0: " ", 0x2007: " ", 0x2008: " ", 0x202F: " ", 0x200C: None}

def sort_rows(raw_lines):
    raw_lines.sort(key=lambda r: r["y0"])
    rows = []
    for r in raw_lines:
        if rows and abs(r["y0"] - rows[-1][0]["y0"]) < 3.0:
            rows[-1].append(r)
        else:
            rows.append([r])
    out = []
    for row in rows:
        row.sort(key=lambda r: r["x0"])
        if len(row) > 1:
            merged = dict(row[0])
            merged["text"] = " ".join(x["text"] for x in row)
            merged["text"] = re.sub(r"\s+([,;.:])", r"\1", merged["text"])
            merged["sizes"] = sorted({s for x in row for s in x["sizes"]})
            out.append(merged)
        else:
            out.append(row[0])
    return out

doc = fitz.open(os.path.join(WORK, "CF88_EC128_livro.pdf"))
out = []
n_header = n_footnote = n_ord = n_ast = 0
for pno in range(11, 132):  # TÍTULO I .. "Brasília, 5 de outubro de 1988"
    page_lines = []
    d = doc[pno].get_text("dict")
    blocks = [b for b in d["blocks"] if b["type"] == 0]
    for b in blocks:
        fonts = {(s["font"], round(s["size"], 1)) for l in b["lines"] for s in l["spans"] if s["text"].strip()}
        if not fonts:
            continue
        if b["bbox"][1] < 60 and all(f == ("MinionPro-It", 8.0) for f in fonts):
            n_header += 1
            continue
        if all(f[1] <= 9.5 for f in fonts) and b["bbox"][1] > 60:
            txt = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"])
            n_footnote += 1
            log(f"[pt2022] nota removida p{pno+1}: {txt.strip()[:110]!r}")
            continue
        for l in b["lines"]:
            parts = []
            bold_start = None
            sizes = []
            for s in l["spans"]:
                st = s["text"]
                if not st.strip():
                    parts.append(" ")
                    continue
                sz = round(s["size"], 1)
                if bold_start is None:
                    bold_start = "Bold" in s["font"]
                if sz <= 7.5:
                    stt = st.strip()
                    if re.fullmatch(r"[oa]s?", stt):
                        n_ord += 1
                        parts.append({"o": "º", "os": "ºs", "a": "ª", "as": "ªs"}[stt])
                        continue
                    if re.fullmatch(r"\*+", stt):
                        n_ast += 1
                        continue
                    if re.fullmatch(r"\d+", stt):
                        log(f"[pt2022] ref. numérica removida p{pno+1}: {stt!r}")
                        continue
                    log(f"[pt2022] AVISO span pequeno p{pno+1}: {stt!r}")
                sizes.append(sz)
                parts.append(st)
            text = "".join(parts).translate(SPACE_MAP).replace("‑", "-")
            stars = len(re.findall(r"\*", text))
            if stars:
                n_ast += stars
                text = text.replace("*", "")
            text = re.sub(r" {2,}", " ", text).strip()
            if not text:
                continue
            page_lines.append({"page": pno + 1, "x0": round(l["bbox"][0], 1), "y0": round(l["bbox"][1], 1),
                               "bold_start": bool(bold_start), "sizes": sorted(set(sizes)), "text": text})
    out.extend(sort_rows(page_lines))
doc.close()
log(f"[pt2022] cabeçalhos removidos: {n_header}, notas: {n_footnote}, ordinais: {n_ord}, asteriscos: {n_ast}")

merged, i, n_join = [], 0, 0
while i < len(out):
    cur = dict(out[i])
    while cur["text"].endswith("­") and i + 1 < len(out):
        cur["text"] = cur["text"][:-1] + out[i + 1]["text"]
        n_join += 1
        i += 1
    cur["text"] = cur["text"].replace("­", "")
    merged.append(cur)
    i += 1
log(f"[pt2022] junções por soft hyphen: {n_join}")

with open(os.path.join(WORK, "pt2022_lines.jsonl"), "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"pt2022: {len(merged)} linhas | cabeçalhos {n_header} | notas {n_footnote} | ordinais {n_ord} | asteriscos {n_ast} | soft-hyphens {n_join}")
