# -*- coding: utf-8 -*-
"""
Etapa 1: extrai linhas lógicas limpas dos dois PDFs.

Saída: work/pt_lines.jsonl e work/ng_lines.jsonl
  cada linha: {"page": n, "x0": f, "y0": f, "bold_start": bool, "sizes": [..], "text": "..."}
Log:   work/extract_log.txt (cabeçalhos/rodapés/notas removidos, junções de hífen, spans descartados)
"""
import sys, io, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import fitz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")
os.makedirs(WORK, exist_ok=True)

LOG = open(os.path.join(WORK, "extract_log.txt"), "w", encoding="utf-8")
def log(*a):
    print(*a, file=LOG)

SPACE_MAP = {0x2002: " ", 0x2003: " ", 0x2004: " ", 0x2005: " ", 0x2009: " ",
             0x00A0: " ", 0x2007: " ", 0x2008: " ", 0x202F: " ", 0x200C: None}

def norm_spaces(t):
    t = t.translate(SPACE_MAP)
    return t

def sort_rows(raw_lines):
    """Ordena linhas da página em fileiras visuais: linhas cujo y0 difere <3pt
    formam uma fileira e são ordenadas por x0 (fragmentos da mesma linha visual
    separados por expoentes de nota caem em blocos distintos com y quase igual)."""
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
        # fragmentos da mesma fileira são fundidos numa única linha lógica
        if len(row) > 1:
            merged = dict(row[0])
            merged["text"] = " ".join(x["text"] for x in row)
            merged["text"] = re.sub(r"\s+([,;.:])", r"\1", merged["text"])
            merged["sizes"] = sorted({s for x in row for s in x["sizes"]})
            out.append(merged)
        else:
            out.append(row[0])
    return out

# ---------------------------------------------------------------- PT ----------
def extract_pt():
    doc = fitz.open(os.path.join(BASE, "Constituicao-PT-BR.pdf"))
    out = []
    n_footer = n_footnote = n_supdig = n_ord = 0
    for pno in range(1, 127):  # índices 1..126 = conteúdo
        page = doc[pno]
        page_lines = []
        d = page.get_text("dict")
        blocks = [b for b in d["blocks"] if b["type"] == 0]
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        for b in blocks:
            sizes_all = [round(s["size"], 1) for l in b["lines"] for s in l["spans"] if s["text"].strip()]
            fonts_all = {(s["font"], round(s["size"], 1)) for l in b["lines"] for s in l["spans"] if s["text"].strip()}
            if not sizes_all:
                continue
            # rodapé: itálico 8.0 no pé da página
            if b["bbox"][1] > 650 and all(f == ("MinionPro-It", 8.0) for f in fonts_all):
                n_footer += 1
                continue
            # nota de rodapé editorial ("NE: ..."): fonte 9.0 (com eventuais dígitos 6.x)
            if all(sz <= 9.5 for sz in sizes_all) and b["bbox"][1] > 550:
                txt = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"])
                n_footnote += 1
                log(f"[PT] footnote removida p{pno+1}: {norm_spaces(txt).strip()[:120]!r}")
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
                        if re.fullmatch(r"\d+", stt):
                            n_supdig += 1
                            log(f"[PT] ref. nota removida p{pno+1}: {stt!r} (contexto: {''.join(parts)[-40:]!r})")
                            continue
                        if re.fullmatch(r"[oa]s?", stt):
                            n_ord += 1
                            parts.append({"o": "º", "os": "ºs", "a": "ª", "as": "ªs"}[stt])
                            continue
                        log(f"[PT] AVISO span pequeno não tratado p{pno+1}: {stt!r}")
                    sizes.append(sz)
                    parts.append(st)
                text = norm_spaces("".join(parts))
                text = re.sub(r" {2,}", " ", text).strip()
                if not text:
                    continue
                page_lines.append({"page": pno + 1, "x0": round(l["bbox"][0], 1), "y0": round(l["bbox"][1], 1),
                                   "bold_start": bool(bold_start), "sizes": sorted(set(sizes)), "text": text})
        out.extend(sort_rows(page_lines))
    doc.close()
    log(f"[PT] rodapés removidos: {n_footer}, notas: {n_footnote}, refs. de nota: {n_supdig}, ordinais convertidos: {n_ord}")
    # junção de soft hyphen
    merged, i, n_join = [], 0, 0
    while i < len(out):
        cur = dict(out[i])
        while cur["text"].endswith("­") and i + 1 < len(out):
            nxt = out[i + 1]
            cur["text"] = cur["text"][:-1] + nxt["text"]
            n_join += 1
            i += 1
        cur["text"] = cur["text"].replace("­", "")  # sobras internas
        merged.append(cur)
        i += 1
    log(f"[PT] junções por soft hyphen: {n_join}")
    return merged

# ---------------------------------------------------------------- NG ----------
MARKER_RE = re.compile(r"^(Art\.|§\s*\d|[IVXLCDM]{1,8}[a-z]?\s*[–-]|[a-z]\)\s|SESEWÁRA|ŨBE|SESÃU|SUBSESÃU|Ũbeusá\s)", re.U)

def extract_ng():
    doc = fitz.open(os.path.join(BASE, "Constituicao-Nheengatu.pdf"))
    out = []
    n_header = 0
    for pno in range(1, 153):  # índices 1..152 = conteúdo
        page = doc[pno]
        page_lines = []
        d = page.get_text("dict")
        blocks = [b for b in d["blocks"] if b["type"] == 0]
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        for b in blocks:
            fonts_all = {(s["font"], round(s["size"], 1)) for l in b["lines"] for s in l["spans"] if s["text"].strip()}
            if not fonts_all:
                continue
            # cabeçalho: itálico 9.0 no topo (nº de página + título corrente)
            if b["bbox"][1] < 55 and all(f[0] == "MinionPro-It" and abs(f[1] - 9.0) < 0.3 for f in fonts_all):
                n_header += 1
                txt = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"]).strip()
                if "Mundu" not in txt and not re.match(r"^\d+", txt):
                    log(f"[NG] AVISO: cabeçalho atípico removido p{pno+1}: {txt[:80]!r}")
                continue
            for l in b["lines"]:
                parts = []
                bold_start = None
                sizes = []
                fonts = []
                for s in l["spans"]:
                    st = s["text"]
                    if not st.strip():
                        parts.append(" ")
                        continue
                    if bold_start is None:
                        bold_start = "Bold" in s["font"]
                    sizes.append(round(s["size"], 1))
                    fonts.append(s["font"])
                    parts.append(st)
                text = norm_spaces("".join(parts))
                text = re.sub(r" {2,}", " ", text).strip()
                if not text:
                    continue
                page_lines.append({"page": pno + 1, "x0": round(l["bbox"][0], 1), "y0": round(l["bbox"][1], 1),
                                   "bold_start": bool(bold_start), "sizes": sorted(set(sizes)),
                                   "fonts": sorted(set(fonts)), "text": text})
        out.extend(sort_rows(page_lines))
    doc.close()
    log(f"[NG] cabeçalhos removidos: {n_header} (esperado: 152)")
    # junção de linhas terminadas em hífen lexical (mantém o hífen)
    merged, i, n_join = [], 0, 0
    while i < len(out):
        cur = dict(out[i])
        while (re.search(r"[^\W\d_]-$", cur["text"], re.U)
               and i + 1 < len(out)
               and not MARKER_RE.match(out[i + 1]["text"])):
            nxt = out[i + 1]
            log(f"[NG] junção hífen p{cur['page']}: ...{cur['text'][-25:]!r} + {nxt['text'][:25]!r}")
            cur["text"] = cur["text"] + nxt["text"]
            n_join += 1
            i += 1
        merged.append(cur)
        i += 1
    log(f"[NG] junções por hífen: {n_join} (candidatas vistas na inspeção: 60)")
    return merged

pt = extract_pt()
ng = extract_ng()

with open(os.path.join(WORK, "pt_lines.jsonl"), "w", encoding="utf-8") as f:
    for r in pt:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(WORK, "ng_lines.jsonl"), "w", encoding="utf-8") as f:
    for r in ng:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"PT: {len(pt)} linhas | NG: {len(ng)} linhas")
LOG.close()
print("log em work/extract_log.txt")
