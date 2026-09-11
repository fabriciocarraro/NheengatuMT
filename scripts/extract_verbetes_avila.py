# -*- coding: utf-8 -*-
"""Extrai os VERBETES (headword -> glosas PT) do dicionário de Ávila (2021).

Chave tipográfica (medida no PDF):
  - headword: fonte CharisSIL-Bold, tamanho ~11.7 (o corpo é 9.7)
  - classe gramatical: TimesNewRomanPS-ItalicMT, "(v. tr.)" etc.
  - glosa PT: TimesNewRomanPSMT após a classe, até o ':' que abre exemplos
  - remissiva: "var. de" + alvo em CharisSIL-Bold 9.7
  - aparato histórico: blocos iniciados por '■' (ignorados)
  - sentidos numerados: "1.", "2." em romano

Saída: work/verbetes_avila.jsonl
  {"yrl": headword, "classe": ..., "glosas": [...], "var_de": ... | null,
   "pagina": N}
Fonte: AVILA, M. T. Proposta de dicionário nheengatu-português. Tese (USP),
2021. doi:10.11606/T.8.2021.tde-10012022-201925 — citação obrigatória.
"""
import fitz, json, re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "work" / "novas_fontes" / "avila_2021_dicionario.pdf"
OUT = ROOT / "work" / "verbetes_avila.jsonl"

HEAD_FONT = "CharisSIL-Bold"
HEAD_SIZE_MIN = 11.0
ROMAN_FONTS = ("TimesNewRomanPSMT",)
ITALIC_ROMAN = ("TimesNewRomanPS-ItalicMT",)

def is_headword_span(s):
    if s["font"] != HEAD_FONT or s["size"] < HEAD_SIZE_MIN:
        return False
    t = s["text"].strip()
    if not t or len(t) > 40:
        return False
    return bool(re.match(r"^[a-zà-ÿãẽĩũỹ]", t))

def spans(doc, page_idx):
    d = doc[page_idx].get_text("dict")
    out = []
    for b in d["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    out.append(s)
    return out

def clean_gloss_raw(raw):
    """Remove aparato: citações (Autor, pág.), blocos ●/◆/■/{, colchetes."""
    raw = re.split(r"[●◆■{]|//", raw)[0]
    raw = re.sub(r"\((?=[^)]*\d)[^()]*\)?", "", raw)          # parênteses com dígitos = citação
    raw = re.sub(r"\((?:var|hist|lit|cf|v\.\s?tb)\.[^)]*\)?", "", raw)
    raw = re.sub(r"\[[^\]]*(\]|$)", "", raw)                   # colchetes, incl. truncados
    return re.sub(r"\s+", " ", raw).strip(" ;,.-–—")

def parse_entry(head, body, page):
    """body = lista de spans até o próximo headword."""
    classe = None
    var_de = None
    ver = None
    glosas = []
    # remissiva: 'var. de' (romano) seguido de bold pequeno
    for i, s in enumerate(body):
        if s["font"] in ROMAN_FONTS and re.search(r"\bvar\.\s*de\s*$", s["text"].strip()):
            for s2 in body[i + 1:i + 3]:
                if s2["font"] == HEAD_FONT and s2["size"] < HEAD_SIZE_MIN:
                    var_de = s2["text"].strip()
                    break
            break
    # remissiva 'v.:' (itálico) seguido de bold pequeno — "veja"
    for i, s in enumerate(body):
        if s["font"] in ITALIC_ROMAN and s["text"].strip() in ("v.:", "v."):
            for s2 in body[i + 1:i + 3]:
                if s2["font"] == HEAD_FONT and s2["size"] < HEAD_SIZE_MIN:
                    ver = re.sub(r"\s*\([^)]*\)\s*$", "", s2["text"].strip())
                    break
            break
    # classe: primeiro itálico-romano com parênteses
    for s in body:
        m = re.match(r"^\((.{1,25}?)\)", s["text"].strip()) if s["font"] in ITALIC_ROMAN else None
        if m:
            classe = m.group(1)
            break
    # glosa: concatenar romanos APÓS a classe, parar no 1º ':' (abre exemplos),
    # em '■' (aparato) ou num novo sentido já capturado
    started = classe is None  # sem classe, começa no 1º romano
    buf = []
    for s in body:
        t = s["text"]
        if s["font"] in ITALIC_ROMAN and classe and f"({classe})" in t.replace(" ", " "):
            started = True
            continue
        if not started:
            continue
        if s["font"] in ROMAN_FONTS:
            if "■" in t:
                break
            if ":" in t:
                buf.append(t.split(":", 1)[0])
                break
            buf.append(t)
        elif s["font"].startswith("CharisSIL") and buf:
            break  # começou material yrl (exemplo) sem ':' explícito
    glosa_raw = clean_gloss_raw(" ".join(buf))
    glosa_raw = re.sub(r"^\d\.\s*", "", glosa_raw)
    if glosa_raw:
        for g in re.split(r"\s*;\s*", glosa_raw):
            g = g.strip(" ,.()")
            # descarta resíduo de citação (contém dígitos), fragmentos e remissivas
            if (g and len(g) >= 2 and not any(ch.isdigit() for ch in g)
                    and g.count("(") == g.count(")")
                    and not g.lower().startswith(("var. de", "reg. hist", "v.:"))):
                glosas.append(g)
    return {"yrl": head, "classe": classe, "glosas": glosas[:6],
            "var_de": var_de, "ver": ver, "pagina": page + 1}

def main():
    doc = fitz.open(PDF)
    entries = []
    carry_head, carry_body = None, []
    for p in range(len(doc)):
        ss = spans(doc, p)
        if not any(is_headword_span(s) for s in ss):
            # página sem headword: se estamos no meio de um verbete, acumula
            if carry_head is not None:
                carry_body.extend(ss)
            continue
        for s in ss:
            if is_headword_span(s):
                if carry_head is not None:
                    entries.append(parse_entry(carry_head, carry_body, p))
                # headwords quebrados em spans consecutivos 11.7-bold: emenda
                if entries and not carry_body and carry_head is None and entries[-1]["yrl"].endswith("-"):
                    entries[-1]["yrl"] += s["text"].strip()
                    carry_head, carry_body = None, []
                    continue
                carry_head, carry_body = s["text"].strip(), []
            else:
                if carry_head is not None:
                    carry_body.append(s)
    if carry_head is not None:
        entries.append(parse_entry(carry_head, carry_body, len(doc) - 1))

    # dedupe headwords idênticos consecutivos (quebra de página repete às vezes)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    com_glosa = sum(1 for e in entries if e["glosas"])
    com_var = sum(1 for e in entries if e["var_de"])
    print(f"[verbetes] {len(entries)} entradas -> {OUT}")
    print(f"  com glosa PT: {com_glosa} | remissivas var_de: {com_var} | "
          f"com classe: {sum(1 for e in entries if e['classe'])}")

if __name__ == "__main__":
    main()
