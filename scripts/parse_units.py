# -*- coding: utf-8 -*-
"""
Etapa 2: converte as linhas em unidades estruturais validadas por sequência.

Máquina de estados: um marcador só é aceito se couber na sequência esperada.
 - Art. n+1 (ou n com sufixo -A/-B...; salto de até +5 com aviso; duplicata vira .bis)
 - § k+1 (primeiro § tem de ser 1; salto com aviso; sufixo -A/-B/-C; referências
   no meio do texto são rejeitadas pela sequência + guardas de pontuação)
 - inciso romano seguinte (sufixo -A; salto com aviso; romano malformado só é
   aceito por distância de edição quando NÃO é um romano canônico válido)
 - alínea seguinte (salto com aviso)
Candidatos rejeitados viram continuação de texto e são registrados no log.

Saída: work/pt_units.jsonl, work/ng_units.jsonl, work/parse_log.txt
"""
import sys, io, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")
LOG = open(os.path.join(WORK, "parse_log.txt"), "w", encoding="utf-8")
def log(*a):
    print(*a, file=LOG)

ROMAN_VALS = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
def int_to_roman(n):
    out = []
    for v, sym in [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
                   (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]:
        while n >= v: out.append(sym); n -= v
    return "".join(out)

def roman_to_int(s):
    """Valor do romano; None se caracteres inválidos; negativo se não canônico."""
    s = s.upper()
    if not s or any(c not in ROMAN_VALS for c in s): return None
    total = 0
    for i, c in enumerate(s):
        v = ROMAN_VALS[c]
        if i + 1 < len(s) and ROMAN_VALS[s[i+1]] > v: total -= v
        else: total += v
    return total if int_to_roman(total) == s else -total

def editdist(a, b):
    if abs(len(a)-len(b)) > 3: return 99
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

BAD_REST = re.compile(r"^[,;.)–—-]|^(do|da|dos|das|deste|desta|e|ou|até)\s", re.U)

# Patches cirúrgicos em linhas do NG: erros do próprio livro, verificados
# manualmente contra o PDF (cada um é documentado no relatório). O script
# FALHA se algum patch não encontrar exatamente uma linha-alvo.
NG_PATCHES = [
    # art. 93, VIII-A impresso como "VIII" (2ª ocorrência) — conteúdo = VIII-A do PT
    {"page": 63, "startswith": "VIII - Kuá tirikasá yupurãdu",
     "replace_prefix": ("VIII - ", "VIII-A – "), "why": "art93 VIII-A rotulado 'VIII' no livro"},
    # art. 109, V-A impresso como "V-" — conteúdo = V-A do PT (direitos humanos, § 5º)
    {"page": 77, "startswith": "V- manungaára waá kuntari sá",
     "replace_prefix": ("V- ", "V-A – "), "why": "art109 V-A rotulado 'V' no livro"},
    # art. 128, § 5º, II: alínea b) impressa como "c)" (conteúdo = b: exercer a advocacia)
    {"page": 84, "startswith": "c) muyãsá kuá adivukasia",
     "replace_prefix": ("c) ", "b) "), "why": "art128 §5 II alínea b rotulada 'c)' no livro"},
    # art. 194: o livro traz o PU traduzido E, depois dele, uma linha em português
    # não traduzida ("I - universalidade..."), removida para não virar falso inciso
    {"page": 129, "startswith": "I - universalidade da cobertura e do atendimento:",
     "delete": True,
     "why": "art194: fragmento em português não traduzido removido (após o PU traduzido)"},
]

def apply_ng_patches(lines):
    applied = []
    out = []
    for ln in lines:
        matched = None
        for p in NG_PATCHES:
            if ln["page"] == p["page"] and ln["text"].startswith(p["startswith"]):
                matched = p
                break
        if matched:
            applied.append(matched["why"])
            log(f"[ng] PATCH p{matched['page']}: {matched['why']}")
            if matched.get("delete"):
                pass
            else:
                old, new = matched["replace_prefix"]
                nl = dict(ln); nl["text"] = new + ln["text"][len(old):]
                out.append(nl)
        else:
            out.append(ln)
    if len(applied) != len(NG_PATCHES):
        missing = [p["why"] for p in NG_PATCHES if p["why"] not in applied]
        raise RuntimeError(f"Patches não aplicados: {missing}")
    return out

# ------------------------------------------------------------- marcadores -----
RE_ART   = re.compile(r"^Art\s*\.{0,2}\s*(\d+)\s*(?:-\s*([A-Z])\b)?\s*º?\s*\.?\s*(.*)$", re.S)
RE_PAR   = re.compile(r"^§\s*(\d+)\s*º?\s*\.?\s*(?:-\s*([A-Z])(?=\s|\.)\s*\.?)?\s*(.*)$", re.S)
RE_PU_PT = re.compile(r"^Parágrafo único\s*[.:]?\s*(.*)$", re.S)
# Variantes reais do marcador de "Parágrafo único" no livro NG (inventariadas):
#   Ũbeusá yepéyũtu. / Yepeyũtu. / Yepéyũ. / Yepéyũ:   |  Ũbeusawá yepẽtu. / Yepẽtu.
#   Ũbeusa Yepẽtu.  |  ūbesáwa yepētu. (macrons)  |  Yũbeusa Yepẽtu.
#   Ũbeu sawá mirĩ. / Ũbeu sawá xinga mirĩ:  |  Yepéyũwaátépinimasá upé. (art. 44)
PU_NG_CORE = (r"(?:[Yy]?[Ũũū]be[uú]?\s?s[aá]\w{0,3}\s+[Yy]ep(?:[eé]?yũtu|[eé]?yũ|[ẽē]tu)"
              r"|[Ũũū]beu\s+sawá\s+(?:xinga\s+)?mirĩ"
              r"|Yepéyũwaátépinimasá\s+upé)")
RE_PU_NG = re.compile(r"^" + PU_NG_CORE + r"\s*[.:]?\s*(.*)$", re.S)
RE_INC_S = re.compile(r"^([IVXLCDM]{1,10})\s*-\s*([A-Z])(?=[\s.º–—-])\s*[.–—-]?\s*(.*)$", re.S)
RE_INC   = re.compile(r"^([IVXLCDM]{1,10})\s*[–—-]\s*(.*)$", re.S)
RE_ALI   = re.compile(r"^([a-z])\)\s*(.*)$", re.S)
RE_CLOSE = re.compile(r"^Brasília,\s*5 de outubro de 1988")
HEAD_PT  = [(re.compile(r"^TÍTULO\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "titulo"),
            (re.compile(r"^CAPÍTULO\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "capitulo"),
            (re.compile(r"^SEÇÃO\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "secao"),
            (re.compile(r"^SUBSEÇÃO\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "subsecao")]
HEAD_NG  = [(re.compile(r"^SESEWÁRA\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "titulo"),
            (re.compile(r"^ŨBE+U?S[AÁ]+RA\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "capitulo"),
            (re.compile(r"^SESÃU\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "secao"),
            (re.compile(r"^SUBSESÃU\s+([IVXLC]+)\s*[–—-]?\s*(.*)$"), "subsecao")]

def clean_rest(r):
    return re.sub(r"^[\s–—-]+", "", r).strip()

def parse(lang):
    lines = [json.loads(l) for l in open(os.path.join(WORK, f"{lang}_lines.jsonl"), encoding="utf-8")]
    if lang == "ng":
        lines = apply_ng_patches(lines)
    HEADS = HEAD_PT if lang.startswith("pt") else HEAD_NG
    RE_PU = RE_PU_PT if lang.startswith("pt") else RE_PU_NG

    units = []
    cur = None
    art_num, art_suf = 0, ""
    para, para_suf = 0, ""          # (nº do §, sufixo letra); -1 = parágrafo único
    container = "caput"
    last_inc, last_inc_suf = 0, ""
    last_ali = ""
    in_heading = False
    hier = {"titulo": None, "capitulo": None, "secao": None, "subsecao": None}

    def aid():
        return f"art{art_num}{('-' + art_suf) if art_suf else ''}"

    def base_id():
        if container == "caput": return f"{aid()}.caput"
        if para == -1: return f"{aid()}.pu"
        return f"{aid()}.par{para}{para_suf}"

    def start(u):
        nonlocal cur
        if cur: units.append(cur)
        cur = u

    for ln in lines:
        t = ln["text"].strip()
        pg = ln["page"]
        if not t: continue

        # ---------- cabeçalhos estruturais ----------
        hit = None
        for rex, level in HEADS:
            m = rex.match(t)
            if m: hit = (level, m); break
        if hit:
            level, m = hit
            val = roman_to_int(m.group(1))
            if val is None or val <= 0:
                log(f"[{lang}] AVISO p{pg}: numeração de {level} não canônica: {m.group(1)!r}")
                val = abs(val) if val else 0
            order = ["titulo", "capitulo", "secao", "subsecao"]
            idx = order.index(level)
            hier[level] = val
            for lower in order[idx+1:]: hier[lower] = None
            hid = ".".join(f"{lv[:3]}{hier[lv]}" for lv in order if hier[lv])
            start({"lang": lang, "type": level, "id": f"h:{hid}", "art": None,
                   "text": clean_rest(m.group(2)), "pages": [pg]})
            in_heading = True
            continue

        # ---------- fecho ----------
        if RE_CLOSE.match(t):
            start({"lang": lang, "type": "closing", "id": "closing", "art": None,
                   "text": t, "pages": [pg]})
            in_heading = False
            continue

        # ---------- artigo ----------
        m = RE_ART.match(t)
        if m and (t.startswith("Art.") or t.startswith("Art ")):
            n, suf, rest = int(m.group(1)), m.group(2) or "", m.group(3)
            note = ""
            accept = False
            if n == art_num + 1 and suf == "":
                accept = True
            elif n == art_num and suf and ((art_suf == "" and suf == "A") or
                                           (art_suf and ord(suf) == ord(art_suf) + 1)):
                accept = True
            elif suf == "" and art_num + 2 <= n <= art_num + 5 and not BAD_REST.match(rest):
                accept, note = True, f" (LACUNA: art(s). {art_num+1}..{n-1} ausente(s))"
            elif n == art_num and suf == art_suf == "":
                # artigo duplicado (ex.: art. 39 impresso duas vezes no NG)
                log(f"[{lang}] p{pg}: Art. {n} DUPLICADO — emitido como .bis")
                start({"lang": lang, "type": "caput_bis", "id": f"{aid()}.caput.bis", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(rest), "pages": [pg]})
                in_heading = False
                continue
            if accept:
                art_num, art_suf = n, suf
                para, para_suf, container, last_inc, last_inc_suf, last_ali = 0, "", "caput", 0, "", ""
                if note: log(f"[{lang}] p{pg}: Art. {n}{note}")
                start({"lang": lang, "type": "caput", "id": f"{aid()}.caput", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(rest), "pages": [pg]})
                in_heading = False
                continue
            else:
                log(f"[{lang}] REJEITADO p{pg}: candidato a artigo {n}{'-'+suf if suf else ''} "
                    f"(esperado {art_num+1}); tratado como continuação: {t[:70]!r}")

        # ---------- parágrafo ----------
        m = RE_PAR.match(t)
        if m and not in_heading:
            k, ksuf, rest = int(m.group(1)), m.group(2) or "", m.group(3)
            rest = re.sub(r"^[–—-]\s+", "", rest)  # "§ 2º - Texto" → separador, não referência
            accept, note = False, ""
            if ksuf and k == para and ((para_suf == "" and ksuf == "A") or
                                       (para_suf and ord(ksuf) == ord(para_suf) + 1)):
                accept = True
            elif not ksuf and k == para + 1 and not BAD_REST.match(rest):
                accept = True
            elif not ksuf and para == 0 and k == 1 and not BAD_REST.match(rest):
                accept = True
            elif not ksuf and para > 0 and para + 1 < k <= para + 6 and not BAD_REST.match(rest):
                accept, note = True, f" (LACUNA: §§ {para+1}..{k-1} ausentes)"
            if accept:
                para, para_suf, container, last_inc, last_inc_suf, last_ali = k, ksuf, "par", 0, "", ""
                if note: log(f"[{lang}] p{pg} {aid()}: § {k}{note}")
                start({"lang": lang, "type": "paragrafo", "id": f"{aid()}.par{k}{ksuf}", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(rest), "pages": [pg]})
                continue
            else:
                log(f"[{lang}] REJEITADO p{pg}: candidato a § {k}{'-'+ksuf if ksuf else ''} "
                    f"(último {para}{para_suf}, {aid()}): {t[:70]!r}")

        m = RE_PU.match(t)
        if m and not in_heading:
            if para == 0:
                para, para_suf, container, last_inc, last_inc_suf, last_ali = -1, "", "par", 0, "", ""
                start({"lang": lang, "type": "paragrafo", "id": f"{aid()}.pu", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(m.group(1)), "pages": [pg]})
                continue
            else:
                log(f"[{lang}] REJEITADO p{pg}: 'Parágrafo único' após § {para} ({aid()})")

        # ---------- inciso ----------
        m = RE_INC_S.match(t)
        if m and not in_heading:
            rom, isuf, rest = m.group(1), m.group(2), m.group(3)
        else:
            m = RE_INC.match(t)
            if m:
                rom, isuf, rest = m.group(1), "", m.group(2)
        if m and not in_heading:
            val = roman_to_int(rom)
            expected = last_inc + 1
            accept, note = None, ""
            if isuf and val == last_inc and val > 0 and ((last_inc_suf == "" and isuf == "A") or
                                                         (last_inc_suf and ord(isuf) == ord(last_inc_suf) + 1)):
                accept = (val, isuf)
            elif not isuf and val == expected:
                accept = (expected, "")
            elif not isuf and val is not None and val > 0 and last_inc > 0 \
                    and expected < val <= expected + 4 and not BAD_REST.match(rest):
                accept, note = (val, ""), f" (LACUNA: incisos {int_to_roman(expected)}..{int_to_roman(val-1)} ausentes)"
            elif (val is None or val < 0) and editdist(rom, int_to_roman(expected)) <= 2 and len(rom) >= 3:
                accept, note = (expected, ""), f" (TIPO: {rom!r} lido como {int_to_roman(expected)!r})"
            if accept:
                last_inc, last_inc_suf = accept
                last_ali = ""
                if note: log(f"[{lang}] p{pg} {base_id()}: inciso{note}")
                start({"lang": lang, "type": "inciso",
                       "id": f"{base_id()}.inc{last_inc}{last_inc_suf}", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(rest), "pages": [pg], "marker_raw": rom + ("-"+isuf if isuf else "")})
                continue
            else:
                log(f"[{lang}] REJEITADO p{pg}: candidato a inciso {rom + ('-'+isuf if isuf else '')!r} "
                    f"(esperado {int_to_roman(expected)}, {base_id()}): {t[:70]!r}")

        # ---------- alínea ----------
        m = RE_ALI.match(t)
        if m and not in_heading and last_inc > 0:
            c, rest = m.group(1), m.group(2)
            expected = chr(ord(last_ali) + 1) if last_ali else "a"
            accept, note = None, ""
            if c == expected:
                accept = c
            elif last_ali and expected < c <= chr(ord(expected) + 3) and not BAD_REST.match(rest):
                accept, note = c, f" (LACUNA: alíneas {expected}..{chr(ord(c)-1)} ausentes)"
            if accept:
                last_ali = accept
                if note: log(f"[{lang}] p{pg} {base_id()}.inc{last_inc}: alínea{note}")
                start({"lang": lang, "type": "alinea",
                       "id": f"{base_id()}.inc{last_inc}{last_inc_suf}.{accept}", "art": aid(),
                       "hier": dict(hier), "text": clean_rest(rest), "pages": [pg]})
                continue
            else:
                log(f"[{lang}] nota p{pg}: possível alínea {c!r} fora de sequência (última {last_ali!r}): {t[:60]!r}")

        # ---------- continuação ----------
        if cur is None:
            log(f"[{lang}] AVISO p{pg}: linha órfã antes do primeiro marcador: {t[:70]!r}")
            continue
        cur["text"] = (cur["text"] + " " + t).strip() if cur["text"] else t
        if pg not in cur["pages"]: cur["pages"].append(pg)

    if cur: units.append(cur)
    return units

import sys as _sys
for lang in (_sys.argv[1:] or ["pt", "ng"]):
    units = parse(lang)
    with open(os.path.join(WORK, f"{lang}_units.jsonl"), "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(u["type"] for u in units)
    arts = [u for u in units if u["type"] == "caput"]
    print(f"{lang.upper()}: {len(units)} unidades | {dict(c)} | artigos: {len(arts)} "
          f"(último: {arts[-1]['art'] if arts else '-'})")
LOG.close()
print("log em work/parse_log.txt")
