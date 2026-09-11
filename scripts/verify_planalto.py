# -*- coding: utf-8 -*-
"""
Etapa 4: verifica cada unidade PT-2016 contra o texto compilado do Planalto.

O compilado mantém redações antigas tachadas (<strike>) seguidas de
"(Redação dada pela Emenda Constitucional nº N...)". Assim:
  - unidade encontrada no texto VIVO  → não mudou desde 2016 → par seguro;
  - encontrada apenas em trecho TACHADO → substituída pela EC N:
      N <= 128 (até 2022, base da tradução NG) → divergência real de conteúdo;
      N >= 129 (2023+)                          → mudou depois da base NG → par seguro;
  - não encontrada → revisão manual (ruído de normalização ou reescrita profunda).

Saída: work/planalto_check.jsonl (id → veredito) e resumo no stdout.
"""
import sys, io, os, json, re, html
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")

raw = open(os.path.join(WORK, "planalto_cf_hist.html"), "rb").read().decode("cp1252", "replace")

# corta no início do ADCT (última ocorrência do título, após o art. 250)
cut = None
for m_adct in re.finditer(r"ATO DAS DISPOSI[ÇC][ÕO]ES CONSTITUCIONAIS", raw, re.I):
    if m_adct.start() > len(raw) * 0.5:
        cut = m_adct.start()
if cut:
    raw = raw[:cut]

# separa segmentos vivos e tachados
parts = re.split(r"(<strike[^>]*>.*?</strike>|<s>.*?</s>|<del[^>]*>.*?</del>)", raw,
                 flags=re.S | re.I)

def to_text(h):
    h = re.sub(r"<[^>]+>", " ", h)
    h = html.unescape(h)
    return h

def norm(t):
    t = t.replace("­", "").replace(" ", " ")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r" ([,;.:!?)])", r"\1", t)
    t = t.replace("( ", "(")
    t = t.lower()
    # grafia pre-Acordo Ortografico usada pelo Planalto em dispositivos antigos
    t = t.replace("qü", "qu").replace("gü", "gu")
    t = t.replace("éia", "eia").replace("éi", "ei").replace("ôo", "oo").replace("êem", "eem")
    t = t.replace("pêlo", "pelo").replace("pára-", "para-")
    return t.strip()

import unicodedata
def fold(t):
    t = re.sub(r"(?<=\d)\s*[oº°ª]s?", "", t.lower())
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if ("a" <= c <= "z") or c.isdigit())

live_parts = []
struck = []   # (texto_tachado_norm, contexto_POSTERIOR — tachado ou vivo)
for i, part in enumerate(parts):
    if re.match(r"<(strike|s\b|del)", part, re.I):
        inner = norm(to_text(part))
        # a EC que SUBSTITUIU este texto está na nota do segmento seguinte
        # (a nota dentro do próprio tachado é a EC de criação do texto antigo)
        nxt = ""
        for j in range(i + 1, min(i + 8, len(parts))):
            nxt += to_text(parts[j])
            if len(nxt) > 600: break
        struck.append((inner, norm(nxt)[:600]))
    else:
        live_parts.append(to_text(part))
live = norm(" ".join(live_parts))
print(f"texto vivo: {len(live)} chars | segmentos tachados: {len(struck)}")

RE_EC_NOTE = re.compile(r"\((?:reda[çc][ãa]o dada|inclu[íi]d[oa]|revogad[oa]|acrescid[oa])[^)]*?"
                        r"emenda constitucional n[ºo°]?\s*(\d+)", re.I)

live_fold = fold(live)
struck_fold = [fold(st) for st, _ in struck]

pairs = [json.loads(l) for l in open(os.path.join(WORK, "pairs.jsonl"), encoding="utf-8")]
results = []
counts = {}
for p in pairs:
    if p["type"] in ("titulo", "capitulo", "secao", "subsecao", "closing"):
        continue
    t = norm(p["pt"])
    r = {"id": p["id"]}
    if not t:
        r["veredito"] = "sem_texto"
    elif len(t) < 15:
        r["veredito"] = "curto_demais"
    elif t in live:
        r["veredito"] = "vivo"
    else:
        found_struck = False
        ec = None
        for st, ctx in struck:
            if t in st:
                found_struck = True
                m = RE_EC_NOTE.search(ctx)
                if m:
                    ec = int(m.group(1))
                break
        if found_struck and ec is not None:
            r["ec"] = ec
            r["veredito"] = "tachado_ec_ate_128" if ec <= 128 else "tachado_ec_129_mais"
        elif found_struck:
            r["veredito"] = "tachado_ec_desconhecida"
        else:
            # passe 2: só letras/dígitos, sem acentos (mata variações de pontuação)
            ft = fold(p["pt"])
            if ft and ft in live_fold:
                r["veredito"] = "vivo_cosmetico"
            else:
                fec = None
                fstruck = False
                for k, (st, ctx) in enumerate(struck):
                    if ft and ft in struck_fold[k]:
                        fstruck = True
                        m = RE_EC_NOTE.search(ctx)
                        if m: fec = int(m.group(1))
                        break
                if fstruck and fec is not None:
                    r["ec"] = fec
                    r["veredito"] = "tachado_ec_ate_128" if fec <= 128 else "tachado_ec_129_mais"
                elif fstruck:
                    r["veredito"] = "tachado_ec_desconhecida"
                else:
                    r["veredito"] = "nao_localizado_verificar"
    counts[r["veredito"]] = counts.get(r["veredito"], 0) + 1
    results.append(r)

with open(os.path.join(WORK, "planalto_check.jsonl"), "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("vereditos:", counts)
print("\n--- tachado com EC <= 128 (divergência real de conteúdo) ---")
for r in results:
    if r["veredito"] == "tachado_ec_ate_128":
        print(f"  {r['id']:32} EC {r['ec']}")
print("\n--- não encontrados (revisar normalização/reescrita) ---")
n = 0
for r in results:
    if r["veredito"] == "nao_localizado_verificar":
        n += 1
        if n <= 40: print(f"  {r['id']}")
print(f"  (total: {n})")
