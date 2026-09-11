# -*- coding: utf-8 -*-
"""
Etapa 3: alinha unidades PT ↔ NG, valida pares e classifica divergências.

- Remapeia erro do livro NG: §§ 1º-3º impressos sob o art. 67 pertencem ao art. 68
  (caput do 68 foi omitido pelo livro; conteúdo conferido manualmente).
- Normaliza o texto NG (variantes de glifo → ortografia consistente); o texto
  bruto é preservado em yrl_raw.
- Remove do PT as referências editoriais "(EC nº ...)" (a edição do Senado as
  acrescenta; não são texto constitucional e não existem no NG); ficam em ec_refs.
- Valida cada par: texto vazio, dígitos citados, razão de comprimento.
Saída: work/pairs.jsonl, work/align_log.txt
"""
import sys, io, os, json, re, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")
LOG = open(os.path.join(WORK, "align_log.txt"), "w", encoding="utf-8")
def log(*a):
    print(*a, file=LOG)

def load(lang):
    return [json.loads(l) for l in open(os.path.join(WORK, f"{lang}_units.jsonl"), encoding="utf-8")]

import sys as _sys
PTL = _sys.argv[1] if len(_sys.argv) > 1 else "pt"
pt_units, ng_units = load(PTL), load("ng")
log(f"[base] lado PT: {PTL}")

# ---------------------------------------------------- remap art. 67→68 (NG) ---
ng_ids = {u["id"] for u in ng_units}
assert not any(i.startswith("art68.") for i in ng_ids), "NG art68 não deveria existir antes do remap"
pt_ids_tmp = {u["id"] for u in pt_units}
assert not any(i.startswith("art67.par") for i in pt_ids_tmp), "PT art67 não deveria ter §§"
assert {"art68.par1", "art68.par2", "art68.par3"} <= pt_ids_tmp, "PT art68 deveria ter §§ 1-3"
n_remap = 0
for u in ng_units:
    m = re.match(r"^art67\.par([123])(\.inc\d+.*)?$", u["id"])
    if m:
        new_id = f"art68.par{m.group(1)}{m.group(2) or ''}"
        u["remapped_from"] = u["id"]
        u["id"] = new_id
        u["art"] = "art68"
        n_remap += 1
assert n_remap == 6, f"esperava remapear 6 unidades (3 §§ + 3 incisos), remapeou {n_remap}"
log(f"[remap] NG art67.par* → art68.par* ({n_remap} unidades; caput do art. 68 omitido pelo livro)")

# ------------------------------- PUs embutidos no fim da unidade anterior -----
# O livro NG frequentemente emenda o "Parágrafo único" na mesma linha do texto
# anterior, com grafias variantes. Divide a unidade em duas no marcador.
PU_MID = re.compile(r"(?<=[.;:”])\s+(?:[Yy]?[Ũũū]be[uú]?\s?s[aá]\w{0,3}\s+"
                    r"[Yy]ep(?:[eé]?yũtu|[eé]?yũ|[ẽē]tu)"
                    r"|[Ũũū]beu\s+sawá\s+(?:xinga\s+)?mirĩ"
                    r"|Yepéyũwaátépinimasá\s+upé)\s*[.:]\s*", re.U)
split_new = []
n_split = 0
for u in ng_units:
    if u["type"] in ("titulo", "capitulo", "secao", "subsecao", "closing"):
        continue
    m = PU_MID.search(u["text"])
    if m:
        art = u["art"]
        pu_id = f"{art}.pu"
        assert pu_id not in ng_ids, f"split de {u['id']}: {pu_id} já existe"
        after = u["text"][m.end():].strip()
        before = u["text"][:m.start()].strip()
        assert len(after) > 10, f"split de {u['id']}: PU vazio"
        marker = m.group(0).strip()
        u["text"] = before
        split_new.append({"lang": "ng", "type": "paragrafo", "id": pu_id, "art": art,
                          "hier": u.get("hier"), "text": after, "pages": list(u["pages"]),
                          "split_from": u["id"], "pu_marker_variant": marker})
        ng_ids.add(pu_id)
        n_split += 1
        log(f"[split-pu] {u['id']} → {pu_id} (marcador: {marker!r})")
ng_units.extend(split_new)
log(f"[split-pu] total de PUs embutidos divididos: {n_split}")

# --------------------------- pareamentos cruzados (erros do livro / ECs) ------
# (pt_id, ng_id, nota) — verificados manualmente pelo conteúdo.
CROSS_PAIRS = [
    ("art92.par2", "art92.par1",
     "Livro NG omitiu o § 1º do art. 92 (sede na Capital Federal) e numerou como '§ 1º' o conteúdo "
     "do § 2º (jurisdição nacional). Pareado pelo conteúdo."),
    ("art105.pu", "art105.par1",
     "EC 125/2022 renumerou o parágrafo único do art. 105 como § 1º; NG segue a numeração nova."),
    ("art105.pu.inc1", "art105.par1.inc1", "Idem art105.pu (EC 125/2022)."),
    ("art105.pu.inc2", "art105.par1.inc2", "Idem art105.pu (EC 125/2022)."),
    ("art160.pu", "art160.par1",
     "EC 113/2021 renumerou o parágrafo único do art. 160 como § 1º; NG segue a numeração nova."),
    ("art160.pu.inc1", "art160.par1.inc1", "Idem art160.pu (EC 113/2021)."),
    ("art160.pu.inc2", "art160.par1.inc2", "Idem art160.pu (EC 113/2021)."),
    ("art201.par6", "art201.par5",
     "Livro NG omitiu o § 5º do art. 201 (vedação de filiação facultativa) e numerou como '§ 5º' o "
     "conteúdo do § 6º (gratificação natalina). Pareado pelo conteúdo."),
    ("art139.caput.inc2", "art139.caput.inc1",
     "Livro NG omitiu o inciso I do art. 139 e renumerou: seu 'I' = II do PT (detenção em edifício)."),
    ("art139.caput.inc3", "art139.caput.inc2",
     "Idem art139: 'II' do NG = III do PT (restrições a comunicações/imprensa)."),
    ("art22.caput.inc12", "art22.caput.inc11",
     "Livro NG omitiu o inciso XI do art. 22 (trânsito e transporte) e numerou como 'XI' o conteúdo "
     "do XII (jazidas, minas, recursos minerais). Pareado pelo conteúdo."),
]
if PTL == "pt2022":
    # a edição 2022 já traz art105 §1º e art160 §§1º-2º (EC 125/113): pareiam por id
    CROSS_PAIRS = [c for c in CROSS_PAIRS if not c[0].startswith(("art105.pu", "art160.pu"))]
cross_by_ng = {ng_id: (pt_id, nota) for pt_id, ng_id, nota in CROSS_PAIRS}
n_cross = 0
for u in ng_units:
    if u["id"] in cross_by_ng:
        pt_id, nota = cross_by_ng[u["id"]]
        u["cross_from"] = u["id"]
        u["cross_note"] = nota
        u["id"] = pt_id
        n_cross += 1
assert n_cross == len(CROSS_PAIRS), f"cross-pairs aplicados: {n_cross}/{len(CROSS_PAIRS)}"
log(f"[cross] pareamentos cruzados aplicados: {n_cross}")

# ------------------------------------------------------- normalização do NG ---
NG_CHARMAP = {
    "ū": "ũ", "ē": "ẽ", "ī": "ĩ", "ῖ": "ĩ", "Ē": "Ẽ", "Ū": "Ũ",
    "ữ": "ũ", "û": "ũ",
    "È": "É", "Ì": "Í", "ì": "í", "À": "Á", "à": "á", "Â": "Ã",
    "ŕ": "r", "ķ": "k", "ă": "ã", "ā": "ã",
    "´": "",
}
ng_norm_counts = {}
def normalize_ng(t):
    for a, b in NG_CHARMAP.items():
        n = t.count(a)
        if n:
            ng_norm_counts[a] = ng_norm_counts.get(a, 0) + n
            t = t.replace(a, b)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" ([,;.:])", r"\1", t)  # espaço antes de pontuação (artefato)
    return t.strip()

# ------------------------------------------- limpeza de refs. de EC no PT -----
RE_EC = re.compile(r"\s*\(\s*(?:EC|ECR)\s*nºs?\s*[^)]*\)")
n_ec = 0
for u in pt_units:
    refs = RE_EC.findall(u["text"])
    if refs:
        n_ec += len(refs)
        u["ec_refs"] = [r.strip() for r in refs]
        u["text"] = RE_EC.sub("", u["text"]).strip()
        u["text"] = re.sub(r" {2,}", " ", u["text"])
log(f"[pt] referências (EC nº ...) removidas: {n_ec} (inventário prévio: 144)")

# --------------------------------------------------------------- indexação ----
pt_by_id = {u["id"]: u for u in pt_units}
ng_by_id = {u["id"]: u for u in ng_units}
assert len(pt_by_id) == len(pt_units), "ids PT duplicados"
assert len(ng_by_id) == len(ng_units), "ids NG duplicados"

common = [i for i in pt_by_id if i in ng_by_id]
pt_only = [i for i in pt_by_id if i not in ng_by_id]
ng_only = [i for i in ng_by_id if i not in pt_by_id]

# ordem canônica: ordem de aparição no PT (e NG para ng_only)
order_pt = {u["id"]: n for n, u in enumerate(pt_units)}
order_ng = {u["id"]: n for n, u in enumerate(ng_units)}
common.sort(key=lambda i: order_pt[i])
pt_only.sort(key=lambda i: order_pt[i])
ng_only.sort(key=lambda i: order_ng[i])

# ------------------------------------------------- anotações manuais ----------
MANUAL_NOTES = {
    "art39.caput": "PT imprime a redação da EC 19/98 (aplicação suspensa pela ADI 2.135); o livro NG "
                   "traduz as duas redações — o par usa a 1ª tradução; a 2ª está em art39.caput.bis (ng_only).",
    "art39.caput.bis": "2ª tradução do caput do art. 39 no livro NG (corresponde à 'redação anterior' "
                       "impressa em nota na edição do Senado).",
    "art68.caput": "Livro NG omitiu o caput do art. 68; seus §§ 1º-3º foram impressos sob o art. 67 e "
                   "remapeados para o art. 68 nesta base.",
    "art194.pu": "Após o PU traduzido, o livro NG trazia uma linha em português sem tradução "
                 "('I - universalidade da cobertura e do atendimento:'), removida desta base.",
    # omissões do livro NG confirmadas por leitura do conteúdo:
    "art4.caput.inc10": "Livro NG omitiu o inciso X do art. 4º (concessão de asilo político); o IX foi "
                        "impresso com numeração 'VIX'.",
    "art21.caput.inc23.d": "Livro NG omitiu a alínea d do art. 21, XXIII (responsabilidade civil por danos nucleares).",
    "art22.caput.inc11": "Livro NG omitiu o inciso XI do art. 22 (trânsito e transporte); o conteúdo do XII foi "
                         "impresso com o rótulo XI (pareado por conteúdo com o XII do PT).",
    "art72.par1": "Livro NG omitiu os §§ 1º e 2º do art. 72.",
    "art72.par2": "Livro NG omitiu os §§ 1º e 2º do art. 72.",
    "art84.caput.inc27": "Livro NG omitiu o inciso XXVII do art. 84 (exercer outras atribuições).",
    "art85.pu": "Livro NG omitiu o parágrafo único do art. 85.",
    "art92.par1": "Livro NG omitiu o § 1º do art. 92 (sede na Capital Federal); seu '§ 1º' é o § 2º do PT "
                  "(pareado por conteúdo).",
    "art102.caput.inc1.i": "Livro NG omitiu a alínea i do art. 102, I (habeas corpus com coator Tribunal Superior).",
    "art128.par5.inc2.f": "Livro NG omitiu a alínea f do art. 128, § 5º, II.",
    "art139.caput.inc1": "Livro NG omitiu o inciso I do art. 139 (obrigação de permanência em localidade); "
                         "renumerou II→'I' e III→'II' (pareados por conteúdo).",
    "art201.par5": "Livro NG omitiu o § 5º do art. 201 (vedação de filiação como segurado facultativo); seu "
                   "'§ 5º' é o § 6º do PT (pareado por conteúdo).",
}

# atribuição de ECs para unidades existentes só no NG (base ~2022) — para o relatório
EC_ATTRIB = [
    (r"^art5\.caput\.inc79$", "EC 115/2022 (proteção de dados pessoais)"),
    (r"^art6\.pu$", "EC 114/2021 (renda básica familiar)"),
    (r"^art14\.par1[23]$", "EC 111/2021 (consultas populares nas eleições municipais)"),
    (r"^art17\.", "EC 97/2017 e EC 117/2022 (partidos: cláusula de desempenho, fundos e federações)"),
    (r"^art21\.caput\.inc26$", "EC 115/2022"),
    (r"^art22\.caput\.inc30$", "EC 115/2022"),
    (r"^art37\.par1[3-6]$", "EC 103/2019 (reforma da previdência)"),
    (r"^art39\.par9$", "EC 103/2019"),
    (r"^art40\.", "EC 103/2019 (reforma da previdência)"),
    (r"^art42\.par3$", "EC 101/2019 (militares estaduais: acumulação)"),
    (r"^art49\.caput\.inc18$", "EC 135/2022? — verificar (inciso XVIII do art. 49)"),
    (r"^art84\.caput\.inc28$", "EC 109/2021 (estado de calamidade — arts. 167-B a 167-G)"),
    (r"^art92\.caput\.inc2A$", "EC 92/2016 (TST no rol do Judiciário; posterior a fev/2016)"),
    (r"^art100\.par(1[1789]|2[0-2])", "EC 113/2021 e EC 114/2021 (precatórios)"),
    (r"^art103-B\.par4\.inc", "EC 103/2019? — verificar"),
    (r"^art105\.par[23]", "EC 125/2022 (relevância no recurso especial)"),
    (r"^art111-A\.par3$", "EC 92/2016"),
    (r"^art144\.caput\.inc6$", "EC 104/2019 (polícias penais)"),
    (r"^art144\.par5A$", "EC 104/2019"),
    (r"^art149\.par1[ABC]$", "EC 103/2019"),
    (r"^art155\.par1\.inc5$", "verificar (art. 155, § 1º, V)"),
    (r"^art156\.par1A$", "EC 116/2022 (ITBI/imunidade? — verificar)"),
    (r"^art159\.caput\.inc1\.f$", "EC 112/2021"),
    (r"^art160\.par2$", "EC 113/2021"),
    (r"^art163-A", "EC 108/2020 (transparência fiscal)"),
    (r"^art16[56]\.par", "EC 100/2019, EC 102/2019 e EC 126/2022 (orçamento impositivo e emendas)"),
    (r"^art166\.par9A$", "EC 126/2022"),
    (r"^art166-A", "EC 105/2019 (transferências especiais de emendas individuais)"),
    (r"^art167\.caput\.inc1[2-4]$", "EC 109/2021"),
    (r"^art167\.par[67]$", "EC 109/2021"),
    (r"^art167-[A-G]", "EC 109/2021 (regime fiscal — despesas obrigatórias)"),
    (r"^art168\.par[12]$", "EC 109/2021"),
    (r"^art193\.pu$", "EC 108/2020 (planejamento das políticas sociais)"),
    (r"^art195\.par14$", "EC 103/2019"),
    (r"^art198\.par([789]|1[0-5])$", "EC 120/2022, EC 124/2022 e EC 127/2022 (ACS/ACE e piso da enfermagem)"),
    (r"^art201\.par(1\.inc[12]|9A|1[4-6])$", "EC 103/2019"),
    (r"^art203\.caput\.inc6$", "EC 114/2021"),
    (r"^art206\.caput\.inc9$", "EC 108/2020 (educação: direito à aprendizagem ao longo da vida)"),
    (r"^art211\.par[67]$", "EC 108/2020 (FUNDEB)"),
    (r"^art212\.par[789]$", "EC 108/2020"),
    (r"^art212-A", "EC 108/2020 (FUNDEB permanente)"),
    (r"^art225\.par1\.inc8$", "EC 123/2022 (biocombustíveis)"),
    (r"^art225\.par7$", "EC 96/2017 (práticas desportivas com animais)"),
    (r"^art239\.par5$", "EC 103/2019"),
]
def ec_attrib(uid):
    for pat, ec in EC_ATTRIB:
        if re.match(pat, uid):
            return ec
    return ""

# artigos tocados por ECs promulgadas depois da edição PT (EC 92/2016 em diante,
# até ~EC 128/2022, compatível com a base da tradução NG) — flag conservadora
WATCHLIST = {
    "art5", "art14", "art16", "art17", "art20", "art21", "art22", "art29-A", "art37", "art38",
    "art39", "art40", "art42", "art49", "art84", "art92", "art100", "art105", "art109",
    "art111-A", "art144", "art149", "art156", "art158", "art159", "art160", "art163",
    "art165", "art166", "art166-A", "art167", "art168", "art195", "art198", "art201",
    "art202", "art208", "art211", "art212", "art212-A", "art213", "art225", "art239",
}

DIGITS = re.compile(r"\d+")
def digitset(t):
    return set(DIGITS.findall(t))

pairs = []
ratios = []
for i in common:
    p, n = pt_by_id[i], ng_by_id[i]
    yrl = normalize_ng(n["text"])
    pair = {"id": i, "type": p["type"], "art": p.get("art"),
            "pt": p["text"], "yrl": yrl, "yrl_raw": n["text"],
            "pt_pages": p["pages"], "ng_pages": n["pages"], "flags": [], "notes": []}
    if "ec_refs" in p: pair["ec_refs"] = p["ec_refs"]
    if "remapped_from" in n: pair["ng_remapped_from"] = n["remapped_from"]
    if "cross_from" in n:
        pair["ng_cross_from"] = n["cross_from"]
        pair["flags"].append("pareado_por_conteudo")
        pair["notes"].append(n["cross_note"])
    if "split_from" in n:
        pair["ng_split_from"] = n["split_from"]
        pair["notes"].append(f"PU embutido no livro NG após {n['split_from']} "
                             f"(marcador: {n.get('pu_marker_variant','')!r}); dividido nesta base.")
    if i == "art39.caput": pair["flags"].append("verificar_manualmente")
    if i in MANUAL_NOTES: pair["notes"].append(MANUAL_NOTES[i])
    if not p["text"].strip(): pair["flags"].append("pt_vazio")
    if not yrl.strip(): pair["flags"].append("ng_vazio")
    dp, dn = digitset(p["text"]), digitset(yrl)
    if dp or dn:
        inter, union = dp & dn, dp | dn
        if len(union) >= 2 and len(inter) / len(union) < 0.5 and len(dp ^ dn) >= 2:
            pair["flags"].append("digitos_divergem")
            pair["digitos"] = {"pt": sorted(dp), "yrl": sorted(dn)}
    if PTL == "pt" and p.get("art") in WATCHLIST:
        pair["flags"].append("artigo_com_emenda_pos_2016")
    if p["text"].strip() and yrl.strip():
        ratios.append((i, len(yrl) / max(1, len(p["text"]))))
    pairs.append(pair)

# razão de comprimento: outliers robustos (log-razão, mediana ± 3*MAD)
import math
lr = {i: math.log(r) for i, r in ratios}
med = statistics.median(lr.values())
mad = statistics.median(abs(v - med) for v in lr.values()) or 0.1
for pair in pairs:
    v = lr.get(pair["id"])
    if v is not None and abs(v - med) > 3.5 * mad:
        pair["flags"].append("razao_comprimento_atipica")
        pair["razao"] = round(math.exp(v), 2)

# ---------------------------------------------------------------- pt_only -----
pt_only_rows = []
for i in pt_only:
    p = pt_by_id[i]
    if re.match(r"^\(?Revogad", p["text"].strip()):
        cls = "pt_revogado_ng_omite"
    elif i in MANUAL_NOTES:
        cls = "ng_omite_livro"
    else:
        cls = "pt_only_investigar"
    pt_only_rows.append({"id": i, "classe": cls, "pt": p["text"], "pages": p["pages"],
                         "nota": MANUAL_NOTES.get(i, "")})

ng_only_rows = []
for i in ng_only:
    n = ng_by_id[i]
    ec = ec_attrib(i)
    cls = "ng_acrescimo_ec_pos2016" if ec else ("ng_extra_livro" if i.endswith(".bis") else "ng_only_investigar")
    ng_only_rows.append({"id": i, "classe": cls, "ec": ec, "yrl": normalize_ng(n["text"]),
                         "yrl_raw": n["text"], "pages": n["pages"], "nota": MANUAL_NOTES.get(i, "")})

with open(os.path.join(WORK, "pairs.jsonl"), "w", encoding="utf-8") as f:
    for r in pairs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(WORK, "unpaired.jsonl"), "w", encoding="utf-8") as f:
    for r in pt_only_rows + ng_only_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

log(f"[norm-ng] substituições: {ng_norm_counts}")
print(f"pares: {len(pairs)} | pt_only: {len(pt_only)} | ng_only: {len(ng_only)}")
from collections import Counter
fc = Counter(fl for p in pairs for fl in p["flags"])
print("flags nos pares:", dict(fc))
print("\n--- PT sem par ---")
for r in pt_only_rows:
    print(f"  {r['classe']:24} {r['id']:28} {r['pt'][:60]!r}")
print("\n--- NG sem par ---")
for r in ng_only_rows:
    print(f"  {r['id']:28} {r['yrl'][:70]!r}")
LOG.close()
