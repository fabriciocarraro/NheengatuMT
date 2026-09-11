# -*- coding: utf-8 -*-
"""
Etapa 5: consolida pares + vereditos do Planalto + adjudicações manuais e
emite o corpus final:

  corpus/constituicao_pt_yrl.tsv        — pares limpos (id, tipo, pt, yrl, flags)
  corpus/divergencias.tsv               — pares divergentes e unidades sem par
  corpus/constituicao_pt_yrl_full.jsonl — tudo, com metadados completos
"""
import sys, io, os, json, re
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(BASE, "work")
OUT = os.path.join(BASE, "corpus")
os.makedirs(OUT, exist_ok=True)

pairs = [json.loads(l) for l in open(os.path.join(WORK, "pairs.jsonl"), encoding="utf-8")]
unpaired = [json.loads(l) for l in open(os.path.join(WORK, "unpaired.jsonl"), encoding="utf-8")]
check = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(WORK, "planalto_check.jsonl"), encoding="utf-8"))}

# Adjudicação manual dos casos que a checagem automática não resolveu.
# "editorial" = diferença tipográfica/editorial entre a edição do Senado (2016) e o
# Planalto — o texto legal é o mesmo; par seguro.
BASE_PT = "pt2022"  # base atual do corpus (edição Senado até a EC 128/2022)

# (kind, nota): "editorial" aplica-se apenas quando a checagem Planalto falhou
# (diferenças tipográficas entre edições); "divergente" aplica-se sempre.
MANUAL = {
    "art73.par1.inc1": ("divergente", "NG traz '60 akayu' onde a redação (EC 122/2022) diz 'menos de setenta' — não corresponde a nenhuma redação (erro do livro)."),
    "art101.caput": ("divergente", "NG traz '65 akayú' (redação pré-EC 122/2022); o PT desta base traz 'menos de setenta' (EC 122). Base inconsistente do livro."),
    "art37.caput.inc16.b": ("editorial", "Falso positivo do tachado (texto idêntico em camada antiga)."),
    "art144.caput.inc5": ("editorial", "Falso positivo do tachado (texto idêntico em camada antiga)."),
    "art239.par3": ("editorial", "Falso positivo do tachado (texto idêntico em camada antiga)."),
    "art100.par11": ("editorial", "§ 11 na redação da EC 113/2021; alterações posteriores são da EC 136/2025 (pós-base NG)."),

    "art14.par9": ("editorial", "Senado: 'a moralidade para o exercício do mandato'; Planalto: 'a moralidade para exercício de mandato' (ECR 4/94)."),
    "art29.caput.inc4.f": ("editorial", "Formatação de números/alíneas da EC 58/2009 difere entre as edições."),
    "art29.caput.inc4.j": ("editorial", "Formatação de números/alíneas da EC 58/2009 difere entre as edições."),
        "art29-A.caput.inc6": ("editorial", "Edição do Senado traz '8.000.0001' (typo; leia-se 8.000.001)."),
        "art29-A.caput": ("editorial", "Diferenças de formatação (percentuais) entre as edições; texto EC 109/2021 em ambas."),
    "art29-A.par3": ("editorial", "Diferenças de formatação entre as edições."),
    "art50.caput": ("editorial", "Senado: 'Ministro de Estado ou quaisquer titulares'; Planalto: 'Ministro de Estado, quaisquer titulares'."),
    "art50.par2": ("editorial", "Diferenças de pontuação entre as edições."),
    "art52.caput.inc13": ("editorial", "Diferenças de pontuação entre as edições (EC 19/98)."),
    "art53.par3": ("editorial", "Diferenças de pontuação entre as edições (EC 35/2001)."),
    "art98.caput.inc1": ("editorial", "Diferenças de pontuação entre as edições."),
    "art150.par3": ("editorial", "Sem EC ≤128 na região; eventual mudança é da EC 132/2023 (posterior à base NG)."),
    "art155.par1.inc3": ("editorial", "Alterações do dispositivo são da EC 132/2023 (posterior à base NG)."),
    "art208.caput.inc7": ("editorial", "Edição do Senado traz 'estapas' (typo; leia-se 'etapas')."),
    "art211.par1": ("editorial", "Edição do Senado traz 'redistribuitiva' (typo; leia-se 'redistributiva')."),
    "art219-B.caput": ("editorial", "Planalto inclui a sigla '(SNCTI)'; edição do Senado não."),
    "art219-B.par1": ("editorial", "Planalto inclui a sigla '(SNCTI)'; edição do Senado não."),
    "art224.caput": ("editorial", "Senado: 'como órgão auxiliar'; Planalto: 'como seu órgão auxiliar'."),
    "art249.caput": ("editorial", "Diferenças de pontuação entre as edições (EC 20/98)."),
        }

# reclassificação dos pt_only remanescentes (unidades extintas por EC até 2022)
PT_ONLY_EC = {}
NG_OMITE_2022 = {
    "art163.caput.inc8": "EC 109/2021 — inciso VIII do art. 163 omitido pelo livro NG",
    "art163.caput.inc8.a": "EC 109/2021 — omitido pelo livro NG",
    "art163.caput.inc8.b": "EC 109/2021 — omitido pelo livro NG",
    "art163.caput.inc8.c": "EC 109/2021 — omitido pelo livro NG",
    "art163.caput.inc8.d": "EC 109/2021 — omitido pelo livro NG",
    "art163.caput.inc8.e": "EC 109/2021 — omitido pelo livro NG",
    "art163.pu": "EC 109/2021 — parágrafo único do art. 163 omitido pelo livro NG",
    "art164-A.caput": "EC 109/2021 — art. 164-A omitido pelo livro NG",
    "art164-A.pu": "EC 109/2021 — art. 164-A omitido pelo livro NG",
    "art100.par21.inc4": "Livro NG funde o conteúdo deste inciso IV no fim do inciso III do § 21 (EC 113/2021).",
}
# (na base pt2022 não há dispositivos 'extintos': eles aparecem como (Revogado) e pareiam ou ficam como revogados)

# notas sobre erros de conteúdo do próprio livro NG (pares mantidos, com aviso)
NOTAS_EXTRA = {
    "art41.par2": "Tradução do livro insere referência espúria a 'art. 142, § 3º, ĩsisu X' (contaminação do art. 42 vizinho).",
    "art130-A.par2.inc2": "Livro NG cita 'art. 39' onde o texto é 'art. 37' (erro de referência).",
    "art235.caput.inc1": "Livro NG traz '10 depautado' onde o PT diz 'dezessete Deputados' (erro numérico).",
    "art167-E.caput": "Livro NG cita 'art. 197' onde o texto é 'art. 167' (erro de referência) e acrescenta paráfrase.",
    "art201.par8": "Tradução livre: NG resolve 'reduzido em 5 anos' como '55 akayu' (impreciso).",
    "art100.par21.inc3": "Livro NG funde neste inciso o conteúdo do inciso IV do § 21 (prestação de contas/desvio).",
}

rows_ok, rows_div = [], []
full = []
stats = Counter()
for p in pairs:
    uid = p["id"]
    ver = check.get(uid, {}).get("veredito", "")
    ec = check.get(uid, {}).get("ec")
    status = "ok"
    detail = ""
    if uid in NOTAS_EXTRA:
        p["notes"].append(NOTAS_EXTRA[uid])
        if "erro_conteudo_livro" not in p["flags"]:
            p["flags"].append("erro_conteudo_livro")
    if uid in MANUAL:
        kind, note = MANUAL[uid]
        if kind == "divergente":
            p["notes"].append(note)
            status = "divergente"
            detail = note
        elif ver in ("nao_localizado_verificar", "tachado_ec_ate_128", "tachado_ec_desconhecida", ""):
            p["notes"].append(note)
            ver = "fonte_editorial"
    elif ver == "tachado_ec_ate_128":
        status = "divergente"
        detail = f"Texto PT (edição 2016) foi alterado pela EC {ec} antes da base NG (~2022): o NG traduz a redação nova."
        p["notes"].append(detail)
    elif ver == "tachado_ec_desconhecida":
        status = "divergente"
        detail = "Texto PT tachado no Planalto; EC substituidora não identificada automaticamente."
        p["notes"].append(detail)
    if "verificar_manualmente" in p["flags"] and status == "ok":
        status = "divergente"
        detail = p["notes"][0] if p["notes"] else "verificar manualmente"
    p["planalto"] = ver or "n/a"
    if ec: p["planalto_ec"] = ec
    p["status"] = status
    # flag de emenda por artigo já não é necessária quando o Planalto confirmou a unidade
    if ver in ("vivo", "vivo_cosmetico", "tachado_ec_129_mais", "fonte_editorial") \
            and "artigo_com_emenda_pos_2016" in p["flags"]:
        p["flags"].remove("artigo_com_emenda_pos_2016")
    full.append(p)
    stats[status] += 1
    if status == "ok":
        rows_ok.append(p)
    else:
        rows_div.append(p)

def clean(t):
    return t.replace("\t", " ").replace("\n", " ").strip()

with open(os.path.join(OUT, "constituicao_pt_yrl.tsv"), "w", encoding="utf-8", newline="") as f:
    f.write("id\ttipo\tpt\tyrl\tflags\n")
    for p in rows_ok:
        f.write(f"{p['id']}\t{p['type']}\t{clean(p['pt'])}\t{clean(p['yrl'])}\t{','.join(p['flags'])}\n")

with open(os.path.join(OUT, "divergencias.tsv"), "w", encoding="utf-8", newline="") as f:
    f.write("tipo_registro\tid\tclasse\tpt\tyrl\tnota\n")
    for p in rows_div:
        nota = " | ".join(p["notes"])
        f.write(f"par_divergente\t{p['id']}\t{p.get('planalto','')}"
                f"{('/EC ' + str(p.get('planalto_ec'))) if p.get('planalto_ec') else ''}\t"
                f"{clean(p['pt'])}\t{clean(p['yrl'])}\t{clean(nota)}\n")
    for r in unpaired:
        if "pt" in r:
            cls = r["classe"]
            nota = r.get("nota", "")
            if r["id"] in PT_ONLY_EC:
                cls = "pt_extinto_por_ec"
                nota = PT_ONLY_EC[r["id"]]
            if r["id"] in NG_OMITE_2022:
                cls = "ng_omite_livro"
                nota = NG_OMITE_2022[r["id"]]
            f.write(f"pt_sem_par\t{r['id']}\t{cls}\t{clean(r['pt'])}\t\t{clean(nota)}\n")
        else:
            nota = r.get("nota", "") or r.get("ec", "")
            f.write(f"ng_sem_par\t{r['id']}\t{r['classe']}\t\t{clean(r['yrl'])}\t{clean(nota)}\n")

with open(os.path.join(OUT, "constituicao_pt_yrl_full.jsonl"), "w", encoding="utf-8") as f:
    for p in full:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for r in unpaired:
        rr = dict(r)
        if rr["id"] in PT_ONLY_EC:
            rr["classe"] = "pt_extinto_por_ec"
            rr["nota"] = PT_ONLY_EC[rr["id"]]
        if rr.get("id") in NG_OMITE_2022:
            rr["classe"] = "ng_omite_livro"
            rr["nota"] = NG_OMITE_2022[rr["id"]]
        rr["status"] = "sem_par"
        f.write(json.dumps(rr, ensure_ascii=False) + "\n")

print("status dos pares:", dict(stats))
print("TSV limpo:", len(rows_ok), "| divergentes:", len(rows_div), "| sem par:", len(unpaired))
fc = Counter(fl for p in rows_ok for fl in p["flags"])
print("flags remanescentes no TSV limpo:", dict(fc))
pc = Counter(p["planalto"] for p in full)
print("veredito planalto nos pares:", dict(pc))
chars_pt = sum(len(p["pt"]) for p in rows_ok)
chars_ng = sum(len(p["yrl"]) for p in rows_ok)
print(f"volume (pares ok): PT {chars_pt} chars, YRL {chars_ng} chars")
