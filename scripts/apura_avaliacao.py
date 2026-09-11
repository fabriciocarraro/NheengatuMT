# -*- coding: utf-8 -*-
"""Apura as respostas dos avaliadores falantes (JSONs devolvidos pelo kit HTML)
usando o gabarito cego. Gera as tabelas da seção de avaliação humana do paper.

Uso: python scripts/apura_avaliacao.py work/avaliacao/respostas/*.json
"""
import json, sys, glob, statistics as st
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
GAB = json.loads((ROOT / "work" / "avaliacao" / "gabarito_cego.json").read_text(encoding="utf-8"))
ITENS = {i["id"]: i for i in
         json.loads((ROOT / "work" / "avaliacao" / "itens.json").read_text(encoding="utf-8"))["itens"]}

arquivos = []
for padrao in (sys.argv[1:] or [str(ROOT / "work" / "avaliacao" / "respostas" / "*.json")]):
    arquivos += glob.glob(padrao)
if not arquivos:
    sys.exit("nenhum arquivo de respostas encontrado")

avaliadores = []
for a in arquivos:
    d = json.loads(Path(a).read_text(encoding="utf-8"))
    avaliadores.append((Path(a).stem, d.get("respostas", {}), d.get("perfil", {})))
print(f"avaliadores: {len(avaliadores)} ({', '.join(n for n, _, _ in avaliadores)})\n")

# ---------------- BLOCO A: qualidade por domínio, com âncoras humanas ----------------
print("=== BLOCO A — qualidade (notas 1-5; SISTEMA vs ÂNCORAS de tradução humana) ===")
por_dom = defaultdict(lambda: defaultdict(list))     # sistema, por domínio
anc = defaultdict(list)                              # âncoras (referência humana)
usab, probs = defaultdict(Counter), Counter()
usab_anc = Counter()
for nome, resp, _ in avaliadores:
    for iid, r in resp.items():
        if not iid.startswith("A"):
            continue
        dom = ITENS[iid]["dominio"]
        e_ancora = GAB[iid].get("ancora", False)
        for campo in ("adequacao", "naturalidade"):
            if r.get(campo):
                (anc[campo] if e_ancora else por_dom[dom][campo]).append(int(r[campo]))
        if r.get("usabilidade"):
            (usab_anc if e_ancora else usab[dom])[r["usabilidade"]] += 1
        if not e_ancora:
            for p in r.get("problemas", []):
                probs[p] += 1

def linha(rot, ad, na, u):
    if not ad:
        return
    tot = sum(u.values()) or 1
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0
    print(f"  {rot:26s} adequação {st.mean(ad):.2f}±{sd(ad):.2f} | "
          f"naturalidade {st.mean(na):.2f}±{sd(na):.2f} | "
          f"usável {100*(u['boa']+u['ajuste'])/tot:.0f}%")

for dom, rot in (("juridico", "sistema · jurídico"), ("proximo", "sistema · dicionário"),
                 ("fala", "sistema · fala")):
    linha(rot, por_dom[dom]["adequacao"], por_dom[dom]["naturalidade"], usab[dom])
todos_ad = [x for d in por_dom.values() for x in d["adequacao"]]
todos_na = [x for d in por_dom.values() for x in d["naturalidade"]]
usab_tot = Counter()
for d in usab.values():
    usab_tot.update(d)
linha("SISTEMA (todos)", todos_ad, todos_na, usab_tot)
linha("ÂNCORA (tradução humana)", anc["adequacao"], anc["naturalidade"], usab_anc)
if todos_ad and anc["adequacao"]:
    print(f"  → o sistema alcança {100*st.mean(todos_ad)/st.mean(anc['adequacao']):.0f}% "
          f"da adequação da tradução humana e "
          f"{100*st.mean(todos_na)/st.mean(anc['naturalidade']):.0f}% da naturalidade, "
          f"no MESMO protocolo e às cegas")
    if st.mean(anc["adequacao"]) < 3.0:
        print("  ⚠ ATENÇÃO: âncoras humanas com nota baixa — checar calibração/atenção do avaliador")
print(f"  problemas marcados (só itens do sistema): {dict(probs.most_common())}")
print(f"  → 'grafia de outra tradição': {probs['grafia_outra']} "
      f"(erro ortográfico ≠ erro de tradução)\n")

# ---------------- BLOCO B: terminologia (cego) ----------------
print("=== BLOCO B — terminologia (decodificação livre × restrita) ===")
cnt = Counter()
for nome, resp, _ in avaliadores:
    for iid, r in resp.items():
        if not iid.startswith("B"):
            continue
        g = GAB[iid]
        for campo, sufixo in (("termo_correto", "termo"), ("preferida", "pref")):
            v = r.get(campo)
            if v in ("A", "B"):
                cnt[f"{sufixo}_{g[v]}"] += 1
            elif v:
                cnt[f"{sufixo}_{v}"] += 1
        if r.get("encaixe"):
            cnt[f"encaixe_{r['encaixe']}"] += 1
print(f"  usa o termo esperado: restrita {cnt['termo_restrita']} × livre {cnt['termo_livre']} "
      f"(ambas {cnt['termo_ambas']}, nenhuma {cnt['termo_nenhuma']})")
print(f"  preferida no geral:   restrita {cnt['pref_restrita']} × livre {cnt['pref_livre']} "
      f"(empate {cnt['pref_iguais']})")
print(f"  encaixe do termo: natural {cnt['encaixe_natural']} | forçado {cnt['encaixe_forcado']} | "
      f"depende {cnt['encaixe_depende']}\n")

# ---------------- BLOCO C: modelo × referência oficial (cego) ----------------
print("=== BLOCO C — modelo × tradução oficial humana (cego) ===")
c = Counter()
for nome, resp, _ in avaliadores:
    for iid, r in resp.items():
        if not iid.startswith("C"):
            continue
        g = GAB[iid]
        for campo in ("melhor", "fiel"):
            v = r.get(campo)
            if v in ("A", "B"):
                c[f"{campo}_{g[v]}"] += 1
            elif v:
                c[f"{campo}_{v}"] += 1
n_melhor = c["melhor_modelo"] + c["melhor_referencia_oficial"] + c["melhor_iguais"] + c["melhor_ambas_ruins"]
if n_melhor:
    print(f"  melhor: modelo {c['melhor_modelo']} × oficial {c['melhor_referencia_oficial']} "
          f"(empate {c['melhor_iguais']}, ambas ruins {c['melhor_ambas_ruins']}) — "
          f"modelo preferido em {100*c['melhor_modelo']/n_melhor:.0f}% dos julgamentos")
    print(f"  mais fiel: modelo {c['fiel_modelo']} × oficial {c['fiel_referencia_oficial']} "
          f"(empate {c['fiel_iguais']})\n")

# ---------------- concordância entre avaliadores (bloco A) ----------------
if len(avaliadores) > 1:
    print("=== concordância entre avaliadores (bloco A) ===")
    for campo in ("adequacao", "naturalidade"):
        difs = []
        ids = set.intersection(*[{k for k, v in r.items() if k.startswith("A") and v.get(campo)}
                                 for _, r, _ in avaliadores])
        for iid in ids:
            notas = [int(r[iid][campo]) for _, r, _ in avaliadores]
            difs.append(max(notas) - min(notas))
        if difs:
            print(f"  {campo}: {len(ids)} itens em comum | diferença média {st.mean(difs):.2f} pontos | "
                  f"exata em {100*sum(1 for d in difs if d==0)/len(difs):.0f}%")
    print()

# ---------------- perguntas abertas ----------------
print("=== BLOCO D — perguntas de língua ===")
for pid in ("D1", "D2", "D3", "D4"):
    for nome, resp, _ in avaliadores:
        txt = (resp.get(pid) or {}).get("resposta", "").strip()
        if txt:
            print(f"  [{pid}] {nome}: {txt[:300]}")
print("\n(colar as respostas de D1-D3 em docs/PERGUNTAS_FALANTES.md e fechar os itens)")
