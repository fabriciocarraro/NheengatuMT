# -*- coding: utf-8 -*-
"""Anotação inline de terminologia (família vencedora do WMT21/23, estilo TLA).

Insere a tradução esperada após o termo-fonte:  "... certidão de óbito | papéra
manusá resé | ..."  — o modelo aprende a USAR o termo fornecido; na inferência,
anota-se com o glossário (rate=1.0). Marcador '|' é ASCII, presente no SPM do
NLLB (nada de UNK).

Regras:
- Só entradas de nível aprovado/corrigido do glossário (parciais nunca).
- Casamento por fronteira de palavra, case-insensitive, termo mais longo
  primeiro, sem sobreposição; no máx. `max_terms` por frase.
- `rate` = probabilidade de anotar cada ocorrência casada (treino ~0.3;
  inferência/dev interno = 1.0). Determinístico dado o seed (RNG sequencial).
- Aplicar SOMENTE ao lado PT da direção por→yrl (nunca no alvo, nunca na
  direção reversa) — responsabilidade do chamador (train_nllb/evaluate_yrl).
"""
import json, re, random

class Annotator:
    def __init__(self, gloss_path, rate=1.0, seed=0, max_terms=2,
                 levels=("aprovado", "corrigido"),
                 prioriza_origem="glossario_fase2"):
        """`prioriza_origem`: entradas com esse valor no campo `origem` ganham as
        vagas de `max_terms` ANTES das demais. Existe porque o glossário pode ser
        estendido por fonte externa (verbetes de dicionário) e, sem isso, uma
        entrada nova mais longa DESLOCA uma entrada auditada — medido no
        dev_const: 54% dos segmentos mudavam de anotação e 18 termos auditados
        eram expulsos, o que confundiria "adicionar cobertura" com "remover
        anotação boa". Arquivos sem o campo `origem` ficam todos empatados, então
        o comportamento das rodadas publicadas é idêntico."""
        self.rate = float(rate)
        self.rng = random.Random(seed)
        self.max_terms = max_terms
        entries = []
        for line in open(gloss_path, encoding="utf-8"):
            e = json.loads(line)
            if e.get("nivel") and e["nivel"] not in levels:
                continue
            pt = e["pt"].strip()
            yrl = e["yrl"][0] if isinstance(e["yrl"], list) else e["yrl"]
            if not pt or not yrl:
                continue
            prio = 0 if e.get("origem") == prioriza_origem else 1
            entries.append((pt, yrl.strip(), prio))
        # prioridade primeiro, depois mais longo; ordem total determinística
        entries.sort(key=lambda x: (x[2], -len(x[0].split()), x[0].lower()))
        self.terms = []
        self.prio = []
        self.by_first = {}
        for pt, yrl, prio in entries:
            pat = re.compile(r"(?<![\w])" + re.escape(pt) + r"(?![\w])",
                             re.IGNORECASE | re.UNICODE)
            idx = len(self.terms)
            self.terms.append((pt, yrl, pat))
            self.prio.append(prio)
            fw = pt.split()[0].lower()
            self.by_first.setdefault(fw, []).append(idx)

    def __call__(self, text):
        words = set(w.lower() for w in re.findall(r"[^\W\d_]+", text, re.UNICODE))
        cand = sorted({i for w in words for i in self.by_first.get(w, [])})
        cand.sort(key=lambda i: (self.prio[i], -len(self.terms[i][0].split()),
                                 self.terms[i][0].lower()))
        spans = []
        for i in cand:
            pt, yrl, pat = self.terms[i]
            m = pat.search(text)
            if not m:
                continue
            s, e = m.span()
            if any(not (e <= s2 or s >= e2) for s2, e2, _ in spans):
                continue
            spans.append((s, e, yrl))
            if len(spans) >= self.max_terms:
                break
        spans = [sp for sp in spans if self.rng.random() < self.rate]
        for s, e, yrl in sorted(spans, key=lambda x: -x[1]):
            text = text[:e] + f" | {yrl} |" + text[e:]
        return text
