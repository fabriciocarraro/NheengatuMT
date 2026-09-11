# -*- coding: utf-8 -*-
"""Extrai pares yrl↔PT dos EAFs do Refubium (Reich, FU Berlin — CC BY-NC-SA).

Estrutura: tiers por falante 'P<n>.ort' (transcrição yrl) e 'P<n>.transl.pt'
(tradução PT; grafias variantes nos nomes: transl/trans/tranls). Tiers são
todos top-level → alinhamento por SOBREPOSIÇÃO TEMPORAL (ort × transl.pt do
mesmo falante; melhor sobreposição, mínimo 50% do menor intervalo).

Saída: work/refubium_pares.jsonl
  {"arquivo", "tarefa", "falante", "t0_ms", "t1_ms", "yrl", "pt"}
Fonte: Reich, U. Corpora amerikanischer Sprachen (Nheengatú). Refubium, 2023.
DOI 10.17169/refubium-39406. Licença CC BY-NC-SA 4.0 — citação obrigatória.
"""
import json, re, sys, io
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "work" / "refubium"
OUT = ROOT / "work" / "refubium_pares.jsonl"

ORT = re.compile(r"^(P\d+)\.ort$", re.IGNORECASE)
TPT = re.compile(r"^(P\d+)\.(?:transl?|tranls)\.pt$", re.IGNORECASE)

def annotations(root, tier):
    ts = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE", 0))
          for s in root.iter("TIME_SLOT") if s.get("TIME_VALUE") is not None}
    out = []
    for a in tier.iter("ALIGNABLE_ANNOTATION"):
        t0 = ts.get(a.get("TIME_SLOT_REF1"))
        t1 = ts.get(a.get("TIME_SLOT_REF2"))
        v = a.find("ANNOTATION_VALUE")
        txt = (v.text or "").strip() if v is not None else ""
        if t0 is not None and t1 is not None and txt:
            out.append((t0, t1, re.sub(r"\s+", " ", txt)))
    out.sort()
    return out

def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0, hi - lo)

def main():
    pares = []
    stats = Counter()
    for eaf in sorted(SRC.glob("*.eaf")):
        tarefa = eaf.name.split("_")[0]
        root = ET.parse(eaf).getroot()
        tiers = {t.get("TIER_ID"): t for t in root.iter("TIER")}
        speakers = {}
        for tid, t in tiers.items():
            m = ORT.match(tid or "")
            if m: speakers.setdefault(m.group(1).upper(), {})["ort"] = t
            m = TPT.match(tid or "")
            if m: speakers.setdefault(m.group(1).upper(), {})["pt"] = t
        for sp, d in speakers.items():
            if "ort" not in d or "pt" not in d:
                stats["falante_sem_par_de_tiers"] += 1
                continue
            orts = annotations(root, d["ort"])
            pts = annotations(root, d["pt"])
            usados = set()
            for o in orts:
                best, bi = 0, None
                for i, p in enumerate(pts):
                    if i in usados: continue
                    ov = overlap(o, p)
                    if ov > best: best, bi = ov, i
                dur_min = min(o[1] - o[0], 1) if o[1] == o[0] else min(o[1] - o[0],
                             (pts[bi][1] - pts[bi][0]) if bi is not None else o[1] - o[0])
                if bi is not None and best >= 0.5 * max(1, dur_min):
                    usados.add(bi)
                    pares.append({"arquivo": eaf.name, "tarefa": tarefa, "falante": sp,
                                  "t0_ms": o[0], "t1_ms": o[1],
                                  "yrl": o[2], "pt": pts[bi][2]})
                    stats["pareado"] += 1
                else:
                    stats["ort_sem_traducao"] += 1
            stats["pt_sem_ort"] += len(pts) - len(usados)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for p in pares:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[refubium] {len(pares)} pares yrl-PT -> {OUT}")
    print(f"  {dict(stats)}")
    por_tarefa = Counter(p["tarefa"] for p in pares)
    print(f"  por tarefa: {dict(por_tarefa)}")
    tok = sum(len(p["yrl"].split()) for p in pares)
    print(f"  tokens yrl: ~{tok}")

if __name__ == "__main__":
    main()
