# -*- coding: utf-8 -*-
"""Extrai o corpus Nheengatu da plataforma Tycho Brahe (Unicamp) via API pública.
Retomável: pula documentos já presentes no arquivo de saída."""
import sys, io, json, time, os, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = "https://www.tycho.iel.unicamp.br"
UA = {"User-Agent": "NheengatuMT-research/1.0", "Content-Type": "application/json"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "work", "novas_fontes", "tycho_yrl.jsonl")

def req(url, body=None, timeout=120):
    r = urllib.request.Request(url, headers=UA,
                               data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
    try: return json.loads(raw)
    except json.JSONDecodeError: return raw.decode("utf-8", "replace")

def collect(obj, rows):
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj.get("translations"), dict):
            pt = obj["translations"].get("pt-BR") or obj["translations"].get("pt")
            rows.append({"yrl": obj["text"], "pt": pt})
        else:
            for v in obj.values(): collect(v, rows)
    elif isinstance(obj, list):
        for v in obj: collect(v, rows)

done_docs = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        done_docs.add(json.loads(l)["doc"])
print(f"já extraídos: {len(done_docs)} docs")

corpora = req(f"{BASE}/api/platform/corpus/open/public/list")
uid = [c for c in corpora if "heengatu" in str(c.get("name", ""))][0]["uid"]
docs = req(f"{BASE}/api/catalog/document/open/page/{uid}",
           {"categories": [], "name": "", "statuses": [], "page": 0, "size": 100})
items = docs.get("content", docs if isinstance(docs, list) else [])
print(f"documentos no corpus: {len(items)}")

out = open(OUT, "a", encoding="utf-8")
for d in items:
    duid, name = d.get("uid"), d.get("name")
    if name in done_docs:
        continue
    try:
        job = req(f"{BASE}/api/io/open/export/{duid}/TYCHO")
        jid = job if isinstance(job, str) else job.get("id") or job.get("jobId") or job
        data = None
        for _ in range(60):
            time.sleep(3)
            try:
                data = req(f"{BASE}/upload/io/export/{jid}.json")
                if not isinstance(data, str):
                    break
            except Exception:
                pass
        if data is None or isinstance(data, str):
            print(f"  TIMEOUT {name}")
            continue
        rows = []
        collect(data, rows)
        for r in rows:
            r["doc"] = name
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        print(f"  {name}: {len(rows)} sentenças")
    except Exception as e:
        print(f"  ERRO {name}: {type(e).__name__} {str(e)[:90]}")
out.close()
n = sum(1 for _ in open(OUT, encoding="utf-8"))
print(f"TOTAL acumulado: {n} sentenças")
