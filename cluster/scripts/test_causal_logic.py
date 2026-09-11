# -*- coding: utf-8 -*-
"""Testa a lógica de dados do train_causal.py sem torch/transformers reais.
O que pode dar errado no cluster e custar 2h de fila: masking do prompt,
truncamento, direção dos pares, e a contagem de exemplos.

Roda com o tokenizador REAL do Qwen (via `tokenizers`, que funciona local)."""
import sys, os, types, json, importlib.util
from pathlib import Path
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
# Portátil entre a máquina local e o cluster: nos dois layouts este script mora
# em <base>/scripts/ e os dados em <base>/data/ (local <base>=cluster, no cluster
# <base>=$ROOT). Nenhum caminho absoluto.
AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent
DADOS = BASE / "data" / "norm3"

# --- stubs mínimos para o import passar (transformers local está quebrado) ---
for nome in ("torch", "numpy", "transformers", "torch.utils", "torch.utils.data"):
    sys.modules.setdefault(nome, types.ModuleType(nome))
sys.modules["torch"].manual_seed = lambda s: None
sys.modules["torch"].tensor = lambda v, dtype=None: v      # o colador só empilha listas
sys.modules["torch"].long = "long"
sys.modules["numpy"].random = types.SimpleNamespace(seed=lambda s: None)
sys.modules["torch.utils.data"].Dataset = object
sys.modules["torch"].utils = sys.modules["torch.utils"]
sys.modules["torch.utils"].data = sys.modules["torch.utils.data"]
for a in ("AutoTokenizer", "AutoModelForCausalLM", "Trainer", "TrainingArguments",
          "DataCollatorForSeq2Seq", "LogitsProcessor", "TrainerCallback"):
    setattr(sys.modules["transformers"], a, object)

spec = importlib.util.spec_from_file_location("tc", AQUI / "train_causal.py")
tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)

class TokQwen:
    """Envelope do tokenizador real do Qwen na interface que o script usa."""
    def __init__(self, path):
        self.t = Tokenizer.from_file(str(path))
        self.eos_token_id = 151645
    def __call__(self, txt, add_special_tokens=False):
        return {"input_ids": self.t.encode(txt).ids}

# tokenizador REAL do Qwen — procura onde o modelo costuma estar nos dois
# ambientes; QWEN_TOK=/caminho/tokenizer.json sobrepõe.
CANDS = [BASE / "Qwen3.5-2B" / "tokenizer.json",
         BASE / "models" / "Qwen3.5-2B" / "tokenizer.json",
         Path(os.environ.get("QWEN_TOK", "/nao/existe"))]
TOKJSON = next((c for c in CANDS if c.is_file()), None)
if TOKJSON is None:
    sys.exit("[erro] tokenizer.json do Qwen não encontrado. Procurei em:\n  "
             + "\n  ".join(str(c) for c in CANDS)
             + "\nPasse o caminho em QWEN_TOK=/.../tokenizer.json")
print(f"[tok] {TOKJSON}")
tok = TokQwen(TOKJSON)
MAXLEN = 512
falhas = []
def checa(nome, cond):
    print(f"  {'OK  ' if cond else 'FALHA'}  {nome}")
    if not cond: falhas.append(nome)

rows = [json.loads(l) for l in open(DADOS / "train.por-yrl.jsonl", encoding="utf-8")]
extra = [json.loads(l) for l in open(DADOS / "train.fra-yrl.jsonl", encoding="utf-8")]
print(f"pares: train.por-yrl {len(rows)} + train.fra-yrl {len(extra)} (o baseline usa os dois)")
langs = {(r["src_lang"], r["tgt_lang"]) for r in rows + extra}
checa(f"PROMPT cobre as direções dos dados {sorted(langs)}", langs <= set(tc.PROMPT))
checa("o molde tem exatamente um {src} em cada direção",
      all(t.count("{src}") == 1 for t in tc.PROMPT.values()))

# o molde do treino e o da avaliação TÊM que ser byte a byte o mesmo
spec_e = importlib.util.spec_from_file_location("ec", AQUI / "evaluate_causal.py")
ec = importlib.util.module_from_spec(spec_e)
sys.modules["sacrebleu"] = types.ModuleType("sacrebleu")
spec_e.loader.exec_module(ec)
checa("molde de treino == molde de avaliação (senão o modelo vê prompt inédito)",
      tc.PROMPT == ec.PROMPT)

ex = tc.build_examples(rows + extra)
n = len(rows) + len(extra)
checa(f"exemplos = 2x pares ({len(ex)} = 2x{n})", len(ex) == 2 * n)
p2y = sum(1 for e in ex if e["h"].startswith("Traduza do português"))
y2p = sum(1 for e in ex if e["h"].startswith("Traduza do nheengatu para o português"))
f2y = sum(1 for e in ex if e["h"].startswith("Traduza do francês"))
y2f = sum(1 for e in ex if e["h"].startswith("Traduza do nheengatu para o francês"))
checa(f"1:1 por direção (p→y {p2y}/y→p {y2p}, f→y {f2y}/y→f {y2f})",
      p2y == y2p == len(rows) and f2y == y2f == len(extra))

print(f"\n=== invariantes nos {len(ex)} exemplos, um a um (tokenizador real) ===")
ds = tc.CausalPairs(ex, tok, max_len=MAXLEN)
sem_eos = longos = sem_sup = desalinhados = alvo_cortado = fonte_cortada = 0
for i in range(len(ds)):
    it = ds[i]
    ids, lab = it["input_ids"], it["labels"]
    if lab[-1] != tok.eos_token_id: sem_eos += 1
    if len(ids) > MAXLEN: longos += 1
    if all(x == -100 for x in lab): sem_sup += 1
    if not (len(ids) == len(lab) == len(it["attention_mask"])): desalinhados += 1
    n_sup = sum(1 for x in lab if x != -100)
    y_int = len(tok(ex[i]["y"])["input_ids"]) + 1
    if n_sup < y_int: alvo_cortado += 1
    # o prompt supervisionado tem que ser prefixo exato de ids
    if ids[len(ids) - n_sup:] != [x for x in lab if x != -100]: desalinhados += 1
checa("EOS é sempre o último label", sem_eos == 0)
checa(f"nenhuma sequência passa de max_len={MAXLEN}", longos == 0)
checa("nenhum exemplo fica sem alvo supervisionado (evita loss NaN)", sem_sup == 0)
checa("input_ids/labels/attention_mask alinhados", desalinhados == 0)
print(f"  info   alvos truncados: {alvo_cortado} ({100*alvo_cortado/len(ds):.3f}%) "
      f"— só quando nem alvo cabe em {MAXLEN}")

print("\n=== masking: exemplo 0 conferido na mão ===")
it = ds[0]
n_prompt = len(tok(ex[0]["h"] + ex[0]["s"] + ex[0]["c"])["input_ids"])
n_mask = sum(1 for x in it["labels"] if x == -100)
checa(f"prompt {n_prompt} tok, mascarados {n_mask}", n_mask == n_prompt)
sup = [x for x in it["labels"] if x != -100]
esp = tok(ex[0]["y"])["input_ids"] + [tok.eos_token_id]
checa(f"continuação supervisionada bate ({len(sup)} tok)", sup == esp)
checa("decodifica de volta no alvo original",
      tok.t.decode(sup[:-1]).strip() == ex[0]["y"].strip())

print("\n=== colador: padding à direita, sem supervisionar o preenchimento ===")
PAD = 151643
col = tc.Colador(PAD)
lote = [ds[i] for i in (0, 3, 7, 11)]          # comprimentos bem diferentes
b = col(lote)
n = max(len(x["input_ids"]) for x in lote)
checa(f"tudo empilhado no mesmo comprimento ({n})",
      all(len(l) == n for k in b for l in b[k]))
checa("preenchimento com pad_id no input_ids",
      all(l[len(o["input_ids"]):] == [PAD] * (n - len(o["input_ids"]))
          for l, o in zip(b["input_ids"], lote)))
checa("preenchimento MASCARADO no attention_mask",
      all(sum(l) == len(o["input_ids"]) for l, o in zip(b["attention_mask"], lote)))
checa("preenchimento com -100 nos labels (não vira alvo de loss)",
      all(set(l[len(o["input_ids"]):]) <= {-100} for l, o in zip(b["labels"], lote)))
checa("nenhum exemplo do lote fica 100% mascarado (loss NaN)",
      all(any(x != -100 for x in l) for l in b["labels"]))
checa("o conteúdo original não foi alterado",
      all(l[:len(o["input_ids"])] == o["input_ids"] for l, o in zip(b["input_ids"], lote)))

print("\n=== anotação inline (dieta do flagship) ===")
class AnnotFake:
    def __call__(self, s): return s.replace("Brasil", "Brasil (braziu)")
ex_a = tc.build_examples(rows[:50], annot=AnnotFake())
tem = [e for e in ex_a if "(braziu)" in e["s"]]
checa("anota só o lado PT da direção por->yrl",
      all(e["h"].startswith("Traduza do português") for e in tem) and len(tem) > 0)
checa("a cópia reversa nunca recebe marcador",
      not any("(braziu)" in e["y"] for e in ex_a))

print("\n" + ("TODOS OS TESTES PASSARAM" if not falhas else f"FALHAS: {falhas}"))
sys.exit(1 if falhas else 0)
