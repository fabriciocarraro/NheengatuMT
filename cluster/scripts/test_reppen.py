# -*- coding: utf-8 -*-
"""Prova numérica de que RepPenSoNaContinuacao faz o que promete:
(a) idêntico ao processor padrão do HF aplicado SÓ à continuação;
(b) diferente do padrão aplicado a prompt+continuação — que é o que
    aconteceria sem esta classe, e é o viés que ela remove."""
import sys, types, importlib.util
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent   # o módulo testado é irmão deste arquivo

# stub só do que o import do módulo exige
tr = types.ModuleType("transformers")
for a in ("AutoTokenizer", "AutoModelForCausalLM"): setattr(tr, a, object)
tr.LogitsProcessor = object
sys.modules["transformers"] = tr
sys.modules["sacrebleu"] = types.ModuleType("sacrebleu")
spec = importlib.util.spec_from_file_location("ec", AQUI / "evaluate_causal.py")
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)

def hf_padrao(input_ids, scores, penalty):
    """Reimplementação literal do RepetitionPenaltyLogitsProcessor do HF."""
    s = torch.gather(scores, 1, input_ids)
    s = torch.where(s < 0, s * penalty, s / penalty)
    return scores.scatter(1, input_ids, s)

torch.manual_seed(0)
V, B, NP, NG = 50, 3, 6, 4            # vocab, lote, tokens de prompt, gerados
PEN = 1.5
prompt = torch.randint(0, V, (B, NP))
gerado = torch.randint(0, V, (B, NG))
ids = torch.cat([prompt, gerado], 1)
scores = torch.randn(B, V)

meu = ec.RepPenSoNaContinuacao(PEN, NP)(ids, scores.clone())
ref_so_continuacao = hf_padrao(gerado, scores.clone(), PEN)
ref_tudo = hf_padrao(ids, scores.clone(), PEN)

falhas = []
def checa(nome, cond):
    print(f"  {'OK  ' if cond else 'FALHA'}  {nome}")
    if not cond: falhas.append(nome)

checa("igual ao HF aplicado só à continuação",
      torch.allclose(meu, ref_so_continuacao))
checa("DIFERENTE do HF aplicado a prompt+continuação (o viés que removemos)",
      not torch.allclose(meu, ref_tudo))

# quantos tokens sofrem castigo indevido no caso padrão
so_prompt = set(prompt.flatten().tolist()) - set(gerado.flatten().tolist())
n = sum(1 for b in range(B) for v in range(V)
        if abs(meu[b, v] - ref_tudo[b, v]) > 1e-6)
print(f"  info   {n} posições (lote x vocab) divergem = penalizações indevidas evitadas")

# primeiro passo de geração: nada gerado ainda, tem que ser no-op
vazio = ec.RepPenSoNaContinuacao(PEN, NP)(prompt, scores.clone())
checa("no primeiro passo (nada gerado) é no-op", torch.allclose(vazio, scores))

# sob beam search as linhas são (lote x beams); prompt continua do mesmo tamanho
ids_beam = ids.repeat_interleave(4, 0); sc_beam = scores.repeat_interleave(4, 0)
mb = ec.RepPenSoNaContinuacao(PEN, NP)(ids_beam, sc_beam.clone())
checa("consistente sob beam search (linhas replicadas)",
      torch.allclose(mb, meu.repeat_interleave(4, 0)))

# scores negativos: a regra do HF multiplica em vez de dividir
sc_neg = -torch.abs(torch.randn(B, V))
mn_ = ec.RepPenSoNaContinuacao(PEN, NP)(ids, sc_neg.clone())
checa("trata score negativo como o HF (multiplica, não divide)",
      torch.allclose(mn_, hf_padrao(gerado, sc_neg.clone(), PEN)))

print("\n" + ("PROCESSOR VALIDADO" if not falhas else f"FALHAS: {falhas}"))
sys.exit(1 if falhas else 0)
