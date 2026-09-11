# -*- coding: utf-8 -*-
"""Fine-tuning causal (decoder-only) PT<->Nheengatu — comparador de ARQUITETURA
para o nosso NLLB seq2seq. Escrito para Qwen3.5-2B, serve para qualquer
AutoModelForCausalLM.

Diferenças de fundo em relação ao train_nllb.py, e por quê:

  1. NÃO adiciona token de língua. Um decoder-only não tem slot de língua; a
     direção vai no prompt, em texto. Isso também segue o achado do LLaMAX
     (arXiv 2407.05975) de que estender o vocabulário DEGRADA um LLM
     monotonicamente — logo, vocabulário original, sem resize.
  2. Loss só na continuação: o prompt (instrução + fonte) recebe -100 nos
     labels, então o modelo é penalizado apenas pelo que gera.
  3. Bidirecional como o nosso seq2seq: cada par vira dois exemplos
     (por->yrl e yrl->por), mantendo o mesmo orçamento de dados.
  4. lr default 2e-5 (faixa de LLM), NÃO o 1e-4 do seq2seq. Usar o do NLLB
     produziria um comparador fraco e legitimamente atacável.
  5. Pesos em fp32 + autocast bf16 (mixed precision de verdade). Carregar o
     modelo já em bf16 faria o AdamW atualizar parâmetros bf16 sem cópia-mestre
     fp32 — instabilidade conhecida que enfraqueceria o comparador de graça.
     Custa memória (~32 GB de estado) e cabe folgado nos 64 GB da H100.
  6. Dev de early stopping nas DUAS direções (o seq2seq usa só por->yrl, por
     causa do forced_bos na geração). Como reportamos as duas, enviesar a
     parada para uma delas penalizaria o comparador justo na direção em que
     ele tem chance real (yrl->pt).

Uso:
  python train_causal.py --model_dir $MODELS/qwen3.5-2b --data_dir $DATA/norm3 \
      --out_dir $ROOT/runs/TAG --max_steps 20000 --auto_resume
  # dieta do flagship (anotação 0,3 + token pairs), se for o caso:
  #   --extra_train $DATA/norm3/train.lex-yrl.jsonl --annotate work/glossario_fase2.jsonl:0.3
"""
import argparse, json, os, random, sys
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments)

# A instrução é sempre em português: é a língua-ponte do projeto e a que o
# modelo conhece bem. Mantida fixa — variar o prompt é um eixo experimental à
# parte. O molde é gerado a partir dos nomes para cobrir também o francês
# (train.fra-yrl.jsonl entra no baseline, logo tem de entrar aqui).
NOMES = {"por_Latn": "português", "yrl_Latn": "nheengatu", "fra_Latn": "francês"}

def molde(sl, tl):
    a, b = NOMES[sl], NOMES[tl]
    return f"Traduza do {a} para o {b}.\n{a.capitalize()}: {{src}}\n{b.capitalize()}:"

PROMPT = {(sl, tl): molde(sl, tl) for sl in NOMES for tl in NOMES if sl != tl}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_class", default="AutoModelForCausalLM",
                   help="classe do transformers a usar. Em checkpoints MULTIMODAIS, a "
                        "classe de texto puro carrega só o language model e ignora a "
                        "torre de visão — é o procedimento documentado pelo HF e o mesmo "
                        "que gerou o checkpoint Qwen3.5-2B-text-only. Ex.: "
                        "--model_class Qwen3_5ForCausalLM")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--extra_train", nargs="*", default=[],
                   help="jsonl extras concatenados ao treino (ex. train.lex-yrl.jsonl)")
    p.add_argument("--annotate", default=None,
                   help="glossario.jsonl:rate — anotação inline de termos no lado PT, "
                        "espelhando o --annotate do train_nllb.py")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)      # batch efetivo 32, igual ao seq2seq
    p.add_argument("--max_len", type=int, default=512)  # prompt+alvo numa sequência só;
    # 512 medido no corpus: mediana 50 tok, p99 230, e só 4/26.050 exemplos truncam.
    # Como o padding é dinâmico (paga-se a média, não o máximo), subir de 320 para
    # 512 não custa memória.
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--dev_subset", type=int, default=600,
                   help="amostra do dev para o eval_loss não dominar o relógio")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="desliga o gradient checkpointing (ligado por padrão: "
                        "troca ~30%% de tempo por memória)")
    return p.parse_args()

def kw_dtype(cls, dt):
    """`torch_dtype` virou `dtype` no transformers 5.x, e o antigo continua
    aceito só via **kwargs em algumas versões (onde vira no-op silencioso — o
    modelo carregaria em fp32 por acidente, ou pior, no dtype do checkpoint).
    Detecta o nome real em vez de apostar; funciona no 4.55 do venv principal e
    no 5.14 do venv causal."""
    import inspect
    par = inspect.signature(cls.from_pretrained).parameters
    return {"dtype": dt} if "dtype" in par else {"torch_dtype": dt}

def kw_fast(cls):
    """`use_fast` perdeu sentido no v5 (os tokenizers lentos foram removidos).
    Só passa se ainda for parâmetro nomeado."""
    import inspect
    return {"use_fast": True} if "use_fast" in inspect.signature(
        cls.from_pretrained).parameters else {}

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def split_prompt(sl, tl):
    """O molde partido em cabeça e cauda, para poder truncar SÓ a fonte."""
    return PROMPT[(sl, tl)].split("{src}")

def build_examples(rows, annot=None):
    """Cada par vira DOIS exemplos, como no build_bi() do seq2seq. A anotação
    de termos vale só para o lado PT da direção por->yrl (a cópia reversa usa o
    src original), exatamente como no train_nllb.py.

    Guarda cabeça/fonte/cauda separadas para que o truncamento caia sobre a
    fonte, e nunca sobre o alvo (ver CausalPairs)."""
    out = []
    for r in rows:
        sl, tl = r["src_lang"], r["tgt_lang"]
        s_fwd = r["src"]
        if annot is not None and sl == "por_Latn":
            s_fwd = annot(s_fwd)
        h, c = split_prompt(sl, tl)
        out.append({"h": h, "s": s_fwd, "c": c, "y": " " + r["tgt"]})
        h, c = split_prompt(tl, sl)
        out.append({"h": h, "s": r["tgt"], "c": c, "y": " " + r["src"]})
    return out

MIN_SRC = 8   # tokens de fonte que nunca se cortam, para o exemplo não virar ruído

class CausalPairs(Dataset):
    """prompt + alvo numa sequência; labels = -100 no prompt.

    O prompt é tokenizado INTEIRO, como na inferência: tokenizar cabeça, fonte e
    cauda em separado dá uma sequência diferente nas junções (o BPE funde através
    da fronteira) e treinaria o modelo numa tokenização que ele nunca vê ao
    traduzir. Medido: 64 tokens em pedaços contra 63 inteiro, já no 1º exemplo.

    Truncamento (0,004% dos exemplos, mas o modo de falha é caro): o alvo tem
    prioridade e o EOS é inegociável. Cortar o fim do alvo ensinaria o modelo a
    não parar — a mesma degeneração que já medimos na decodificação restrita,
    aqui autoinfligida. E um exemplo cujo alvo sumisse inteiro ficaria com labels
    todos -100: loss NaN se calhar de encher um batch. Por isso quem cede é a
    fonte, e só nesse caso raro se paga a tokenização em pedaços."""
    def __init__(self, ex, tok, max_len):
        self.ex, self.tok, self.max_len = ex, tok, max_len
    def __len__(self):
        return len(self.ex)
    def _ids(self, txt):
        return self.tok(txt, add_special_tokens=False)["input_ids"]
    def __getitem__(self, i):
        e = self.ex[i]
        pid = self._ids(e["h"] + e["s"] + e["c"])
        yid = self._ids(e["y"]) + [self.tok.eos_token_id]
        if len(pid) + len(yid) > self.max_len:               # caminho raro e seguro
            hid, cid = self._ids(e["h"]), self._ids(e["c"])
            disp = self.max_len - len(hid) - len(cid)        # sobra p/ fonte + alvo
            n_y = max(1, min(len(yid), disp - MIN_SRC))
            yid = yid[: n_y - 1] + [self.tok.eos_token_id]
            pid = hid + self._ids(e["s"])[: max(0, disp - len(yid))] + cid
        return {"input_ids": pid + yid, "attention_mask": [1] * (len(pid) + len(yid)),
                "labels": [-100] * len(pid) + yid}

class TrainerComMem(Trainer):
    """Separa custo FIXO (pesos+gradientes+AdamW) de custo por LOTE (ativações).

    O pico total não distingue os dois, e a decisão depende disso: se o fixo já
    domina, reduzir o batch não salva e o caminho é outro (kernels do DeltaNet,
    optimizer de 8 bits); se as ativações dominam, basta trocar batch por
    acumulação, sem mexer no batch efetivo nem no protocolo.

    Sobrescreve `training_step` em vez de usar TrainerCallback: a API de
    callbacks mudou no transformers 5.x e o `on_step_end` não disparou — o
    recibo simplesmente não saiu no log, que é o pior modo de falhar para uma
    instrumentação."""
    _vistos = set()
    ALVOS = (1, 2, 10, 50, 200)
    def training_step(self, *args, **kwargs):
        saida = super().training_step(*args, **kwargs)
        n = int(self.state.global_step)
        if torch.cuda.is_available() and n in self.ALVOS and n not in self._vistos:
            self._vistos.add(n)
            G = 2 ** 30
            print(f"[mem] passo {n:>3}: "
                  f"alocado {torch.cuda.memory_allocated()/G:5.1f} | "
                  f"pico alocado {torch.cuda.max_memory_allocated()/G:5.1f} | "
                  f"reservado {torch.cuda.max_memory_reserved()/G:5.1f} GB", flush=True)
        return saida

def diagnostico_kernels():
    """A doc do Qwen3.5 avisa que, sem causal_conv1d/fla, o caminho Gated
    DeltaNet cai EM SILÊNCIO em ops PyTorch mais lentas e mais famintas. Aqui a
    ausência deixa de ser silenciosa."""
    import importlib.util as iu
    faltam = [m for m in ("causal_conv1d", "fla") if iu.find_spec(m) is None]
    if faltam:
        print(f"[kernels] AUSENTES: {', '.join(faltam)} — o Gated DeltaNet vai usar "
              f"o fallback PyTorch (mais lento e mais faminto de memória).", flush=True)
    else:
        print("[kernels] causal_conv1d e fla presentes", flush=True)

class Colador:
    """Padding à direita; labels preenchidos com -100.

    Explícito de propósito. O DataCollatorForSeq2Seq funciona aqui, mas o que
    ele faz com um modelo causal (ordem do padding de labels, uso do
    model.prepare_decoder_input_ids_from_labels) já mudou entre versões do
    transformers. São 10 linhas para não depender disso num cluster onde um
    erro custa duas horas de fila."""
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, lote):
        n = max(len(x["input_ids"]) for x in lote)
        saida = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in lote:
            f = n - len(x["input_ids"])
            saida["input_ids"].append(x["input_ids"] + [self.pad_id] * f)
            saida["attention_mask"].append(x["attention_mask"] + [0] * f)
            saida["labels"].append(x["labels"] + [-100] * f)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in saida.items()}

def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    # Primeira coisa: quanto a GPU TEM DE LIVRE, visto de dentro deste processo.
    # O nvidia-smi do job mostra o total, que não distingue "placa ocupada por
    # processo vazado" de "alocador quebrado" — os dois dão OOM em 7,5 GB.
    if torch.cuda.is_available():
        livre, total = torch.cuda.mem_get_info()
        print(f"[gpu] {torch.cuda.get_device_name(0)} | livre {livre/2**30:.1f} "
              f"de {total/2**30:.1f} GB | dispositivos visíveis: {torch.cuda.device_count()}",
              flush=True)
    else:
        print("[gpu] CUDA indisponível neste processo", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_dir, **kw_fast(AutoTokenizer))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Recibo do tokenizador: as contas do pré-registro (fertilidade 2,74 no yrl,
    # teto de 196 tokens, p99 de comprimento) foram feitas sobre um vocabulário
    # de 248.044. Se este checkpoint trouxer outro, aquelas contas não valem.
    print(f"[tok] vocab {len(tok):,} | eos {tok.eos_token_id} ({tok.eos_token!r}) "
          f"| pad {tok.pad_token_id} ({tok.pad_token!r})", flush=True)
    # fp32 explícito + autocast bf16 do Trainer: ver nota 5 do cabeçalho. Explícito
    # de propósito — se um dia o default do from_pretrained virar "auto", o modelo
    # carregaria em bf16 e o AdamW passaria a atualizar pesos bf16 sem cópia-mestre,
    # em silêncio.
    import transformers
    Cls = getattr(transformers, a.model_class, None)
    if Cls is None:
        sys.exit(f"[erro] transformers {transformers.__version__} não expõe "
                 f"'{a.model_class}'")
    # RETOMADA: descoberta do checkpoint AQUI, antes de criar o modelo, de
    # propósito. O `Trainer` restaura o state_dict por um caminho de baixo nível
    # que NÃO aplica a conversão de nomes que o `from_pretrained` aplica. Com o
    # Qwen3.5 o efeito é total: o `save_pretrained` grava as chaves como
    # `model.language_model.*` e o objeto de texto puro espera `model.*`, então
    # a interseção é VAZIA — nada é restaurado e o treino recomeça do modelo
    # BASE, em silêncio, a cada janela de fila. Medido em 06/08/2026: a loss
    # saltou 0,27 -> 3,67 nas duas retomadas, com LR e época contínuas, e três
    # seeds perderam 7.000 passos cada. Solução: os PESOS vêm do checkpoint via
    # from_pretrained (que sabe converter); ao Trainer sobra o que ele restaura
    # bem — otimizador, agendador, RNG e contador de passos.
    ckpt = None
    if a.auto_resume and os.path.isdir(a.out_dir):
        # job morto NO MEIO de uma gravação deixa o checkpoint mais novo
        # truncado; recua para o anterior em vez de derrubar a submissão.
        for c in sorted((d for d in os.listdir(a.out_dir) if d.startswith("checkpoint-")),
                        key=lambda d: int(d.split("-")[1]), reverse=True):
            caminho = os.path.join(a.out_dir, c)
            if os.path.isfile(os.path.join(caminho, "trainer_state.json")):
                ckpt = caminho
                break
            print(f"[resume] {c} incompleto (sem trainer_state.json) — ignorado", flush=True)

    origem = ckpt or a.model_dir
    try:
        model, info = Cls.from_pretrained(origem, output_loading_info=True,
                                          **kw_dtype(Cls, torch.float32))
    except TypeError:                       # versão sem output_loading_info
        model, info = Cls.from_pretrained(origem, **kw_dtype(Cls, torch.float32)), {}
    # TRAVA: a biblioteca é o juiz de quais pesos entraram. Peso amarrado
    # (lm_head <-> embed_tokens) aparece como ausente e é legítimo; qualquer
    # outro ausente significa treinar a partir do base achando que se retomou.
    faltando = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    print(f"[modelo] pesos de: {origem} | ausentes={len(info.get('missing_keys', []))} "
          f"inesperados={len(info.get('unexpected_keys', []))}", flush=True)
    if faltando:
        sys.exit(f"[erro] {len(faltando)} pesos NÃO foram carregados de {origem} "
                 f"(ex.: {faltando[:5]}). Abortando: treinar assim recomeçaria do "
                 f"modelo base em silêncio, que é exatamente o bug de 06/08/2026.")
    # Recibo de parâmetros: é ele que prova que a torre de visão ficou de fora
    # num checkpoint multimodal (Qwen3.5-2B: 2,21 B completo x 1,88 B só texto).
    n_par = sum(p_.numel() for p_ in model.parameters())
    print(f"[modelo] {a.model_class} | {n_par/1e9:.3f} B parâmetros | "
          f"{type(model).__name__}", flush=True)
    if any(k in type(model).__name__.lower() for k in ("conditionalgeneration", "vision")):
        print("[modelo] AVISO: a classe carregada parece multimodal — a torre de visão "
              "não recebe gradiente nenhum aqui e ainda assim consome estado do AdamW. "
              "Use a classe de texto puro (--model_class).", flush=True)
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # confirmar que pegou: nem toda arquitetura implementa, e falhar em silêncio
    # aqui explicaria sozinho um pico de memória fora da conta
    print(f"[grad_ckpt] pedido={not a.no_grad_ckpt} | "
          f"ativo={getattr(model, 'is_gradient_checkpointing', 'desconhecido')}", flush=True)
    diagnostico_kernels()

    annot_train = annot_dev = None
    if a.annotate:
        from term_annotate import Annotator
        gpath, grate = a.annotate.rsplit(":", 1)
        annot_train = Annotator(gpath, rate=float(grate), seed=a.seed)
        annot_dev = Annotator(gpath, rate=1.0, seed=0)
        print(f"[annot] terminologia inline: {gpath} rate={grate} "
              f"({len(annot_train.terms)} termos; dev interno com rate=1.0)", flush=True)

    train_rows = load_jsonl(os.path.join(a.data_dir, "train.por-yrl.jsonl"))
    for f in a.extra_train:
        train_rows += load_jsonl(f)
    # falha aqui, e não dentro de um worker do DataLoader duas horas depois
    faltando = {(r["src_lang"], r["tgt_lang"]) for r in train_rows} - set(PROMPT)
    if faltando:
        sys.exit(f"[erro] sem molde de prompt para {faltando}; acrescente em NOMES")
    ex_train = build_examples(train_rows, annot=annot_train)
    random.shuffle(ex_train)

    dev_rows = (load_jsonl(os.path.join(a.data_dir, "dev_const.por-yrl.jsonl"))
                + load_jsonl(os.path.join(a.data_dir, "dev_extra.por-yrl.jsonl")))
    ex_dev = build_examples(dev_rows, annot=annot_dev)
    if a.dev_subset and len(ex_dev) > a.dev_subset:
        random.Random(0).shuffle(ex_dev)
        ex_dev = ex_dev[: a.dev_subset]

    train = CausalPairs(ex_train, tok, a.max_len)
    dev = CausalPairs(ex_dev, tok, a.max_len)
    print(f"[data] treino {len(train)} exemplos bidirecionais "
          f"({len(train_rows)} pares) | dev {len(dev)}", flush=True)

    args = TrainingArguments(
        output_dir=a.out_dir, seed=a.seed,
        per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=max(4, a.bsz // 2),
        learning_rate=a.lr, warmup_steps=a.warmup, max_steps=a.max_steps,
        lr_scheduler_type="cosine",          # padrão de LLM, não inverse-sqrt do seq2seq
        weight_decay=0.01, bf16=torch.cuda.is_available(), optim="adamw_torch_fused",
        logging_steps=50, eval_strategy="steps", eval_steps=a.eval_steps,
        save_strategy="steps", save_steps=a.eval_steps, save_total_limit=2,
        # o baseline escolhe o melhor checkpoint por chrF GERADO
        # (predict_with_generate); aqui é eval_loss, porque beam search sobre o
        # dev a cada 1.000 passos num modelo de 2B não cabe na janela de 2h.
        # Assimetria declarada no §5b do pré-registro: selecionar por loss tende
        # a ser LIGEIRAMENTE PIOR para o chrF, ou seja, é conservador contra o
        # comparador — nunca a favor dele.
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to=[],
    )
    trainer = TrainerComMem(model=model, args=args, train_dataset=train,
                            eval_dataset=dev, data_collator=Colador(tok.pad_token_id))

    # `ckpt` já foi descoberto lá em cima, antes de o modelo ser criado, e os
    # pesos já vieram dele. Aqui o Trainer restaura só otimizador, agendador,
    # RNG e contador de passos — a parte que ele restaura bem.
    if ckpt:
        print(f"[resume] otimizador/agendador/passo de {ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=ckpt)
    # Pico de memória: o caminho Gated DeltaNet cai em ops PyTorch lentas e mais
    # famintas quando causal_conv1d/fla não estão instalados, e faz isso EM
    # SILÊNCIO (doc do HF). Este número é o que denuncia.
    if torch.cuda.is_available():
        print(f"[mem] pico reservado: {torch.cuda.max_memory_reserved()/2**30:.1f} GB "
              f"de {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB",
              flush=True)
    best = os.path.join(a.out_dir, "best")
    trainer.save_model(best)
    tok.save_pretrained(best)
    print(f"[done] melhor checkpoint em {best}", flush=True)

if __name__ == "__main__":
    main()
