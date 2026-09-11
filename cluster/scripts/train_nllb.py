# -*- coding: utf-8 -*-
"""
Fine-tuning bidirecional do NLLB-200 para PT<->Nheengatu (yrl_Latn) no cluster.

- Adiciona o token de língua yrl_Latn inicializado do embedding de grn_Latn (Guarani).
- Treina as duas direções no mesmo modelo (tag da língua-alvo no início dos labels).
- Early stopping por chrF++ na direção principal (por->yrl) sobre os dois devs;
  avaliação final completa (2 direções x 2 devs) via evaluate_yrl.py.
- Offline por padrão (definir HF_HUB_OFFLINE=1 no ambiente do job).

Uso típico:
  python train_nllb.py --model_dir $MODELS/nllb-200-distilled-600M \
      --data_dir $DATA/raw --out_dir $OUT/run1 \
      [--extra_train $DATA/raw/train.fra-yrl.jsonl] \
      [--gn_mix $DATA/gn/train.spa-grn.jsonl:0.1] [--lora] [--seed 1]
"""
import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback)
import sacrebleu

LANGS = ["por_Latn", "yrl_Latn", "fra_Latn", "spa_Latn", "grn_Latn", "eng_Latn"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--extra_train", nargs="*", default=[])
    p.add_argument("--gn_mix", default=None, help="arquivo.jsonl:proporcao")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--eval_steps", type=int, default=400)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--dev_subset", type=int, default=0, help="limitar geração de dev p/ caber no wallclock")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--new_lang", default="yrl_Latn")
    p.add_argument("--init_from", default="grn_Latn")
    p.add_argument("--annotate", default=None,
                   help="glossario.jsonl:rate — anotação inline de termos no lado PT "
                        "da direção por→yrl (treino com rate; dev interno com 1.0)")
    p.add_argument("--reg_tags", action="store_true",
                   help="prefixa tags de registro ([fala]/[escolar]) na FONTE das "
                        "linhas de grupos de registro, ambas as direções")
    return p.parse_args()

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

class PairDataset(Dataset):
    """Cada item: fonte com tag src_lang; labels = [tag_tgt] + alvo + </s>."""
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.lang_ids = {t: tok.convert_tokens_to_ids(t) for t in LANGS
                         if tok.convert_tokens_to_ids(t) != tok.unk_token_id}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        self.tok.src_lang = r["sl"]
        self.tok.tgt_lang = r["tl"]
        enc = self.tok(r["s"], truncation=True, max_length=self.max_len)
        lab = self.tok(text_target=r["t"], truncation=True, max_length=self.max_len)
        ids = lab["input_ids"]
        if ids and ids[0] in self.lang_ids.values():
            ids = ids[1:]
        enc["labels"] = [self.lang_ids[r["tl"]]] + ids
        return enc

REG_TAGS = (("refubium_fala", "[fala]"), ("leetra_", "[escolar]"))

def reg_tag(grupo):
    for pref, tag in REG_TAGS:
        if grupo.startswith(pref):
            return tag
    return None

def build_bi(rows, annot=None, tags=False):
    """Bidirecional; a anotação de termos aplica-se SÓ ao lado PT da direção
    por→yrl (a cópia reversa usa o src original, sem marcadores). Com
    tags=True, linhas de grupos de registro ganham prefixo NA FONTE de cada
    direção (nunca no alvo — o modelo não deve emitir tags)."""
    out = []
    for r in rows:
        s_fwd = r["src"]
        if annot is not None and r["src_lang"] == "por_Latn":
            s_fwd = annot(s_fwd)
        pre = ""
        if tags:
            t = reg_tag(r.get("grupo", "") or "")
            if t:
                pre = t + " "
        out.append({"sl": r["src_lang"], "tl": r["tgt_lang"], "s": pre + s_fwd, "t": r["tgt"]})
        out.append({"sl": r["tgt_lang"], "tl": r["src_lang"], "s": pre + r["tgt"], "t": r["src"]})
    return out

def add_language(tok, model, new_lang, init_from):
    if tok.convert_tokens_to_ids(new_lang) != tok.unk_token_id:
        print(f"[lang] {new_lang} já existe")
        return
    tok.add_special_tokens({"additional_special_tokens": [new_lang]}, replace_additional_special_tokens=False)
    emb_rows = model.get_input_embeddings().weight.shape[0]
    if len(tok) > emb_rows:
        model.resize_token_embeddings(len(tok))
        print(f"[lang] resize {emb_rows} -> {len(tok)}")
    else:
        print(f"[lang] sem resize (matriz {emb_rows} >= vocab {len(tok)})")
    with torch.no_grad():
        src_id = tok.convert_tokens_to_ids(init_from)
        new_id = tok.convert_tokens_to_ids(new_lang)
        model.get_input_embeddings().weight[new_id] = model.get_input_embeddings().weight[src_id].clone()
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
            out_emb.weight[new_id] = out_emb.weight[src_id].clone()
    print(f"[lang] {new_lang} (id {new_id}) inicializado de {init_from} (id {src_id})")

def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model_dir, use_fast=True)
    assert tok.is_fast, "Tokenizer lento do NLLB não suporta código de língua novo em src_lang — instale 'tokenizers'"
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_dir)
    add_language(tok, model, a.new_lang, a.init_from)
    yrl_id = tok.convert_tokens_to_ids(a.new_lang)

    annot_train = annot_dev = None
    if a.annotate:
        from term_annotate import Annotator
        gpath, grate = a.annotate.rsplit(":", 1)
        annot_train = Annotator(gpath, rate=float(grate), seed=a.seed)
        annot_dev = Annotator(gpath, rate=1.0, seed=0)
        print(f"[annot] terminologia inline: {gpath} rate={grate} "
              f"({len(annot_train.terms)} termos; dev interno com rate=1.0)")

    train_rows = load_jsonl(os.path.join(a.data_dir, "train.por-yrl.jsonl"))
    for f in a.extra_train:
        train_rows += load_jsonl(f)
    if a.reg_tags:
        n_tag = sum(1 for r in train_rows if reg_tag(r.get("grupo", "") or ""))
        print(f"[tags] tags de registro ativas: {n_tag} linhas prefixadas "
              f"({', '.join(t for _, t in REG_TAGS)})")
    bi = build_bi(train_rows, annot=annot_train, tags=a.reg_tags)
    if a.gn_mix:
        path, prop = a.gn_mix.rsplit(":", 1)
        gn = build_bi(load_jsonl(path))
        random.shuffle(gn)
        k = min(int(len(bi) * float(prop)), len(gn))
        bi += gn[:k]
        print(f"[mix] +{k} exemplos Guarani")
    random.shuffle(bi)
    print(f"[data] treino bidirecional: {len(bi)} exemplos")

    # dev para early stopping: SÓ direção por->yrl (forced_bos constante na geração)
    dev = (load_jsonl(os.path.join(a.data_dir, "dev_const.por-yrl.jsonl"))
           + load_jsonl(os.path.join(a.data_dir, "dev_extra.por-yrl.jsonl")))
    dev_fwd = [{"sl": r["src_lang"], "tl": r["tgt_lang"],
                "s": annot_dev(r["src"]) if annot_dev else r["src"],
                "t": r["tgt"]} for r in dev]
    if a.dev_subset and len(dev_fwd) > a.dev_subset:
        random.Random(0).shuffle(dev_fwd)
        dev_fwd = dev_fwd[:a.dev_subset]

    if a.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)
    args = Seq2SeqTrainingArguments(
        output_dir=a.out_dir, seed=a.seed,
        per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=max(8, a.bsz // 2),
        learning_rate=a.lr, warmup_steps=a.warmup, max_steps=a.max_steps,
        lr_scheduler_type="inverse_sqrt", weight_decay=0.01,
        label_smoothing_factor=0.0,
        logging_steps=50, eval_steps=a.eval_steps, eval_strategy="steps",
        save_steps=a.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="chrf",
        greater_is_better=True, predict_with_generate=True,
        generation_max_length=a.max_len, generation_num_beams=4,
        bf16=torch.cuda.is_available(), report_to=[],
    )
    model.generation_config.forced_bos_token_id = yrl_id

    refs = [r["t"] for r in dev_fwd]
    def compute_metrics(eval_pred):
        preds, _ = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        hyp = tok.batch_decode(preds, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(hyp, [refs[:len(hyp)]], word_order=2).score
        return {"chrf": chrf}

    trainer = Seq2SeqTrainer(
        model=model, args=args, data_collator=collator,
        train_dataset=PairDataset(bi, tok, a.max_len),
        eval_dataset=PairDataset(dev_fwd, tok, a.max_len),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)],
    )
    ckpt = None
    if a.auto_resume:
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(a.out_dir)
        if ckpt: print(f"[resume] retomando de {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    best = os.path.join(a.out_dir, "best")
    trainer.save_model(best)
    tok.save_pretrained(best)
    print(f"[done] melhor checkpoint em {best}")

if __name__ == "__main__":
    main()
