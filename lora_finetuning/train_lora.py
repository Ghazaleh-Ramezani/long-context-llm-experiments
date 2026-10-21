"""
LoRA fine-tuning for long-context instruction following.

Fine-tunes a causal LM (default: Llama-3.2-1B-Instruct) on a long-context QA
dataset with PEFT LoRA, optionally on a 4-bit NF4 base (QLoRA). Sized for a
single A100 or T4.

Design choices:
  - LoRA rank r=16, alpha=32. The adapter's update is scaled by alpha/r, so
    holding alpha/r fixed while sweeping r keeps the update magnitude roughly
    constant and the learning rate need not be retuned for each rank.
  - Target modules q_proj, v_proj: the attention projections that decide what
    each position retrieves from the context, at the lowest adapter cost.
  - Loss on the response only. Instruction and context tokens are labelled
    -100, so the gradient comes from producing the answer rather than from
    modelling an 8k-token document.
  - Context truncation, never answer truncation. When an example is longer
    than --max_seq_len, tokens are cut from the end of the context; the
    instruction, the response and the closing EOS are always kept.
  - An explicit EOS is appended to every response and kept in the labels, so
    the model learns where to stop even when PAD and EOS share an id.

Usage:
    python train_lora.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --data_path data/train.jsonl \
        --output_dir checkpoints/lora-llama3-lc \
        --max_seq_len 8192 --lora_r 16 --lora_alpha 32 --num_epochs 3
"""

from __future__ import annotations

import argparse
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# The prompt is assembled from token pieces so the context can be truncated
# without touching the instruction or the response.
PROMPT_HEAD = "### Instruction:\n{instruction}\n\n### Context:\n"
PROMPT_TAIL = "\n\n### Response:\n"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset_name", default="json")
    p.add_argument("--data_path", default="data/train.jsonl")
    p.add_argument("--output_dir", default="checkpoints/lora-run")
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=32, help="Update is scaled by alpha/r.")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--num_epochs", type=float, default=3)
    p.add_argument("--max_steps", type=int, default=-1, help="Overrides num_epochs when > 0.")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--use_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit NF4 (needs CUDA and bitsandbytes). --no-use_4bit to disable.",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute activations in the backward pass; needed for 8k sequences on one GPU.",
    )
    p.add_argument("--trust_remote_code", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _precision() -> dict:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"dtype": torch.bfloat16, "bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"dtype": torch.float16, "bf16": False, "fp16": True}
    return {"dtype": torch.float32, "bf16": False, "fp16": False}


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prec = _precision()
    # Non-reentrant checkpointing: with a frozen base, the reentrant variant
    # only back-propagates into blocks whose inputs require grad.
    gc_kwargs = {"use_reentrant": False}
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--use_4bit needs a CUDA device; pass --no-use_4bit on CPU.")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=prec["dtype"],
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=gc_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=prec["dtype"],
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=args.trust_remote_code,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def build_lora_model(model, args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_example(instruction: str, context: str, response: str, tokenizer, max_len: int):
    """Tokenise one example. Returns None when the instruction and response
    alone exceed `max_len`, since no amount of context truncation fits it."""
    head = tokenizer(PROMPT_HEAD.format(instruction=instruction), add_special_tokens=True)["input_ids"]
    ctx = tokenizer(context or "", add_special_tokens=False)["input_ids"]
    tail = tokenizer(PROMPT_TAIL, add_special_tokens=False)["input_ids"]
    resp = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    room = max_len - len(head) - len(tail) - len(resp)
    if room < 0:
        return None
    truncated = int(len(ctx) > room)
    prompt = head + ctx[:room] + tail
    return {
        "input_ids": prompt + resp,
        "attention_mask": [1] * (len(prompt) + len(resp)),
        "labels": [IGNORE_INDEX] * len(prompt) + resp,
        "context_truncated": truncated,
    }


def preprocess(examples, tokenizer, max_len):
    n = len(examples["instruction"])
    contexts = examples["context"] if "context" in examples else [""] * n
    out = {"input_ids": [], "attention_mask": [], "labels": [], "context_truncated": []}
    for instruction, context, response in zip(examples["instruction"], contexts, examples["response"]):
        ex = build_example(instruction, context, response, tokenizer, max_len)
        if ex is None:
            continue
        for k in out:
            out[k].append(ex[k])
    return out


def build_dataset(path: str, tokenizer, max_len: int, dataset_name: str = "json"):
    raw = load_dataset(dataset_name, data_files={"train": path})["train"]
    data = raw.map(
        lambda ex: preprocess(ex, tokenizer, max_len),
        batched=True,
        remove_columns=raw.column_names,
    )
    dropped = len(raw) - len(data)
    truncated = sum(data["context_truncated"])
    logger.info(
        "Dataset: %d examples kept, %d dropped (instruction + response > max_seq_len), "
        "%d with context truncated",
        len(data),
        dropped,
        truncated,
    )
    return data.remove_columns("context_truncated")


def build_collator(tokenizer):
    # Pads input_ids with pad_token_id and labels with -100, and leaves the
    # labels it is given untouched. DataCollatorForLanguageModeling instead
    # rebuilds labels from input_ids and masks every pad id -- which, with
    # PAD == EOS, also masks the real EOS.
    return DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX, padding=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    logger.info("Loading model: %s", args.model_name)
    model, tokenizer = load_model_and_tokenizer(args)
    model = build_lora_model(model, args)

    logger.info("Loading dataset from: %s", args.data_path)
    dataset = build_dataset(args.data_path, tokenizer, args.max_seq_len, args.dataset_name)

    prec = _precision()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=build_collator(tokenizer),
    )

    logger.info("Starting training...")
    result = trainer.train()

    logger.info("Saving adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return result


if __name__ == "__main__":
    main()
