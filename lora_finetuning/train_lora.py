"""
LoRA fine-tuning for long-context instruction following.

Fine-tunes a causal LM (default: Llama-3.2-1B-Instruct) on a long-context
QA dataset using PEFT LoRA. Designed to run on a single A100/T4 GPU (Google
Colab) with 4-bit quantization.

Key choices (each defensible in an interview):
  - LoRA rank r=16, alpha=32: r controls the intrinsic dimension of the
    weight update; alpha/r=2 is a standard scaling that approximates the
    learning-rate effect of full fine-tuning at rank r.
  - Target modules: q_proj, v_proj — empirically sufficient for instruction
    following; adding k_proj/o_proj gives marginal gain at higher memory cost.
  - 4-bit NF4 quantization (bitsandbytes): keeps base model at ~1 GB for 1B
    model, leaving headroom for the LoRA adapters + optimizer states.

Usage:
    python train_lora.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --dataset_name json \
        --data_path data/train.jsonl \
        --output_dir checkpoints/lora-llama3-lc \
        --max_seq_len 8192 \
        --lora_r 16 \
        --lora_alpha 32 \
        --num_epochs 3
"""

import argparse
import logging
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset_name", default="json")
    p.add_argument("--data_path", default="data/train.jsonl")
    p.add_argument("--output_dir", default="checkpoints/lora-run")
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank. Higher = more parameters, more expressivity.")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA scaling: effective lr scales as alpha/r.")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+",
                   default=["q_proj", "v_proj"],
                   help="Which weight matrices to apply LoRA to.")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--use_4bit", action="store_true", default=True,
                   help="Load base model in 4-bit (NF4) for memory efficiency.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (with optional 4-bit quantization)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # needed for causal LM training

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLMs
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,       # saves ~0.4 bits/param extra
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

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
    # Example output for r=16, q_proj+v_proj on Llama-3.2-1B:
    #   trainable params: 1,048,576 || all params: 1,236,924,416 || trainable%: 0.085%
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n{response}"
)


def preprocess(examples, tokenizer, max_len):
    texts = [
        PROMPT_TEMPLATE.format(
            instruction=i, context=c, response=r
        )
        for i, c, r in zip(
            examples["instruction"],
            examples.get("context", [""] * len(examples["instruction"])),
            examples["response"],
        )
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)
    model = build_lora_model(model, args)

    logger.info(f"Loading dataset from: {args.data_path}")
    raw = load_dataset(args.dataset_name, data_files={"train": args.data_path})
    dataset = raw["train"].map(
        lambda ex: preprocess(ex, tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
