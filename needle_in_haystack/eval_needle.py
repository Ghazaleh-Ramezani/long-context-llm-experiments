"""
Needle-in-a-Haystack Evaluation
================================
Tests a language model's ability to retrieve a specific fact (the "needle")
embedded at various positions inside a long document (the "haystack").

This is the standard diagnostic for long-context capability (Kamradt, 2023).
The key insight: models that struggle with long context fail specifically
when the needle is in the MIDDLE of the document — not at the start or end.

Output: a CSV + optional heatmap of accuracy vs (context_length, needle_depth).

Usage:
    # Against any HF model (or LoRA adapter):
    python eval_needle.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --context_lengths 1024 2048 4096 8192 \
        --depth_percents 10 25 50 75 90 \
        --n_runs 3 \
        --output_dir results/

    # Against a LoRA checkpoint:
    python eval_needle.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --adapter_path checkpoints/lora-llama3 \
        --context_lengths 4096 8192 \
        --output_dir results/lora/
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# The needle and question (swap these for domain-specific tests)
# ---------------------------------------------------------------------------

NEEDLE = (
    "The secret passcode for the research lab is: AURORA-7749."
)

QUESTION = "What is the secret passcode for the research lab?"

EXPECTED_ANSWER = "AURORA-7749"

# Filler text to build the haystack from
HAYSTACK_FILLER = (
    "The study of materials science encompasses a wide range of topics, "
    "including the structure, properties, processing, and performance of materials. "
    "Nanocellulose-based composites have shown promising results for strain sensing "
    "applications due to their high surface area and mechanical tunability. "
    "Reciprocal rank fusion is a robust method for combining heterogeneous ranked lists "
    "without requiring score calibration across retrieval systems. "
    "Bayesian optimization with Gaussian process surrogates provides sample-efficient "
    "exploration of high-dimensional design spaces for materials optimization. "
)


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------

def build_haystack(n_tokens: int, needle: str, depth_pct: float, tokenizer) -> str:
    """
    Construct a document of approximately n_tokens tokens with the needle
    inserted at depth_pct% of the way through.

    Args:
        n_tokens: Target total token count (approximate).
        needle: The fact to embed.
        depth_pct: Position of needle as a percentage (0=start, 100=end).
        tokenizer: HF tokenizer for token counting.

    Returns:
        Full document as a string.
    """
    # Repeat filler until we have enough tokens
    filler_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    budget = max(n_tokens - len(needle_tokens), 100)

    reps = (budget // len(filler_tokens)) + 2
    full_filler_tokens = (filler_tokens * reps)[:budget]

    # Split filler at the insertion point
    insert_at = int(len(full_filler_tokens) * depth_pct / 100)
    before = full_filler_tokens[:insert_at]
    after = full_filler_tokens[insert_at:]

    all_tokens = before + needle_tokens + after
    return tokenizer.decode(all_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_name: str, adapter_path: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path is not None:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


PROMPT_TEMPLATE = (
    "Read the following document carefully, then answer the question.\n\n"
    "Document:\n{document}\n\n"
    "Question: {question}\n\n"
    "Answer (be concise):"
)


def query_model(model, tokenizer, document: str, question: str, max_new_tokens: int = 50) -> str:
    prompt = PROMPT_TEMPLATE.format(document=document, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(response: str, expected: str) -> bool:
    """Check if the expected answer appears in the model response."""
    return expected.lower() in response.lower()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model, tokenizer = load_model(args.model_name, args.adapter_path, device)

    results = []

    for ctx_len in args.context_lengths:
        for depth in args.depth_percents:
            run_scores = []
            for run in range(args.n_runs):
                doc = build_haystack(ctx_len, NEEDLE, depth, tokenizer)
                actual_len = len(tokenizer.encode(doc, add_special_tokens=False))

                response = query_model(model, tokenizer, doc, QUESTION)
                correct = score_response(response, EXPECTED_ANSWER)
                run_scores.append(int(correct))

                logger.info(
                    f"ctx={ctx_len} depth={depth}% run={run+1}/{args.n_runs} "
                    f"correct={correct} | response: '{response[:80]}'"
                )

            accuracy = sum(run_scores) / len(run_scores)
            results.append({
                "model": args.model_name,
                "adapter": args.adapter_path or "none",
                "context_length": ctx_len,
                "depth_pct": depth,
                "accuracy": accuracy,
                "n_runs": args.n_runs,
            })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(output_dir, f"needle_results_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, f"needle_results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n=== Needle-in-Haystack Results ===")
    print(f"{'Context':>10} {'Depth%':>8} {'Accuracy':>10}")
    print("-" * 32)
    for r in results:
        print(f"{r['context_length']:>10} {r['depth_pct']:>8} {r['accuracy']:>10.2f}")


def plot_heatmap(results: list, output_dir: str):
    """Optional: heatmap of accuracy vs context_length × depth."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        ctx_lens = sorted(set(r["context_length"] for r in results))
        depths = sorted(set(r["depth_pct"] for r in results))

        matrix = np.zeros((len(depths), len(ctx_lens)))
        for r in results:
            i = depths.index(r["depth_pct"])
            j = ctx_lens.index(r["context_length"])
            matrix[i, j] = r["accuracy"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ctx_lens)))
        ax.set_xticklabels([f"{c//1024}k" for c in ctx_lens])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d}%" for d in depths])
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Needle Depth")
        ax.set_title("Needle-in-Haystack Accuracy")
        plt.colorbar(im, ax=ax, label="Accuracy")

        for i in range(len(depths)):
            for j in range(len(ctx_lens)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)

        path = os.path.join(output_dir, "heatmap.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        logger.info(f"Heatmap saved to {path}")
    except ImportError:
        logger.warning("matplotlib not installed — skipping heatmap")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--adapter_path", default=None,
                   help="Path to LoRA adapter (optional). If set, loads adapter on top of base model.")
    p.add_argument("--context_lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192])
    p.add_argument("--depth_percents", type=float, nargs="+",
                   default=[10, 25, 50, 75, 90])
    p.add_argument("--n_runs", type=int, default=3,
                   help="Number of runs per (context_length, depth) cell for averaging.")
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--plot", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args)
    save_results(results, args.output_dir)
    if args.plot:
        plot_heatmap(results, args.output_dir)
