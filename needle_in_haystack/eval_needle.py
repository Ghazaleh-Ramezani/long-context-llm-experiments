"""
Needle-in-a-Haystack Evaluation
================================
Tests a language model's ability to retrieve a specific fact (the "needle")
embedded at a chosen depth inside a long document (the "haystack"), across
context lengths (Kamradt, 2023). Accuracy that dips at middle depths and
recovers at the edges is the "lost in the middle" pattern (Liu et al., 2023);
accuracy that falls with length at every depth is a context-length limit.

Each run builds a different haystack (filler sentences shuffled with the run's
seed), so with greedy decoding the runs in a cell are distinct trials rather
than repeats of one prompt. The needle is inserted between sentences, at the
sentence boundary closest to the requested depth.

An answer is correct when the only passcode it contains is the expected one:
"AURORA-7749" and "The passcode is AURORA-7749." pass; "AURORA-7749 or
BOREALIS-1234" and "BOREALIS-1234" do not.

Output: per-run and per-cell results as CSV and JSON, plus a heatmap of
accuracy vs (context_length, needle_depth).

Usage:
    python eval_needle.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --context_lengths 1024 2048 4096 8192 \
        --depth_percents 10 25 50 75 90 \
        --n_runs 3 --output_dir results/base/

    # With a LoRA adapter on top of the base model:
    python eval_needle.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --adapter_path checkpoints/lora-llama3 \
        --context_lengths 4096 8192 --output_dir results/lora/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NEEDLE = "The secret passcode for the research lab is: AURORA-7749."
QUESTION = "What is the secret passcode for the research lab?"
EXPECTED_ANSWER = "AURORA-7749"
PASSCODE_PATTERN = re.compile(r"\b[A-Za-z]+-\d{3,}\b")

FILLER_SENTENCES = [
    "The study of materials science encompasses the structure, properties, processing, "
    "and performance of materials.",
    "Nanocellulose-based composites have shown promising results for strain sensing "
    "because of their high surface area and mechanical tunability.",
    "Reciprocal rank fusion combines heterogeneous ranked lists without requiring score "
    "calibration across retrieval systems.",
    "Bayesian optimization with Gaussian process surrogates explores high-dimensional "
    "design spaces with few experiments.",
    "Thin films deposited by vacuum filtration often show anisotropic conductivity "
    "between the in-plane and through-plane directions.",
    "Tensile tests on polymer films are sensitive to humidity, strain rate and specimen "
    "geometry.",
    "Reduced graphene oxide restores part of the conductivity lost during oxidation, "
    "depending on the reducing agent.",
    "Cross-validation estimates generalisation error by holding out folds of the data in turn.",
]


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------


def build_haystack(n_tokens: int, needle: str, depth_pct: float, tokenizer, seed: int = 0):
    """Build a document of about `n_tokens` tokens with `needle` inserted at
    the sentence boundary closest to `depth_pct` percent of the way through.

    Returns (document, realised_depth_pct), where the realised depth is the
    share of filler tokens that precede the needle.
    """
    rng = random.Random(seed)
    needle_len = len(tokenizer.encode(needle, add_special_tokens=False))
    budget = max(n_tokens - needle_len, 1)

    sentences, lengths, total, pool = [], [], 0, []
    while True:
        if not pool:  # each pass uses every filler sentence once, in a fresh order
            pool = FILLER_SENTENCES[:]
            rng.shuffle(pool)
        s = pool.pop()
        n = len(tokenizer.encode(" " + s, add_special_tokens=False))
        if total + n > budget and sentences:
            break
        sentences.append(s)
        lengths.append(n)
        total += n

    target = total * depth_pct / 100
    cumulative, best_i, best_gap = 0, 0, abs(target)
    for i, n in enumerate(lengths, start=1):
        cumulative += n
        if abs(cumulative - target) < best_gap:
            best_i, best_gap = i, abs(cumulative - target)
    before = sentences[:best_i]
    after = sentences[best_i:]
    document = " ".join(before + [needle] + after)
    realised = 100 * sum(lengths[:best_i]) / max(total, 1)
    return document, realised


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def load_model(model_name: str, adapter_path: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    if adapter_path is not None:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


PROMPT_TEMPLATE = (
    "Read the following document carefully, then answer the question.\n\n"
    "Document:\n{document}\n\n"
    "Question: {question}\n\n"
    "Answer (be concise):"
)


def build_prompt_ids(tokenizer, document: str, question: str) -> torch.Tensor:
    """Use the model's chat template when it has one: an instruct model
    prompted without it answers a different distribution of inputs from the
    one it was tuned on."""
    content = PROMPT_TEMPLATE.format(document=document, question=question)
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt"
        )
        return ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
    return tokenizer(content, return_tensors="pt")["input_ids"]


def query_model(model, tokenizer, document: str, question: str, max_new_tokens: int = 32):
    input_ids = build_prompt_ids(tokenizer, document, question).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), int(input_ids.shape[1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_response(response: str, expected: str = EXPECTED_ANSWER) -> bool:
    """Correct iff the response names exactly one passcode and it is the
    expected one. A substring check would also accept a response that lists
    the right code alongside a wrong one."""
    codes = {c.upper() for c in PASSCODE_PATTERN.findall(response)}
    return codes == {expected.upper()}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    model, tokenizer = load_model(args.model_name, args.adapter_path, device)

    cells, runs = [], []
    for ctx_len in args.context_lengths:
        for depth in args.depth_percents:
            scores = []
            for run in range(args.n_runs):
                doc, realised = build_haystack(ctx_len, NEEDLE, depth, tokenizer, seed=run)
                response, prompt_tokens = query_model(model, tokenizer, doc, QUESTION)
                correct = score_response(response)
                scores.append(int(correct))
                runs.append(
                    {
                        "context_length": ctx_len,
                        "depth_pct": depth,
                        "realised_depth_pct": round(realised, 2),
                        "run": run,
                        "prompt_tokens": prompt_tokens,
                        "correct": int(correct),
                        "response": response,
                    }
                )
                logger.info(
                    "ctx=%d depth=%s%% run=%d/%d correct=%s | %r",
                    ctx_len, depth, run + 1, args.n_runs, correct, response[:80],
                )
            cells.append(
                {
                    "model": args.model_name,
                    "adapter": args.adapter_path or "none",
                    "context_length": ctx_len,
                    "depth_pct": depth,
                    "accuracy": sum(scores) / len(scores),
                    "n_runs": args.n_runs,
                }
            )
    return cells, runs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(cells: list, runs: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, rows in (("needle_results", cells), ("needle_runs", runs)):
        with open(os.path.join(output_dir, f"{name}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    logger.info("Results saved to %s", output_dir)

    print("\n=== Needle-in-Haystack Results ===")
    print(f"{'Context':>10} {'Depth%':>8} {'Accuracy':>10}")
    print("-" * 32)
    for r in cells:
        print(f"{r['context_length']:>10} {r['depth_pct']:>8} {r['accuracy']:>10.2f}")


def plot_heatmap(cells: list, output_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ctx_lens = sorted({r["context_length"] for r in cells})
    depths = sorted({r["depth_pct"] for r in cells})
    matrix = np.zeros((len(depths), len(ctx_lens)))
    for r in cells:
        matrix[depths.index(r["depth_pct"]), ctx_lens.index(r["context_length"])] = r["accuracy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ctx_lens)))
    ax.set_xticklabels([f"{c / 1024:g}k" if c >= 1024 else str(c) for c in ctx_lens])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d:g}%" for d in depths])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Needle depth")
    ax.set_title("Needle-in-a-haystack accuracy")
    plt.colorbar(im, ax=ax, label="Accuracy")
    for i in range(len(depths)):
        for j in range(len(ctx_lens)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    path = os.path.join(output_dir, "heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Heatmap saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--adapter_path", default=None, help="Optional LoRA adapter on top of the base model.")
    p.add_argument("--context_lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--depth_percents", type=float, nargs="+", default=[10, 25, 50, 75, 90])
    p.add_argument("--n_runs", type=int, default=3, help="Distinct haystacks per (length, depth) cell.")
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cells, runs = evaluate(args)
    save_results(cells, runs, args.output_dir)
    if args.plot:
        plot_heatmap(cells, args.output_dir)
    return cells, runs


if __name__ == "__main__":
    main()
