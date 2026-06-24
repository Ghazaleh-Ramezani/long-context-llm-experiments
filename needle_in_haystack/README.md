# Needle-in-a-Haystack Evaluation

Measures a model's retrieval accuracy for a specific fact embedded at varying positions and depths in a long document.

## What this tests

Standard long-context diagnostic (Kamradt, 2023). The key failure mode:
**models that struggle with long context fail specifically when the needle is in the middle of the document**, not at the start or end. This is consistent with the "lost-in-the-middle" phenomenon (Liu et al., 2023).

## Usage

```bash
# Base model
python eval_needle.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --context_lengths 1024 2048 4096 8192 \
    --depth_percents 10 25 50 75 90 \
    --n_runs 3 \
    --output_dir results/base/

# LoRA fine-tuned model
python eval_needle.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path checkpoints/lora-llama3 \
    --context_lengths 4096 8192 \
    --output_dir results/lora/
```

## Output

- `results/needle_results_<timestamp>.csv` — full per-cell accuracy table
- `results/needle_results_<timestamp>.json` — same in JSON
- `results/heatmap.png` — accuracy heatmap (context_length × needle_depth)

## Interpreting the heatmap

| Color | Meaning |
|-------|---------|
| Green | Model reliably finds the needle |
| Yellow | Inconsistent retrieval |
| Red | Model fails to retrieve the needle |

A well-functioning long-context model should be uniformly green across all depths.
A model with positional bias shows red in the middle rows (high depth%) at long context lengths.

## References

- Kamradt, G. (2023). [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- Liu et al. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
