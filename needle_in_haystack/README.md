# Needle-in-a-Haystack Evaluation

Measures a model's accuracy at retrieving a specific fact embedded at a chosen
depth in a long document, across context lengths.

## What this tests

The standard long-context diagnostic (Kamradt, 2023). Two failure patterns
read differently on the heatmap:

- **Lost in the middle** (Liu et al., 2023): accuracy dips at middle depths
  and recovers near the start and end of the document.
- **Context-length limit**: accuracy falls with length at every depth.

## How a cell is measured

- **Distinct trials.** Each run shuffles the filler sentences with its own
  seed, so with greedy decoding the `--n_runs` trials in a cell are different
  documents rather than one prompt repeated.
- **Sentence-boundary insertion.** The needle goes between sentences, at the
  boundary closest to the requested depth; the realised depth is recorded per
  run.
- **Chat template.** Instruct models are prompted through their chat template.
- **Strict scoring.** An answer is correct when the only passcode it names is
  the expected one: "The passcode is AURORA-7749." passes, "AURORA-7749 or
  BOREALIS-1234" does not.

## Usage

```bash
# Base model
python eval_needle.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --context_lengths 1024 2048 4096 8192 \
    --depth_percents 10 25 50 75 90 \
    --n_runs 3 --output_dir results/base/

# LoRA fine-tuned model
python eval_needle.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path checkpoints/lora-llama3 \
    --context_lengths 4096 8192 --output_dir results/lora/
```

## Output

- `needle_results.csv` / `.json`: accuracy per (context length, depth) cell
- `needle_runs.csv` / `.json`: every trial, with realised depth, prompt length and the raw response
- `heatmap.png`: accuracy over context length × needle depth (`--no-plot` to skip)

## Reading the heatmap

| Color | Meaning |
|-------|---------|
| Green | The model reliably retrieves the needle |
| Yellow | Retrieval is inconsistent across trials |
| Red | The model misses the needle |

A model with positional bias shows red in the middle-depth rows at long
context lengths; a model at its length limit shows red across whole columns.

## References

- Kamradt, G. (2023). [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- Liu et al. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
