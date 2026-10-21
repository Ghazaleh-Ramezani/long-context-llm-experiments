# Long-Context LLM Experiments

Practical experiments on long-context LLM capability: LoRA fine-tuning and
positional-retrieval evaluation via the needle-in-a-haystack protocol.

## Contents

| Directory | What it does |
|-----------|-------------|
| `lora_finetuning/` | LoRA fine-tuning with PEFT + 4-bit quantization on long-context QA |
| `needle_in_haystack/` | Needle-in-haystack eval: accuracy vs. context length and needle depth |

## Motivation

Two concrete questions drove these experiments:

1. **Can LoRA efficiently adapt a pretrained model to follow instructions over very long contexts?**
   Standard fine-tuning at 8k+ token sequences is memory-prohibitive on a single GPU.
   LoRA (rank 16, alpha 32, targeting q_proj + v_proj; 1.7M trainable parameters on
   Llama-3.2-1B) combined with 4-bit NF4 quantization makes this tractable on a T4/A100.

2. **Where does a long-context model actually fail?**
   The needle-in-haystack protocol isolates positional retrieval from reasoning,
   letting you identify whether failures are due to context length per se or to
   the "lost-in-the-middle" effect (Liu et al., 2023).

## Quick start

```bash
pip install -r requirements.txt

# Fine-tune
python lora_finetuning/train_lora.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --data_path data/train.jsonl \
    --output_dir checkpoints/lora-run

# Evaluate long-context retrieval
python needle_in_haystack/eval_needle.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --context_lengths 1024 2048 4096 8192 \
    --output_dir results/
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q        # ~15 s on CPU, no GPU or Hub access needed
```

The tests build a tiny Llama-architecture model and tokenizer locally and
check what each script feeds the model: only response tokens and the closing
EOS are supervised, padding is masked while the real EOS keeps its label,
over-length examples lose context rather than the answer, the needle lands at
the requested depth, trials in a cell use distinct haystacks, and the scorer
accepts an answer only when the expected passcode is the one it names. Both
scripts also run end to end on that model.

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023). [QLoRA: Efficient Fine-Tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Liu et al. (2023). [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Kamradt (2023). [Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
