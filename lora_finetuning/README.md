# LoRA Fine-Tuning for Long-Context Instruction Following

Fine-tunes a causal LM on long-context QA data using PEFT LoRA with 4-bit quantization.
Runs on a single GPU (tested on A100 40GB / T4 16GB via Google Colab).

## Design choices

| Choice | Value | Why |
|--------|-------|-----|
| LoRA rank `r` | 16 | Controls intrinsic dim of the weight update ΔW = BA, B∈ℝ^{d×r}, A∈ℝ^{r×k} |
| `lora_alpha` | 32 | Scaling factor: effective lr ∝ alpha/r = 2; matches full fine-tuning scale |
| Target modules | `q_proj`, `v_proj` | Sufficient for instruction following; adding k/o gives marginal gain |
| Quantization | 4-bit NF4 (bitsandbytes) | Keeps 1B model at ~1 GB; leaves headroom for adapters + optimizer |
| `padding_side` | `right` | Required for causal LM; left-padding causes label misalignment |

## Usage

```bash
pip install -r ../requirements.txt

python train_lora.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --data_path data/train.jsonl \
    --output_dir checkpoints/lora-llama3 \
    --max_seq_len 8192 \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 3
```

## Data format (`train.jsonl`)

```json
{"instruction": "Answer based on the document.", "context": "<long doc>", "response": "..."}
```

## Expected output

```
trainable params: 1,048,576 || all params: 1,236,924,416 || trainable%: 0.085%
```

The adapter is saved separately from the base model and loaded with `PeftModel.from_pretrained()`.
