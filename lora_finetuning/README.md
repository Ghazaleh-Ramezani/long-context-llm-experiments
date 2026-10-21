# LoRA Fine-Tuning for Long-Context Instruction Following

Fine-tunes a causal LM on long-context QA data with PEFT LoRA, optionally on a
4-bit NF4 base (QLoRA). Sized for a single A100 40GB or T4 16GB.

## Design choices

| Choice | Value | Why |
|--------|-------|-----|
| LoRA rank `r` | 16 | Rank of the update ΔW = BA, B∈ℝ^{d×r}, A∈ℝ^{r×k} |
| `lora_alpha` | 32 | The update is scaled by alpha/r = 2; holding alpha/r fixed across ranks keeps the update magnitude, so the learning rate carries over |
| Target modules | `q_proj`, `v_proj` | The projections that decide what each position retrieves from the context |
| Loss | Response tokens only | Instruction and context are labelled -100; the gradient comes from producing the answer |
| Over-length examples | Context truncated from the end | Instruction, response and the closing EOS are always kept |
| EOS | Appended to every response and supervised | The model learns where to stop, with PAD and EOS sharing an id |
| Collator | `DataCollatorForSeq2Seq` | Pads labels with -100 and keeps the labels it is given |
| Quantization | 4-bit NF4, double quant | `--no-use_4bit` for full-precision LoRA or CPU runs |
| Precision | bf16 where supported, else fp16, fp32 on CPU | Chosen from the device |
| Gradient checkpointing | On, non-reentrant | Needed for 8k sequences; non-reentrant reaches every adapter under a frozen base |

With r=16 on `q_proj` and `v_proj`, Llama-3.2-1B has **1,703,936 trainable
parameters (0.138%)**: per layer, q_proj adds 16×(2048+2048) and v_proj adds
16×(2048+512), because grouped-query attention gives v_proj 8 KV heads of 64
dimensions. The base model has 1,235,814,400 parameters.

## Usage

```bash
pip install -r ../requirements.txt

python train_lora.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --data_path data/train.jsonl \
    --output_dir checkpoints/lora-llama3 \
    --max_seq_len 8192 --lora_r 16 --lora_alpha 32 --num_epochs 3
```

At start-up the script logs how many examples were kept, how many were
dropped because instruction plus response alone exceed `--max_seq_len`, and
how many had their context truncated.

## Data format (`train.jsonl`)

```json
{"instruction": "Answer based on the document.", "context": "<long doc>", "response": "..."}
```

The adapter is saved on its own and loaded with `PeftModel.from_pretrained()`.
