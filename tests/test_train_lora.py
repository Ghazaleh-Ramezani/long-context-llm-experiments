"""Example construction, labels and padding for LoRA fine-tuning."""

from __future__ import annotations

import json

import pytest
from train_lora import IGNORE_INDEX, build_collator, build_example, main

LONG_DOC = " ".join(["Tensile tests on polymer films are sensitive to humidity."] * 400)


def test_only_the_response_is_supervised(tokenizer):
    ex = build_example("Answer from the document.", "The film is 95 um thick.", "95 um", tokenizer, 512)
    resp = tokenizer("95 um", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    supervised = [t for t in ex["labels"] if t != IGNORE_INDEX]
    assert supervised == resp
    # Prompt positions are masked; the supervised tail is the response itself.
    n_prompt = len(ex["labels"]) - len(resp)
    assert ex["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert ex["input_ids"][n_prompt:] == resp


def test_eos_stays_supervised_when_pad_is_eos(tokenizer):
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    short = build_example("Q", "short", "yes", tokenizer, 512)
    long = build_example("Q", "a much longer context than the other one", "no", tokenizer, 512)
    batch = build_collator(tokenizer)([short, long])
    eos = tokenizer.eos_token_id
    labels, ids = batch["labels"], batch["input_ids"]
    # The short example is padded with EOS ids; the padding is masked but its
    # real, final EOS keeps its label.
    n = len(short["input_ids"])
    assert labels[0, n - 1].item() == eos
    assert (labels[0, n:] == IGNORE_INDEX).all()
    assert (ids[0, n:] == eos).all()
    assert labels[1, len(long["input_ids"]) - 1].item() == eos


def test_long_context_is_truncated_and_the_answer_kept(tokenizer):
    ex = build_example("Answer from the document.", LONG_DOC, "humidity", tokenizer, 256)
    resp = tokenizer("humidity", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    assert len(ex["input_ids"]) == 256
    assert ex["input_ids"][-len(resp):] == resp
    assert [t for t in ex["labels"] if t != IGNORE_INDEX] == resp
    assert ex["context_truncated"] == 1


def test_example_that_cannot_fit_is_dropped(tokenizer):
    assert build_example("Q", "", "x " * 300, tokenizer, 64) is None


def test_training_runs_end_to_end_on_cpu(tiny_model_dir, tmp_path):
    rows = [
        {
            "instruction": "Answer from the document.",
            "context": LONG_DOC[: 200 + 40 * i],
            "response": "humidity",
        }
        for i in range(8)
    ] + [{"instruction": "Q", "context": "", "response": "x " * 400}]  # too long: dropped
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "adapter"
    result = main(
        [
            "--model_name", tiny_model_dir, "--data_path", str(data), "--output_dir", str(out),
            "--no-use_4bit", "--max_seq_len", "128", "--max_steps", "3", "--grad_accum_steps", "1",
            "--per_device_batch_size", "2", "--lr", "1e-3", "--target_modules", "q_proj", "v_proj",
        ]
    )
    assert result.global_step == 3
    assert (out / "adapter_config.json").exists()


def test_4bit_requires_cuda(tiny_model_dir, tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(SystemExit, match="needs a CUDA device"):
        main(["--model_name", tiny_model_dir, "--data_path", str(tmp_path / "x.jsonl")])
