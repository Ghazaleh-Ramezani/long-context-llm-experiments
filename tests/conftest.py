"""An offline tiny Llama and tokenizer, so the tests run on CPU with no Hub
access. The numbers such a model produces mean nothing; what the tests check
is how examples are built, labelled, padded and scored."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lora_finetuning"))
sys.path.insert(0, str(ROOT / "needle_in_haystack"))

SPECIALS = ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    import torch
    from eval_needle import FILLER_SENTENCES, NEEDLE, PROMPT_TEMPLATE, QUESTION
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        processors,
        trainers,
    )
    from train_lora import PROMPT_HEAD, PROMPT_TAIL
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    corpus = FILLER_SENTENCES + [NEEDLE, QUESTION, PROMPT_TEMPLATE, PROMPT_HEAD, PROMPT_TAIL] * 5
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=600,
            special_tokens=SPECIALS,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    bos = tok.token_to_id("<|begin_of_text|>")
    # Like Llama 3: BOS prepended when special tokens are requested, no EOS.
    tok.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        chat_template=CHAT_TEMPLATE,
        model_input_names=["input_ids", "attention_mask"],
    )
    out = tmp_path_factory.mktemp("tiny-llama")
    fast.save_pretrained(out)
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(out)
    return str(out)


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    tok.pad_token = tok.eos_token  # the PAD == EOS setup the training script uses for Llama
    return tok
