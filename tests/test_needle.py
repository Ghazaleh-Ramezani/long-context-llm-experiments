"""Haystack construction, scoring, and an end-to-end run on a tiny model."""

from __future__ import annotations

import json

import pytest
from eval_needle import NEEDLE, build_haystack, main, score_response


@pytest.mark.parametrize(
    "response, correct",
    [
        ("AURORA-7749", True),
        ("The passcode is AURORA-7749.", True),
        ("aurora-7749", True),
        ("AURORA-7749 or BOREALIS-1234", False),
        ("It is BOREALIS-1234, not AURORA-7749.", False),
        ("BOREALIS-1234", False),
        ("I could not find a passcode.", False),
        ("AURORA-77490", False),
    ],
)
def test_scorer_requires_exactly_the_expected_code(response, correct):
    assert score_response(response) is correct


@pytest.mark.parametrize("depth", [0, 10, 25, 50, 75, 90, 100])
def test_needle_lands_near_the_requested_depth(tokenizer, depth):
    doc, realised = build_haystack(1024, NEEDLE, depth, tokenizer, seed=0)
    assert doc.count(NEEDLE) == 1
    assert abs(realised - depth) < 3  # within one filler sentence of the target
    n = len(tokenizer.encode(doc, add_special_tokens=False))
    assert 900 <= n <= 1040


def test_runs_use_different_haystacks(tokenizer):
    docs = {build_haystack(512, NEEDLE, 50, tokenizer, seed=s)[0] for s in range(3)}
    assert len(docs) == 3


def test_evaluation_writes_results(tiny_model_dir, tmp_path):
    cells, runs = main(
        [
            "--model_name", tiny_model_dir, "--context_lengths", "64", "128",
            "--depth_percents", "10", "90", "--n_runs", "2", "--output_dir", str(tmp_path), "--no-plot",
        ]
    )
    assert len(cells) == 4 and len(runs) == 8
    saved = json.loads((tmp_path / "needle_results.json").read_text())
    assert [c["accuracy"] for c in saved] == [c["accuracy"] for c in cells]
    assert {r["run"] for r in runs} == {0, 1}
