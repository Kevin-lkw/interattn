import math
from types import SimpleNamespace

import torch

from ..multisample.common import (
    ALIGNED_PROTOCOL,
    LEGACY_PROTOCOL,
    aggregate_samples,
    build_sample_context,
    effective_causal_budget,
)
from .merge_settings import merge_setting_summaries


def _context(protocol):
    encoded = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    return build_sample_context(
        model=SimpleNamespace(config=SimpleNamespace()),
        tokenizer=None,
        encoded=encoded,
        dtype=torch.float32,
        device="cpu",
        start=0,
        seq_len=4,
        ppl_protocol=protocol,
    )


def test_aligned_labels_stay_inside_chunk():
    ctx = _context(ALIGNED_PROTOCOL)
    assert ctx.inputs["input_ids"].tolist() == [[10, 11, 12, 13]]
    assert ctx.gt_label.tolist() == [[11, 12, 13]]


def test_legacy_labels_keep_cross_chunk_target():
    ctx = _context(LEGACY_PROTOCOL)
    assert ctx.gt_label.tolist() == [[11, 12, 13, 14]]


def test_budget_uses_only_scored_query_positions():
    budget = effective_causal_budget(
        seq_len=4,
        budget=0.5,
        num_layers=1,
        full_attention_layers=0,
        pos_list=[0, 1, 2],
    )
    assert math.isclose(budget, 5 / 6)


def test_corpus_ppl_is_token_weighted():
    summary = {
        "samples": {
            0: {
                "teacher_nll": math.log(2),
                "teacher_ppl": 2.0,
                "num_tokens": 1,
                "results": {
                    0.5: {
                        "student_nll": math.log(2),
                        "student_ppl": 2.0,
                        "num_tokens": 1,
                        "measured_budget": 0.5,
                    }
                },
            },
            1: {
                "teacher_nll": math.log(4),
                "teacher_ppl": 4.0,
                "num_tokens": 3,
                "results": {
                    0.5: {
                        "student_nll": math.log(4),
                        "student_ppl": 4.0,
                        "num_tokens": 3,
                        "measured_budget": 0.5,
                    }
                },
            },
        }
    }
    aggregate = aggregate_samples(summary)
    expected = math.exp((math.log(2) + 3 * math.log(4)) / 4)
    assert math.isclose(aggregate["teacher"]["corpus_ppl"], expected)
    assert math.isclose(
        aggregate["settings"][0.5]["corpus_ppl"],
        expected,
    )
    assert aggregate["teacher"]["num_tokens"] == 4


def test_setting_merge_preserves_samples_and_extends_grid():
    def summary(setting, student_nll):
        samples = {}
        for index, start in enumerate((0, 4)):
            samples[index] = {
                "start": start,
                "teacher_nll": math.log(2),
                "teacher_ppl": 2.0,
                "num_tokens": 3,
                "results": {
                    setting: {
                        "student_nll": student_nll,
                        "student_ppl": math.exp(student_nll),
                        "measured_budget": setting,
                        "num_tokens": 3,
                    }
                },
            }
        return {
            "method": "h2o",
            "config": {
                "ppl_protocol": ALIGNED_PROTOCOL,
                "seq_len": 4,
                "dtype": "bfloat16",
                "model": "test",
                "budgets": [setting],
            },
            "starts": [0, 4],
            "settings": [setting],
            "samples": samples,
        }

    merged = merge_setting_summaries(
        [summary(0.5, math.log(3)), summary(0.25, math.log(4))]
    )
    assert merged["settings"] == [0.25, 0.5]
    assert merged["config"]["budgets"] == [0.25, 0.5]
    assert merged["aggregate"]["num_completed_samples"] == 2
    assert set(merged["samples"][0]["results"]) == {0.25, 0.5}
