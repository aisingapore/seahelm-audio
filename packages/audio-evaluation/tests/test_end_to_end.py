"""Golden-file test: the six real pipeline rows scored exactly as the toolkit scores them.

Expected values were computed independently with the toolkit's own functions on the same
rows; this freezes them so a port drift fails loudly. No parquet, no network.
"""

import json
from pathlib import Path

import pytest
from audio_evaluation.score import (
    join_report,
    report_problems,
    score_rows,
    validate_rows,
)
from audio_evaluation.summary import summarize
from audio_evaluation.tasks.registry import TASKS

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "golden.json").read_text(encoding="utf-8")
)


def _run(task: str):
    spec = TASKS[task]
    rows, gold = GOLDEN[task]["rows"], GOLDEN[task]["gold"]
    assert validate_rows(rows) == []
    assert report_problems(join_report(rows, gold, spec)) == []
    records, config_ids = score_rows(rows, gold, spec)
    return records, summarize(records, spec, config_ids), spec


def test_mmlu_lite_zh_rows():
    records, summary, _ = _run("mmlu_lite")
    assert [r["asr_choice"] for r in records] == ["B", "A", "B"]
    assert [r["asr_parse_status"] for r in records] == ["tagged", "ordinal", "ordinal"]
    assert [r["label"] for r in records] == ["B", "A", "A"]
    assert [r["asr_score"] for r in records] == [1.0, 1.0, 0.0]
    assert [r["word_error_rate"] for r in records] == [None, None, None]
    assert [round(r["character_error_rate"], 4) for r in records] == [
        0.0460,
        0.0235,
        0.0448,
    ]
    assert summary["accuracy_asr"]["avg"] == pytest.approx(0.6667)
    assert (
        summary["n"] == 3
        and summary["n_unparsed_asr"] == 0
        and summary["n_null_asr"] == 0
    )
    assert "wer" not in summary and summary["n_wer"] == 0
    assert summary["normalizer_config_ids"]["zh"].startswith(
        "zh|normalizer=basic|remove_spaces=1|zhconv=zh-cn"
    )


def test_ifeval_en_rows():
    records, summary, _ = _run("ifeval")
    assert [r["subcategory"] for r in records] == ["combination:repeat_prompt"] * 3
    assert [r["asr_score"] for r in records] == [0.0, 1.0, 0.0]
    assert all(r["asr_correct_language"] for r in records)
    assert [r["asr_detected_language"] for r in records] == ["en"] * 3
    # the English normalizer maps "seven years" -> "7 years", so the front end is error-free on
    # rows 0 and 1 while the checker still fails row 0: that pair is the benchmark's signal
    assert [r["word_error_rate"] for r in records[:2]] == [0.0, 0.0]
    assert [r["character_error_rate"] for r in records[:2]] == [0.0, 0.0]
    assert summary["overall_lang_normalized_acc_asr"] == {
        "avg": 33.3333,
        "min": 0.0,
        "max": 100.0,
    }
    assert summary["subcategories_asr"] == {"combination:repeat_prompt": 0.3333}
    assert summary["n_wer"] == 3 and "wer" in summary


def test_gold_is_inline():
    records, _, _ = _run("ifeval")
    for r in records:
        assert "kwargs" in r and "prompt_to_repeat" in r["kwargs"]
        assert r["category"] == "combination" and r["task"] == "ifeval"
