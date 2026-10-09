"""The scorer refuses bad inputs before scoring anything."""

from audio_evaluation.score import join_report, report_problems, validate_rows
from audio_evaluation.tasks.registry import TASKS, resolve_task


def _row(fn: str, lang: str = "en") -> dict:
    return {
        "filename": fn,
        "language": lang,
        "spoken_text": "a",
        "transcript": "a",
        "reply": "a",
    }


def test_missing_keys_and_duplicates_are_reported():
    rows = [
        _row("seahelm_ifeval_en_000000.wav"),
        _row("seahelm_ifeval_en_000000.wav"),
        {"filename": "x"},
    ]
    problems = validate_rows(rows)
    assert any("duplicate" in p for p in problems)
    assert any("missing keys" in p for p in problems)


def test_join_report_flags_missing_gold_language_and_prefix():
    spec = TASKS["ifeval"]
    rows = [
        _row("seahelm_ifeval_en_000000.wav"),
        _row("seahelm_ifeval_en_000001.wav", "zh"),
        _row("other.wav"),
    ]
    gold = {"seahelm_ifeval_en_000001.wav": {"language": "en"}}
    report = join_report(rows, gold, spec)
    assert report["n_gold_found"] == 1
    assert report["missing"] == ["seahelm_ifeval_en_000000.wav", "other.wav"]
    assert report["language_mismatch"] == [("seahelm_ifeval_en_000001.wav", "zh", "en")]
    assert report["prefix_mismatch"] == ["other.wav"]
    assert len(report_problems(report)) == 3


def test_task_resolution_order():
    assert resolve_task("mmlu_lite").kind == "mcq"
    assert resolve_task(None, "outputs/pipeline/ifeval/results.jsonl").name == "ifeval"
    assert (
        resolve_task(
            None, "/nowhere/results.jsonl", ["seahelm_global_mmlu_lite_zh_000000.wav"]
        ).name
        == "mmlu_lite"
    )
