"""Aggregate scored records into one summary per task (ported from ``SeaHelmStage.summarize``).

Adds what the toolkit lacked: explicit counts (``n``, nulls, unparsed, rows with a WER)
and the normalizer config ids plus library versions recorded once.
"""

from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

from .common.stats import aggregate_stats
from .tasks.registry import TaskSpec

_VERSIONED = (
    "audio-evaluation",
    "jiwer",
    "whisper-normalizer",
    "zhconv",
    "pythainlp",
    "fast-langdetect",
    "jieba",
    "pyarrow",
)


def library_versions() -> dict[str, str | None]:
    """Installed versions of every library that can move a number."""
    out: dict[str, str | None] = {}
    for name in _VERSIONED:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def summarize(
    records: list[dict],
    spec: TaskSpec,
    normalizer_config_ids: dict[str, str],
    extra: dict | None = None,
) -> dict:
    """Build the summary for one task's scored records.

    Args:
        records (list[dict]): Output of ``score.score_rows``.
        spec (TaskSpec): The task.
        normalizer_config_ids (dict[str, str]): ``language -> config id`` used for WER/CER.
        extra (dict | None): Provenance to carry verbatim (results path, pipeline metadata, ...).

    Returns:
        summary (dict): Counts, WER/CER stats, ``<metric>_asr``, per-subcategory
            breakdowns for ifeval, and provenance.
    """
    n = len(records)
    cers = [
        r["character_error_rate"]
        for r in records
        if r.get("character_error_rate") is not None
    ]
    wers = [
        r["word_error_rate"] for r in records if r.get("word_error_rate") is not None
    ]
    summary: dict = {
        "task": spec.name,
        "kind": spec.kind,
        "quality_metric": spec.metric,
        "quality_scale": spec.scale,
        "n": n,
        "languages": sorted({r["language"] for r in records}),
        "n_wer": len(wers),
        "cer": aggregate_stats(cers),
    }
    if wers:
        summary["wer"] = aggregate_stats(wers)

    scores = [r["asr_score"] for r in records if r.get("asr_score") is not None]
    summary["n_null_asr"] = n - len(scores)
    if spec.kind == "mcq":
        summary["n_unparsed_asr"] = sum(
            1 for r in records if r.get("asr_parse_status") == "none"
        )
    summary[f"{spec.metric}_asr"] = aggregate_stats([spec.scale * s for s in scores])
    if spec.kind == "ifeval":
        by_subcat: dict[str, list[float]] = {}
        for r in records:
            s = r.get("asr_score")
            if s is not None:
                by_subcat.setdefault(r.get("subcategory"), []).append(s)
        summary["subcategories_asr"] = {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_subcat.items())
        }

    summary["normalizer_config_ids"] = normalizer_config_ids
    summary["versions"] = library_versions()
    summary["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if extra:
        summary.update(extra)
    return summary
