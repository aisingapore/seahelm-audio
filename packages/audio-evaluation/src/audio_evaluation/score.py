"""Join result rows with gold and score both arms (ported from ``SeaHelmStage.score``).

Each result row (from ``model-pipeline``) carries ``filename, language, spoken_text,
transcript, reply``: ``reply`` is the LLM's answer to the ASR transcript, so its score
(``asr_score``) is the cascade's score. The scorer never calls a model.
"""

from pathlib import Path

from .common.transcription import TranscriptionScorer
from .tasks import registry
from .tasks.harm import scoring as harm_scoring
from .tasks.ifeval import scoring as ifeval_scoring
from .tasks.mcq import scoring as mcq_scoring
from .tasks.qa import scoring as qa_scoring

REQUIRED_KEYS = ("filename", "language", "spoken_text", "transcript", "reply")


def validate_rows(rows: list[dict]) -> list[str]:
    """Check result rows for missing keys and duplicate filenames.

    Args:
        rows (list[dict]): Result rows.

    Returns:
        problems (list[str]): Human-readable problems; empty when the rows are usable.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            problems.append(f"row {i}: missing keys {missing}")
            continue
        fn = row["filename"]
        if fn in seen:
            problems.append(f"row {i}: duplicate filename {fn!r}")
        seen.add(fn)
    return problems


def join_report(
    rows: list[dict], gold: dict[str, dict], spec: registry.TaskSpec
) -> dict:
    """Describe how the result rows join to gold, before any scoring happens.

    Args:
        rows (list[dict]): Validated result rows.
        gold (dict[str, dict]): Gold keyed by filename.
        spec (registry.TaskSpec): The task.

    Returns:
        report (dict): ``task``, ``n_results``, ``n_gold_found``, ``missing`` (filenames
            with no gold row), ``language_mismatch`` (filename, row language, gold
            language) and ``prefix_mismatch`` (filenames not matching the task prefix).
    """
    missing = [r["filename"] for r in rows if r["filename"] not in gold]
    mismatch = [
        (r["filename"], r["language"], gold[r["filename"]].get("language"))
        for r in rows
        if r["filename"] in gold
        and gold[r["filename"]].get("language") not in (None, r["language"])
    ]
    prefix = [
        r["filename"]
        for r in rows
        if not r["filename"].startswith(spec.filename_prefix)
    ]
    return {
        "task": spec.name,
        "n_results": len(rows),
        "n_gold_found": len(rows) - len(missing),
        "missing": missing,
        "language_mismatch": mismatch,
        "prefix_mismatch": prefix,
    }


def report_problems(report: dict) -> list[str]:
    """Turn a join report into blocking problems (empty when the join is clean)."""
    problems = []
    if report["missing"]:
        problems.append(
            f"{len(report['missing'])} result row(s) have no gold: {report['missing'][:5]}"
        )
    if report["language_mismatch"]:
        problems.append(
            f"language mismatch on {len(report['language_mismatch'])} row(s): {report['language_mismatch'][:5]}"
        )
    if report["prefix_mismatch"]:
        problems.append(
            f"{len(report['prefix_mismatch'])} filename(s) do not match task prefix: {report['prefix_mismatch'][:5]}"
        )
    return problems


def _score_arm(
    spec: registry.TaskSpec, response: str, gold: dict, language: str
) -> dict:
    """Score one arm's response for one row; returns ``score`` plus task-specific extras."""
    if spec.kind == "ifeval":
        result = ifeval_scoring.score_response(
            gold["subcategory"],
            gold["category"],
            gold.get("kwargs"),
            response,
            language,
        )
        if (
            result is None
        ):  # instruction skipped for this language (e.g. number_words on th)
            return {
                "score": None,
                "result": None,
                "correct_language": None,
                "detected_language": None,
            }
        return {
            "score": float(result["lang_normalized_result"]),
            "result": bool(result["result"]),
            "correct_language": bool(result["correct_language"]),
            "detected_language": result["detected_language"],
        }
    if spec.kind == "mcq":
        detail = mcq_scoring.score_response_detailed(response, gold["label"])
        return {
            "score": detail["score"],
            "choice": detail["choice"],
            "parse_status": detail["parse_status"],
        }
    if spec.kind == "qa":
        return {"score": qa_scoring.score_response(response, gold["label"], language)}
    if spec.kind == "harm":
        return {"score": harm_scoring.score_response(response)}
    raise ValueError(f"unknown task kind {spec.kind!r}")


def score_row(
    row: dict,
    gold: dict,
    spec: registry.TaskSpec,
    scorers: dict[str, TranscriptionScorer],
) -> dict:
    """Score one result row: WER/CER of the front end, then the reply against gold.

    Args:
        row (dict): Result row.
        gold (dict): Flat gold for this filename (from ``gold.load_gold``).
        spec (registry.TaskSpec): The task.
        scorers (dict[str, TranscriptionScorer]): Cache of per-language scorers; filled lazily.

    Returns:
        record (dict): The row, the gold inline, ``task``/``kind``, ``word_error_rate``
            (None when the language scores CER only), ``character_error_rate``,
            ``asr_score`` and the task's ``asr_*`` extras (ifeval: ``asr_result``,
            ``asr_correct_language``, ``asr_detected_language``; mcq: ``asr_choice``,
            ``asr_parse_status``).
    """
    language = row["language"]
    scorer = scorers.get(language)
    if scorer is None:
        scorer = scorers[language] = TranscriptionScorer(language)
    wer_cer = scorer.evaluate_sample(row["spoken_text"], row["transcript"])

    record = dict(row)
    record["task"] = spec.name
    record["kind"] = spec.kind
    for k, v in gold.items():
        if k != "language":
            record[k] = v
    record["word_error_rate"] = wer_cer.get("word_error_rate")
    record["character_error_rate"] = wer_cer["character_error_rate"]

    result = _score_arm(spec, row["reply"], gold, language)
    record["asr_score"] = result.pop("score")
    for k, v in result.items():
        record[f"asr_{k}"] = v
    return record


def score_rows(
    rows: list[dict], gold: dict[str, dict], spec: registry.TaskSpec
) -> tuple[list[dict], dict[str, str]]:
    """Score every row (rows must already have passed the join).

    Returns:
        result (tuple[list[dict], dict[str, str]]): the scored records and the
            ``language -> normalizer config id`` map actually used.
    """
    scorers: dict[str, TranscriptionScorer] = {}
    records = [score_row(row, gold[row["filename"]], spec, scorers) for row in rows]
    return records, {lang: s.config_id for lang, s in sorted(scorers.items())}


def default_out_dir(spec: registry.TaskSpec) -> Path:
    """Where ``cli score`` writes when ``--out`` is not given."""
    return Path("outputs") / "evaluation" / spec.name
