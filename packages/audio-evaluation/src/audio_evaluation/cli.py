"""Command line: ``audio-evaluation download`` and ``audio-evaluation score``.

    audio-evaluation download --dest data/seahelm_audio_data [--revision <sha>]
    audio-evaluation score --results outputs/pipeline/ifeval/results.jsonl            # -> outputs/evaluation/ifeval/
    audio-evaluation score --results outputs/pipeline/mmlu_lite/results.jsonl --dry-run  # join report only

Run from the repo root; the default data root and output dir are relative to it.

Each subcommand is a pair of functions kept together: ``register_<name>`` declares its
arguments on the subparser and binds ``cmd_<name>`` as the handler via ``set_defaults``;
``main`` dispatches on whichever handler the parsed subcommand selected.
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from . import score as score_mod
from .common.jsonl import read_jsonl, write_jsonl
from .gold import load_gold
from .summary import summarize
from .tasks.registry import TASKS, resolve_task

DEFAULT_DATA_ROOT = "data/seahelm_audio_data"


# --------------------------------------------------------------------------- #
# Download handler
# --------------------------------------------------------------------------- #
def register_download(subparsers) -> None:
    """Declare ``audio-evaluation download``."""
    p = subparsers.add_parser(
        "download", help="Snapshot the dataset from the Hugging Face Hub."
    )
    p.add_argument(
        "--dest", default=DEFAULT_DATA_ROOT, help="Local directory to populate."
    )
    p.add_argument(
        "--revision", default=None, help="Hub commit/branch to pin (recommended)."
    )
    p.set_defaults(func=cmd_download)


def cmd_download(args: argparse.Namespace) -> int:
    """Download the dataset snapshot; returns the process exit code."""
    from .downloader import download

    download(args.dest, revision=args.revision)
    return 0


# --------------------------------------------------------------------------- #
# Scoring handler
# --------------------------------------------------------------------------- #
def register_score(subparsers) -> None:
    """Declare ``audio-evaluation score``."""
    p = subparsers.add_parser(
        "score", help="Score a results.jsonl produced by a pipeline."
    )
    p.add_argument("--results", required=True, help="Path to results.jsonl.")
    p.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT, help="Dataset snapshot root."
    )
    p.add_argument(
        "--task", choices=sorted(TASKS), default=None, help="Override task inference."
    )
    p.add_argument(
        "--out", default=None, help="Output dir (default outputs/evaluation/<task>)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the join report and stop; exits non-zero if the join has problems.",
    )
    p.set_defaults(func=cmd_score)


def cmd_score(args: argparse.Namespace) -> int:
    """Validate, join with gold, score, and write scored.jsonl + summary.json.

    Returns the process exit code: 0 on success, 2 when the rows are unusable or
    the join to gold is incomplete (nothing is written in that case).
    """
    results_path = Path(args.results)
    rows = read_jsonl(results_path)
    if not rows:
        logger.error(f"no rows in {results_path}")
        return 2
    problems = score_mod.validate_rows(rows)
    if problems:
        for p in problems:
            logger.error(p)
        return 2

    spec = resolve_task(args.task, results_path, [r["filename"] for r in rows])
    data_root = Path(args.data_root)
    gold = load_gold(spec, data_root, {r["filename"] for r in rows})
    report = score_mod.join_report(rows, gold, spec)
    logger.info(
        f"[{spec.name}] join: {report['n_gold_found']}/{report['n_results']} rows matched gold "
        f"from {data_root / spec.data_dir}"
    )
    join_problems = score_mod.report_problems(report)
    for p in join_problems:
        logger.error(p)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not join_problems else 2
    if join_problems:
        return 2

    records, config_ids = score_mod.score_rows(rows, gold, spec)
    out_dir = Path(args.out) if args.out else score_mod.default_out_dir(spec)
    meta_path = results_path.with_name("metadata.json")
    extra = {
        "results_path": str(results_path),
        "data_dir": str(data_root / spec.data_dir),
        "pipeline_metadata": json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else None,
    }
    summary = summarize(records, spec, config_ids, extra)

    n = write_jsonl(out_dir / "scored.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metric = summary.get(f"{spec.metric}_asr", {})
    logger.info(
        f"[{spec.name}] scored {n} row(s) -> {out_dir}  "
        f"{spec.metric}_asr={metric.get('avg')}  cer={summary['cer']['avg']}"
        + (f"  wer={summary['wer']['avg']}" if "wer" in summary else "")
    )
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and register every subcommand."""
    parser = argparse.ArgumentParser(
        prog="audio-evaluation", description=__doc__.splitlines()[0]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_download(subparsers)
    register_score(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (or ``sys.argv``), run the selected subcommand, exit with its code."""
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
