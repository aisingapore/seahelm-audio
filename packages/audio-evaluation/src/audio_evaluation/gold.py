"""Gold lookup from the published parquet dataset, keyed by sample filename."""

from pathlib import Path

import pyarrow.parquet as pq

from .tasks.registry import TaskSpec


def parquet_files(data_dir: Path) -> list[Path]:
    """All parquet shards in a task's dataset directory, sorted.

    Args:
        data_dir (Path): ``<dataset root>/<task data_dir>``.

    Returns:
        files (list[Path]): Shard paths.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no shards.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir.resolve()}")
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet shards under {data_dir.resolve()}")
    return files


def _flatten(spec: TaskSpec, row: dict) -> dict:
    """Turn one raw parquet row into the flat gold dict the scorer joins in."""
    gold = {"language": row.get("language")}
    if spec.kind == "ifeval":
        meta = row.get("metadata") or {}
        gold["category"] = meta.get("category")
        gold["subcategory"] = meta.get("subcategory")
        kwargs_list = row.get("kwargs") or []
        first = kwargs_list[0] if kwargs_list else {}
        gold["kwargs"] = {k: v for k, v in (first or {}).items() if v is not None}
    elif spec.kind == "mcq":
        gold["label"] = row.get("label")
        gold["subject"] = row.get("subject")
    elif spec.kind == "qa":
        label = row.get("label")
        gold["label"] = label if isinstance(label, list) else [label]
    # harm: nothing beyond language
    return gold


def load_gold(
    spec: TaskSpec, data_root: Path, filenames: set[str] | None = None
) -> dict[str, dict]:
    """Load gold for one task, keyed by ``filename``.

    Args:
        spec (TaskSpec): The task (selects the data dir and gold columns).
        data_root (Path): Dataset root (``data/seahelm_audio_data``).
        filenames (set[str] | None): If given, only these filenames are kept.

    Returns:
        gold (dict[str, dict]): ``filename -> flat gold dict`` (see ``_flatten``).
    """
    columns = ["filename", "language", *spec.gold_columns]
    wanted = set(filenames) if filenames is not None else None
    gold: dict[str, dict] = {}
    for shard in parquet_files(data_root / spec.data_dir):
        table = pq.read_table(shard, columns=columns)
        for row in table.to_pylist():
            fn = row["filename"]
            if wanted is not None and fn not in wanted:
                continue
            gold[fn] = _flatten(spec, row)
    return gold
