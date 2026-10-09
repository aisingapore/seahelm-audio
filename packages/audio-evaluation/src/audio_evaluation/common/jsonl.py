"""Minimal JSONL read/write helpers."""

import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines.

    Args:
        path (str | Path): Path to the JSONL file.

    Returns:
        rows (list[dict]): One parsed dict per non-blank line, in file order.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write dicts as one JSON object per line (UTF-8, not ASCII-escaped).

    Args:
        path (str | Path): Destination file; parent directories are created.
        rows (Iterable[dict]): Records to write.

    Returns:
        n (int): Number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
