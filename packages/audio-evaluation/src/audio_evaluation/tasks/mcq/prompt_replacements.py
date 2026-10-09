# Copied from ss-voice-toolkit@cac26e4 (evaluation/scorers/e2e/external_libraries/...). Keep in sync by re-copying, not editing.
"""Template-scaffolding prompt replacements for seahelm_audio/mcq's PROMPT_TEMPLATES, driven by
sibling replacements.jsonl -- more preprocessing rules (for zh or future
languages) can be added by editing that file alone, without touching data_prep.py or
answer_extraction.py's logic.

Replacements are applied to PROMPT_TEMPLATES[iso]["preamble"]/["task_template"] RAW strings.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

_REPLACEMENTS_PATH = Path(__file__).with_name("replacements.jsonl")


def _load_rows() -> list[dict]:
    """Load replacements.jsonl verbatim, one dict per non-blank line.

    Returns:
        rows (list[dict]): Each has "language", "replacements", and "letter_replacements"
            (the latter two are token -> translated text dicts). Empty list if the file
            doesn't exist.
    """
    if not _REPLACEMENTS_PATH.is_file():
        return []
    with open(_REPLACEMENTS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@lru_cache(maxsize=1)
def _rows_by_language() -> dict[str, dict]:
    """Cache replacements.jsonl rows keyed by language (module-level cache; static config)."""
    return {row["language"]: row for row in _load_rows()}


@lru_cache(maxsize=None)
def _pattern_for(iso: str) -> "re.Pattern | None":
    """Compile one whole-word, case-sensitive alternation regex for this iso's tokens, or None."""
    row = _rows_by_language().get(iso)
    if not row:
        return None
    tokens = {**row.get("replacements", {}), **row.get("letter_replacements", {})}
    if not tokens:
        return None
    ordered = sorted(
        tokens, key=len, reverse=True
    )  # longest-first, defensive against prefixes
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in ordered) + r")\b")


def apply_replacements(text: str, iso: str) -> str:
    """Replace configured tokens in a RAW (pre-.format()) template string, used in data_prep.py.

    Args:
        text (str): PROMPT_TEMPLATES[iso]["preamble"] or ["task_template"], BEFORE .format()
            interpolates real question/option content. Callers must never call this
            post-interpolation: doing so risks rewriting a coincidental "A" or "ANSWER" inside
            real dataset text (e.g. an option literally containing "Vitamin A"), which
            wouldn't exist in the string yet at this point.
        iso (str): SEA-HELM ISO language code.

    Returns:
        text (str): text with every case-sensitive, whole-word match of a configured token
            replaced by its translation; unchanged if iso has no replacements.jsonl entry.
    """
    pattern = _pattern_for(iso)
    if pattern is None:
        return text
    row = _rows_by_language()[iso]
    by_token = {**row.get("replacements", {}), **row.get("letter_replacements", {})}
    return pattern.sub(lambda m: by_token[m.group(1)], text)


@lru_cache(maxsize=1)
def reverse_choice_map() -> dict[str, str]:
    """Merge every language's letter_replacements into one reverse lookup.

    Only "letter_replacements" entries participate; cosmetic-only "replacements" entries
    (e.g. zh's ANSWER -> 答案) are excluded since they don't stand in for a scoreable choice
    letter.

    Returns:
        mapping (dict[str, str]): translated surface form -> its canonical "A"/"B"/"C"/"D"
            letter (e.g. {"第一": "A", "第二": "B", "第三": "C", "第四": "D"}), merged across
            every language in replacements.jsonl. Consumed by answer_extraction.py as a
            language-agnostic fallback.
    """
    merged: dict[str, str] = {}
    for row in _rows_by_language().values():
        for letter, translated in row.get("letter_replacements", {}).items():
            merged[translated] = letter
    return merged
