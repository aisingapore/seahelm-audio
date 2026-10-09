# Copied from ss-voice-toolkit@cac26e4 (evaluation/scorers/e2e/external_libraries/...). Keep in sync by re-copying, not editing.
"""Recover the A-D choice letter from a free-text MCQ response."""

import re
from functools import lru_cache

from ...common.mcq_extract import extract_choice
from .prompt_replacements import reverse_choice_map

_ANSWER_TAG_WORDS = [
    "answer",
    "jawaban",
    "အဖြေ",
    "答案",
    "câu trả lời",
    "jawapan",
    "sagot",
]

# CJK ordinal -> its Arabic-digit spelling, for _choice_words below.
_CJK_ORDINAL_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4"}

# LaTeX/math spans -- notation rather than prose. See _strip_math_spans.
_MATH_SPAN_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]*\$", re.S
)  # consume display math first


def _strip_math_spans(text: str) -> str:
    """Replace ``$...$`` / ``$$...$$`` spans with a space, for ``extract_choice`` only.

    Masking is applied to Stage 1 extraction alone: ``_extract_tagged_choice`` is anchored on an answer
    tag and ``_extract_ordinal_choice`` is structure-aware, so neither reads notation as an
    answer, and the latter ranks matches by position against the unmodified string.

    Args:
        text (str): The candidate's free-text response.

    Returns:
        text (str): ``text`` with math spans replaced by a single space -- offsets are not
            preserved because the caller only does substring tests, but a space rather than an
            empty string keeps neighbouring tokens from gluing into a spurious match. A lone
            ``$`` (a price) starts no span and is left alone.
    """
    return _MATH_SPAN_RE.sub(" ", text or "")


@lru_cache(maxsize=1)
def _choice_words() -> dict[str, str]:
    """``reverse_choice_map`` plus a digit spelling of each CJK ordinal, longest surface first.

    zh scaffolding replaces A/B/C/D with 第一/第二/第三/第四, but an ASR model that normalizes
    spoken ordinals to digits hands the LLM 第1/第2/第3/第4 instead -- and the LLM mirrors that
    form in its answer, leaving a choice ``reverse_choice_map`` alone cannot resolve. Adding
    the digit spelling as an alias keeps both forms scoreable. The transform is a no-op for
    languages whose surface forms contain no CJK numeral, so it stays language-agnostic.

    Longest-first ordering matters for the regex alternation built from this table in
    ``_ordinal_re``: alternation is first-match-wins, so it prevents a short surface form
    from shadowing a longer one that starts with it.

    Returns:
        mapping (dict[str, str]): surface form -> its "A"/"B"/"C"/"D" letter, insertion-ordered
            longest surface form first -- so iterating it yields the surfaces in the order
            ``_ordinal_re`` needs, while indexing it resolves a match to its letter. Cached --
            the underlying map is static config.
    """
    merged = dict(reverse_choice_map())
    for word, letter in list(merged.items()):
        merged.setdefault(
            "".join(_CJK_ORDINAL_DIGITS.get(ch, ch) for ch in word), letter
        )
    return dict(sorted(merged.items(), key=lambda kv: -len(kv[0])))


@lru_cache(maxsize=1)
def _ordinal_re() -> "re.Pattern":
    """Alternation matching any one translated-ordinal surface form.

    Built from ``_choice_words``, so any language that gains a ``letter_replacements`` entry in replacements.jsonl becomes searchable for free and this module keeps the language-agnostic property ``_choice_words`` already has.

    Returns:
        pattern (re.Pattern): Longest-first alternation over every known surface form.
            Cached -- the underlying table is static config.
    """
    return re.compile("|".join(re.escape(word) for word in _choice_words()))


@lru_cache(maxsize=1)
def _enumeration_re() -> "re.Pattern":
    """Alternation matching a *recited option list*, e.g. 第一、第二、第三或第四.

    Responses routinely echo the prompt's own list of choices, or re-read the question with
    its options inline. Every ordinal in such a run is a decoy, so a search that counts them
    picks an option-list entry over the real answer -- this is the trap that makes a naive
    first-match search win by luck. Masking these runs out first is what makes the search safe.

    Returns:
        pattern (re.Pattern): Two or more ordinals joined by list separators.
            Current version: The separators are CJK-specific, so this is a no-op for a language whose surface forms never appear in such a run.
            Cached -- derives from static config.
    """
    ordinal = _ordinal_re().pattern
    link = r"\s*(?:[、,，]|或|or)\s*"
    return re.compile(f"(?:{ordinal})(?:{link}(?:{ordinal}))+")


def _ordinal_spots(text: str) -> list["re.Match"]:
    """Every translated ordinal in ``text``, with recited option lists masked out.

    The single way this module looks for ordinals, so no caller can search unmasked text and
    reintroduce the option-list trap ``_enumeration_re`` exists to close.

    Args:
        text (str): Any span of a response -- the whole thing, or the part after an answer tag.

    Returns:
        spots (list[re.Match]): Matches in order of appearance, each resolvable to a letter via
            ``_choice_words``. Enumerations are blanked to spaces rather than deleted, so
            ``match.start()`` still lines up with the original string for callers that rank
            matches by position.
    """
    masked = _enumeration_re().sub(lambda m: " " * len(m.group()), text or "")
    return list(_ordinal_re().finditer(masked))


def _extract_tagged_choice(response: str) -> str | None:
    """Fallback extraction for SEA-HELM's own "{answer_tag} ANSWER" format.

    Args:
        response (str): The candidate's free-text response.

    Returns:
        choice (str | None): The A-D letter found just past a known answer
            tag word (skipping any punctuation/whitespace separator),
            uppercased, or None if no tag is found or no such letter follows.
            A translated ordinal in either spelling (第一 / 第1) counts as that
            letter -- see ``_choice_words``.
    """
    lowered = (response or "").lower()
    for tag in _ANSWER_TAG_WORDS:
        idx = lowered.find(tag)
        if idx == -1:
            continue
        # skip punctuation/whitespace after the tag, then check the first character
        tail = lowered[idx + len(tag) :].lstrip(" :：,，-\t\n")
        if tail[:1].upper() in ("A", "B", "C", "D"):
            return tail[:1].upper()

        # Fallback for languages whose scaffolding replaces bare letters with translated ordinal
        # words (e.g. zh: A/B/C/D -> 第一/第二/第三/第四, per replacements.jsonl), in either the
        # translated or the ASR-normalized digit spelling (第一 / 第1).
        spots = _ordinal_spots(tail)
        if spots:
            return _choice_words()[spots[0].group()]  # first ordinal after the tag
    return None


def _extract_ordinal_choice(response: str) -> str | None:
    """Fallback that searches the whole response for a translated ordinal word.

    Which ordinal wins is decided by structure rather than raw position, since a response
    may also contain a recited option list or an incidental number (a chapter reference, a
    measurement, a legal citation).

    The search is anchored on a known answer tag if one exists, else the first ordinal found is returned.
    The search is safe because enumerations are masked out first.

    Args:
        response (str): The candidate's free-text response.

    Returns:
        choice (str | None): The A-D letter for the ordinal nearest the answer tag, in either
            spelling (第一 / 第1); or for the first ordinal when the response carries no tag at
            all, an untagged answer tending to lead. None if no ordinal is found.
    """
    text = response or ""
    spots = _ordinal_spots(text)
    if not spots:
        return None
    lowered = text.lower()
    anchors = [
        idx for idx in (lowered.find(tag) for tag in _ANSWER_TAG_WORDS) if idx != -1
    ]
    # If answer is untagged and tends to lead -> pick the first ordinal found
    if not anchors:
        return _choice_words()[spots[0].group()]
    anchor = min(anchors)
    # If answer is tagged -> pick the ordinal nearest the tag
    nearest = min(spots, key=lambda m: (abs(m.start() - anchor), m.start() < anchor))
    return _choice_words()[nearest.group()]


STAGE_TEMPLATE = "template"
STAGE_TAGGED = "tagged"
STAGE_ORDINAL = "ordinal"
STAGE_NONE = "none"


def extract_answer_choice_with_stage(response: str) -> tuple[str | None, str]:
    """Recover the choice letter and report which extraction stage found it.

    Three stages, each a fallback for the one before it:
     - ``template``: VoiceBench's template matching over Latin a/b/c/d, with math spans masked
     - ``tagged``: search anchored on a known answer tag
     - ``ordinal``: search of the whole response for a translated ordinal

    Stage 1 sees a math-stripped copy (see ``_strip_math_spans``) because it matches short
    substrings anywhere in the text and would otherwise read a LaTeX variable such as ``$P(D)$``
    as "option D" -- and being first in the chain, a false hit there shadows the later stages.

    Args:
        response (str): The candidate's free-text response.

    Returns:
        result (tuple[str | None, str]): ``(choice, stage)`` -- "A".."D" and the stage name,
            or ``(None, "none")`` when no choice could be read. The stage is what makes an
            unparsed response distinguishable from a wrong one in the scored output.
    """
    choice = extract_choice(_strip_math_spans(response))
    if choice:
        return choice, STAGE_TEMPLATE
    choice = _extract_tagged_choice(response)
    if choice:
        return choice, STAGE_TAGGED
    choice = _extract_ordinal_choice(response)
    if choice:
        return choice, STAGE_ORDINAL
    return None, STAGE_NONE


def extract_answer_choice(response: str) -> str | None:
    """Recover the choice letter from a free-text response (see ``extract_answer_choice_with_stage``).

    Args:
        response (str): The candidate's free-text response.

    Returns:
        choice (str | None): "A", "B", "C" or "D", or None if no choice could be read.
    """
    return extract_answer_choice_with_stage(response)[0]
