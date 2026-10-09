"""SEA-HELM question_answering (QA) scoring: token-overlap F1 against reference answers."""

import re
import string
from collections import Counter

# Per-language answer tags, verbatim from the task's prompt templates (toolkit
# seahelm_audio/qa/data_prep.py PROMPT_TEMPLATES[*]["answer_tag"]).
ANSWER_TAGS = {
    "id": "Jawaban:",
    "vi": "Câu trả lời:",
    "th": "คำตอบ:",
    "ta": "பதில்:",
    "zh": "答案：",
}


def extract_answer(response: str, language: str) -> str:
    """Keep only what follows the language's answer tag (seahelm-core's extract_response).

    Args:
        response (str): The candidate's free-text response.
        language (str): ISO code, selects the tag in ``ANSWER_TAGS``.

    Returns:
        answer (str): Text after the first tag occurrence (leading whitespace, backticks
            and asterisks skipped), or the whole response when the tag is absent.
    """
    tag = ANSWER_TAGS.get(language)
    text = response or ""
    if not tag:
        return text
    match = re.search(re.escape(tag) + r"[\s\r\n`*]*(.*)", text, flags=re.DOTALL)
    return match.group(1) if match else text


def _normalize_answer(s: str) -> str:
    """Lowercase and strip punctuation/extra whitespace, verbatim from QuestionAnsweringMetric."""
    text = s.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _f1(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between a prediction and one reference answer."""
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _tokenize_thai(text: str) -> str:
    """Space-join Thai word segments (PyThaiNLP ``newmm``)."""
    from pythainlp import word_tokenize

    return " ".join(word_tokenize(engine="newmm", text=text))


def _tokenize_chinese(text: str) -> str:
    """Space-join Chinese word segments (jieba)."""
    import jieba

    return " ".join(jieba.cut(text))


def score_response(response: str, labels: list[str], language: str) -> float:
    """Score one QA response against its reference answer(s).

    Args:
        response (str): The candidate's free-text response.
        labels (list[str]): One or more acceptable reference answers.
        language (str): ISO code; ``th``/``zh`` are word-segmented before scoring.

    Returns:
        f1 (float): The best (max-over-references) token-overlap F1, 0.0 if no reference
            is provided or none overlaps at all.
    """
    refs = [label or "" for label in (labels or [])] or [""]
    resp = extract_answer(response or "", language)
    if language == "th":
        resp = _tokenize_thai(resp)
        refs = [_tokenize_thai(ref) for ref in refs]
    elif language == "zh":
        resp = _tokenize_chinese(resp)
        refs = [_tokenize_chinese(ref) for ref in refs]
    return max(_f1(resp, ref) for ref in refs)
