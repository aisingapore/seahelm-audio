# Copied from ss-voice-toolkit@cac26e4 (evaluation/scorers/e2e/external_libraries/...). Keep in sync by re-copying, not editing.
"""SEA-HELM if-eval scoring: category dispatch + per-response evaluation."""

from . import instruction_checkers as ic

# subcategory (category:subcategory key, as it appears in the dataset's metadata) -> checker class.
# Verbatim from if_eval.py's CATEGORY_MAP.
CATEGORY_MAP = {
    "length_constraints:number_words": ic.NumberOfWordsChecker,
    "length_constraints:number_sentences": ic.NumberOfSentencesChecker,
    "length_constraints:number_paragraphs": ic.NumberOfParagraphsChecker,
    "detectable_content:postscript": ic.PostscriptChecker,
    "detectable_content:number_placeholders": ic.PlaceholderChecker,
    "detectable_format:title": ic.TitleChecker,
    "detectable_format:number_bullet_lists": ic.BulletListChecker,
    "detectable_format:multiple_sections": ic.SectionChecker,
    "detectable_format:number_highlighted_sections": ic.HighlightedSectionChecker,
    "detectable_format:json_format": ic.JSONChecker,
    "detectable_format:constrained_response": ic.ConstrainedOptionsChecker,
    "startend:first_word": ic.StartChecker,
    "startend:end_checker": ic.EndChecker,
    "startend:quotation": ic.QuotationMarkChecker,
    "keywords:existence": ic.WordExistenceChecker,
    "keywords:forbidden_words": ic.WordAbsenceChecker,
    "keywords:frequency": ic.WordFrequencyChecker,
    "keywords:number_frequency": ic.NumberFrequencyChecker,
    "combination:repeat_prompt": ic.RepeatPromptChecker,
    "combination:two_responses": ic.TwoResponsesChecker,
    "language:response_language": ic.ResponseLanguageChecker,
}

# Languages that don't use whitespace to separate words -- length_constraints:number_words
# is meaningless for them (and NumberOfWordsChecker asserts against being run on them).
NO_WORD_BOUNDARY_LANGS = {"th", "lo", "km", "my"}


def check_language(text: str, language: str) -> tuple[bool, str | None]:
    """Detect whether ``text`` is written in ``language`` (id/ms cross-accepted).

    Ported from if_eval.py's IFEvalMetric.check_language.

    Args:
        text (str): Response text to detect the language of (newlines
            replaced with spaces -- fast_langdetect chokes on them).
        language (str): Expected ISO language code (e.g. "ta").

    Returns:
        result (tuple[bool, str | None]): ``(True, detected_lang)`` if the
            detected language matches ``language`` (Malay/Indonesian accepted
            as each other), else ``(False, None)``.
    """
    import fast_langdetect

    detected = fast_langdetect.detect(text)["lang"]
    if language in ("id", "ms"):
        if detected in ("id", "ms"):
            return True, detected
    elif detected == language:
        return True, detected
    return False, None


def score_response(
    subcategory: str, category: str, kwargs: dict | None, response: str, language: str
) -> dict | None:
    """Score one response against one SEA-HELM if-eval instruction.

    Ported from IFEvalMetric.evaluate_response, scoring a single response
    instead of a whole dataframe (this module scores the text-arm and
    cascade-arm responses of one sample independently, not a full run).

    Args:
        subcategory (str): The instruction's "category:subcategory" key
            (e.g. "keywords:existence"), a ``CATEGORY_MAP`` key -- the
            dataset's native ``metadata.subcategory`` field.
        category (str): The instruction's top-level category (e.g.
            "keywords") -- the dataset's native ``metadata.category`` field.
            Used only to detect the "language" category, whose own
            ``response_language`` kwarg (not the task's ambient ``language``)
            is what the outer language check should target.
        kwargs (dict | None): The instruction's checker-constructor kwargs
            (the dataset's native ``kwargs`` field), with None values
            dropped before being passed to the checker.
        response (str): The candidate response to check.
        language (str): The task's ISO language code (e.g. "ta").

    Returns:
        result (dict | None): ``{"result": bool, "correct_language": bool,
            "detected_language": str | None, "lang_normalized_result": bool}``,
            or None when this instruction is skipped for ``language``
            (mirrors SEA-HELM's own per-row skip + dataframe-level filter in
            ``summarize_results`` for length_constraints:number_words in
            ``NO_WORD_BOUNDARY_LANGS``).
    """
    if language in NO_WORD_BOUNDARY_LANGS and "number_words" in subcategory:
        return None
    checker_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    checker = CATEGORY_MAP[subcategory](
        category=subcategory, language=language, response=response, **checker_kwargs
    )
    result = checker.evaluate_response()
    expected_lang = (
        language if category != "language" else checker_kwargs["response_language"]
    )
    correct_language, detected_language = check_language(
        response.replace("\n", " "), expected_lang
    )
    return {
        "result": result,
        "correct_language": correct_language,
        "detected_language": detected_language,
        "lang_normalized_result": result if correct_language else False,
    }
