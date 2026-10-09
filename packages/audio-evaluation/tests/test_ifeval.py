"""ifeval checker behaviour the scorer relies on, offline."""

from importlib import metadata as importlib_metadata

from audio_evaluation.tasks.ifeval.scoring import check_language, score_response

SUBCAT = "combination:repeat_prompt"
CAT = "combination"


def _score(response: str, prompt: str, language: str = "en"):
    return score_response(SUBCAT, CAT, {"prompt_to_repeat": prompt}, response, language)


def test_repeat_prompt_passes_when_prompt_is_repeated_then_answered():
    r = _score(
        "Tell me a joke. Why did the chicken cross the road? To get to the other side.",
        "Tell me a joke.",
    )
    assert r["result"] is True
    assert r["correct_language"] is True
    assert r["detected_language"] == "en"
    assert r["lang_normalized_result"] is True


def test_repeat_prompt_fails_without_an_answer():
    r = _score("Tell me a joke.", "Tell me a joke.")
    assert r["result"] is False


def test_repeat_prompt_is_a_raw_text_match_so_asr_number_words_fail():
    # "seven years" vs the gold "7 years": this mismatch is the front-end cost the benchmark
    # measures, so it must stay a failure (do not normalise numbers in the checker).
    r = _score(
        "I have seven years of experience. Here is my answer.",
        "I have 7 years of experience.",
    )
    assert r["result"] is False


def test_number_words_is_skipped_for_languages_without_word_boundaries():
    r = score_response(
        "length_constraints:number_words",
        "length_constraints",
        {"num_words": 5, "relation": "at least"},
        "สวัสดี",
        "th",
    )
    assert r is None


def test_language_detection_is_offline_and_pinned():
    assert importlib_metadata.version("fast-langdetect") == "0.3.2"
    assert check_language("hello world, this is a plain english sentence", "en") == (
        True,
        "en",
    )


def test_malay_and_indonesian_are_cross_accepted():
    ok, detected = check_language("Saya suka makan nasi goreng setiap pagi", "ms")
    assert ok is True
    assert detected in ("id", "ms")
