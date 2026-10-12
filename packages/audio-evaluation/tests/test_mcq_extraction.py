"""Choice extraction on the real zh replies plus the decoys the toolkit got wrong before."""

from audio_evaluation.tasks.mcq.answer_extraction import (
    extract_answer_choice,
    extract_answer_choice_with_stage,
)
from audio_evaluation.tasks.mcq.scoring import score_response, score_response_detailed


def test_zh_tagged_reply():
    assert extract_answer_choice_with_stage("答案：第二") == ("B", "tagged")


def test_zh_bare_ordinal_replies():
    assert extract_answer_choice_with_stage("第一") == ("A", "ordinal")
    assert extract_answer_choice_with_stage("第二") == ("B", "ordinal")


def test_zh_digit_spelling_from_asr_normalisation():
    assert extract_answer_choice("答案：第3") == "C"


def test_math_span_is_not_read_as_option_d():
    choice, stage = extract_answer_choice_with_stage("概率是 $P(D)$，答案：第三")
    assert choice == "C"
    assert stage != "template"


def test_recited_option_list_is_masked():
    # the enumeration 第一、第二、第三或第四 is a decoy; the real answer follows the tag
    assert extract_answer_choice("选项有第一、第二、第三或第四。答案：第四") == "D"


def test_english_tagged_reply():
    assert extract_answer_choice("Answer: C") == "C"


def test_unparsed_reply_is_none_not_a_letter():
    assert extract_answer_choice_with_stage("I am not sure.") == (None, "none")
    detail = score_response_detailed("I am not sure.", "A")
    assert detail == {"choice": None, "parse_status": "none", "score": 0.0}


def test_score_matches_label_case_insensitively():
    assert score_response("答案：第二", "b") == 1.0
    assert score_response("第二", "A") == 0.0
