"""WER/CER recipe parity with the toolkit's TranscriptionScorer."""

from audio_evaluation.common.transcription import TranscriptionScorer, canonical_iso


def test_english_normalizer_maps_number_words_so_wer_is_zero():
    s = TranscriptionScorer("en")
    r = s.evaluate_sample(
        "I have 7 years of experience", "I have seven years of experience"
    )
    assert r["word_error_rate"] == 0.0
    assert r["character_error_rate"] == 0.0
    assert s.normalizer_name == "english"


def test_chinese_is_cer_only_with_spaces_removed_and_simplified():
    s = TranscriptionScorer("zh")
    assert s.remove_spaces and s.zh_trad_to_simp
    r = s.evaluate_sample("答案 是 第二", "答案是第二")
    assert "word_error_rate" not in r
    assert r["character_error_rate"] == 0.0
    assert s.evaluate_sample("國家", "国家")["character_error_rate"] == 0.0
    assert s.config_id.startswith("zh|normalizer=basic|remove_spaces=1|zhconv=zh-cn|")


def test_empty_string_rules_never_reach_jiwer():
    s = TranscriptionScorer("en")
    assert s.evaluate_sample("", "")["character_error_rate"] == 0.0
    assert s.evaluate_sample("", "")["word_error_rate"] == 0.0
    assert s.evaluate_sample("hello", "")["character_error_rate"] == 1.0
    assert s.evaluate_sample("", "hello")["word_error_rate"] == 1.0


def test_zh_hans_alias_folds_to_zh():
    assert canonical_iso("zh-Hans") == "zh"
    assert TranscriptionScorer("zh-Hans").language == "zh"


def test_unlisted_language_gets_basic_normalizer_with_wer():
    s = TranscriptionScorer("ms")
    assert s.normalizer_name == "basic" and not s.remove_spaces
    assert "word_error_rate" in s.evaluate_sample("saya makan", "saya makan")


def test_tamil_uses_indic_normalizer():
    s = TranscriptionScorer("ta")
    assert s.normalizer_name == "indic:TamilNormalizer"
    assert s.evaluate_sample("வணக்கம்", "வணக்கம்")["character_error_rate"] == 0.0
