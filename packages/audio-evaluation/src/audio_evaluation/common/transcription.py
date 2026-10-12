"""WER/CER scoring with the toolkit's normalization recipe, keyed by ISO 639-1 code.

Ported from ss-voice-toolkit@cac26e4 ``evaluation/scorers/common/transcription_scorer.py``
(the ``task_type="asr"`` branch only) together with the per-language ``options`` blocks of
``evaluation/configs/languages_config.yaml``. The toolkit selects normalizers by dataset
language *name* ("english", "chinese"); result rows carry ISO codes, so the mapping from
code to recipe lives here, in ``LANGUAGE_OPTIONS`` and ``_INDIC_NORMALIZER_NAMES``.

Recipe, in order: text normalizer -> ``.strip()`` -> remove spaces (if set) -> traditional
to simplified Chinese (if set). Empty strings never reach jiwer: both empty scores 0.0, one
empty scores 1.0. When spaces are removed only CER is meaningful, so ``word_error_rate`` is
omitted rather than reported as 0.
"""

import re
import unicodedata

from jiwer import cer, wer

# ISO code -> whisper-normalizer class for Brahmic scripts. BasicTextNormalizer strips
# these scripts' combining vowel marks and shreds words into consonants, so the
# script-specific normalizers are used instead (same reasoning as the toolkit).
_INDIC_NORMALIZER_NAMES = {
    "ta": "TamilNormalizer",
    "hi": "HindiNormalizer",
    "bn": "BengaliNormalizer",
    "te": "TeluguNormalizer",
    "ml": "MalayalamNormalizer",
    "kn": "KannadaNormalizer",
    "gu": "GujaratiNormalizer",
    "pa": "PunjabiNormalizer",
    "or": "OdiaNormalizer",
}

# Per-language options, verbatim from the toolkit's languages_config.yaml ``options``
# blocks (chinese: remove_spaces + zhconv; thai: remove_spaces; lao/khmer/burmese:
# normalize off + remove_spaces). Any code not listed gets ``_DEFAULT_OPTIONS``.
LANGUAGE_OPTIONS: dict[str, dict] = {
    "zh": {"normalize": True, "remove_spaces": True, "zh_trad_to_simp": True},
    "th": {"normalize": True, "remove_spaces": True, "zh_trad_to_simp": False},
    "lo": {"normalize": False, "remove_spaces": True, "zh_trad_to_simp": False},
    "km": {"normalize": False, "remove_spaces": True, "zh_trad_to_simp": False},
    "my": {"normalize": False, "remove_spaces": True, "zh_trad_to_simp": False},
}
_DEFAULT_OPTIONS = {"normalize": True, "remove_spaces": False, "zh_trad_to_simp": False}

# The toolkit's languages_config.yaml tags Chinese as ``zh-Hans``; results carry ``zh``.
_ALIASES = {"zh-hans": "zh", "zh_hans": "zh", "zh-cn": "zh", "zh_cn": "zh"}


def canonical_iso(language: str) -> str:
    """Lowercase an ISO code and fold known aliases (e.g. ``zh-Hans`` -> ``zh``).

    Args:
        language (str): ISO 639-1 code as found in a result row.

    Returns:
        iso (str): The canonical code used to look up ``LANGUAGE_OPTIONS``.
    """
    code = (language or "").strip().lower()
    return _ALIASES.get(code, code)


def options_for(language: str) -> dict:
    """Resolve the normalization options for one language.

    Args:
        language (str): ISO code (aliases accepted).

    Returns:
        options (dict): ``normalize``, ``remove_spaces``, ``zh_trad_to_simp``.
    """
    return dict(LANGUAGE_OPTIONS.get(canonical_iso(language), _DEFAULT_OPTIONS))


def _strip_punctuation_keep_marks(text: str) -> str:
    """Strip punctuation/symbols but preserve combining marks, then lowercase + collapse whitespace.

    Args:
        text (str): Script-canonicalized text (punctuation/case not yet touched).

    Returns:
        normalized (str): Lowercased text with punctuation/symbols removed and
            whitespace collapsed; combining marks are preserved.
    """
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip().lower()


class IndicTextNormalizer:
    """Score-time normalizer for Brahmic/Indic scripts (whisper-normalizer's script
    normalizer, then the shared punctuation strip that keeps combining marks)."""

    def __init__(self, iso: str):
        from whisper_normalizer import indic

        self._script = getattr(indic, _INDIC_NORMALIZER_NAMES[iso])()

    def __call__(self, text: str) -> str:
        return _strip_punctuation_keep_marks(self._script(text or ""))


class ThaiTextNormalizer:
    """Score-time normalizer for Thai (pythainlp canonicalization, then the shared
    punctuation strip that keeps combining vowel/tone marks)."""

    def __call__(self, text: str) -> str:
        from pythainlp.util import normalize as thai_normalize

        return _strip_punctuation_keep_marks(thai_normalize(text or ""))


class TranscriptionScorer:
    """Scores a hypothesis transcript against a reference (WER/CER) for one language.

    Args:
        language (str): ISO code of the reference/hypothesis text; selects the
            normalizer (``en`` -> whisper EnglishTextNormalizer, Indic codes -> the
            script normalizer, ``th`` -> pythainlp, else BasicTextNormalizer) and the
            default options.
        normalize (bool | None): Override ``LANGUAGE_OPTIONS``; None keeps the default.
        remove_spaces (bool | None): Override; when True, ``word_error_rate`` is omitted.
        zh_trad_to_simp (bool | None): Override; converts to simplified Chinese last.
    """

    def __init__(
        self,
        language: str,
        normalize: bool | None = None,
        remove_spaces: bool | None = None,
        zh_trad_to_simp: bool | None = None,
    ):
        self.language = canonical_iso(language)
        opts = options_for(self.language)
        self.normalize = opts["normalize"] if normalize is None else normalize
        self.remove_spaces = (
            opts["remove_spaces"] if remove_spaces is None else remove_spaces
        )
        self.zh_trad_to_simp = (
            opts["zh_trad_to_simp"] if zh_trad_to_simp is None else zh_trad_to_simp
        )

        self._normalizer = None
        self.normalizer_name = "none"
        if self.normalize:
            if self.language == "en":
                from whisper_normalizer.english import EnglishTextNormalizer

                self._normalizer = EnglishTextNormalizer()
                self.normalizer_name = "english"
            elif self.language in _INDIC_NORMALIZER_NAMES:
                self._normalizer = IndicTextNormalizer(self.language)
                self.normalizer_name = f"indic:{_INDIC_NORMALIZER_NAMES[self.language]}"
            elif self.language == "th":
                self._normalizer = ThaiTextNormalizer()
                self.normalizer_name = "thai"
            else:
                from whisper_normalizer.basic import BasicTextNormalizer

                self._normalizer = BasicTextNormalizer()
                self.normalizer_name = "basic"

    @property
    def config_id(self) -> str:
        """A string that pins every choice affecting the numbers (library versions are
        recorded separately in the summary)."""
        return (
            f"{self.language}|normalizer={self.normalizer_name}"
            f"|remove_spaces={int(self.remove_spaces)}"
            f"|zhconv={'zh-cn' if self.zh_trad_to_simp else '0'}"
            "|order=normalize>strip>remove_spaces>zhconv"
            "|empty=both0.0/one1.0"
            f"|wer={'omitted' if self.remove_spaces else 'jiwer'}"
        )

    def _apply_normalize(self, text: str) -> str:
        """Normalize one string per this scorer's configuration (None-safe)."""
        text = text or ""
        if self._normalizer is not None:
            text = self._normalizer(text).strip()
        else:
            text = text.strip()
        if self.remove_spaces:
            text = text.replace(" ", "")
        if self.zh_trad_to_simp:
            import zhconv

            text = zhconv.convert(text, "zh-cn")
        return text

    def evaluate_sample(self, reference_text: str, predicted_text: str) -> dict:
        """Score one transcript against its reference.

        Args:
            reference_text (str): Ground-truth text (what was spoken).
            predicted_text (str): The ASR transcript.

        Returns:
            result (dict): ``reference_normalized``, ``predicted_normalized``,
                ``character_error_rate`` and, unless ``remove_spaces``,
                ``word_error_rate``.
        """
        ref_norm = self._apply_normalize(reference_text)
        hyp_norm = self._apply_normalize(predicted_text)
        result = {"reference_normalized": ref_norm, "predicted_normalized": hyp_norm}
        if ref_norm and hyp_norm:
            if not self.remove_spaces:
                result["word_error_rate"] = wer(ref_norm, hyp_norm)
            result["character_error_rate"] = cer(ref_norm, hyp_norm)
        else:
            error_val = 0.0 if (not ref_norm and not hyp_norm) else 1.0
            if not self.remove_spaces:
                result["word_error_rate"] = error_val
            result["character_error_rate"] = error_val
        return result
