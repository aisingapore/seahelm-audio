# Copied from ss-voice-toolkit@cac26e4 (evaluation/scorers/e2e/external_libraries/...). Keep in sync by re-copying, not editing.
"""Shared multiple-choice letter extraction, reused by VoiceBench (mmsu/openbookqa) and
SEA-HELM (global_mmlu_lite) e2e scoring.

Verbatim port of VoiceBench's own MCQ template-matching
(VoiceBench/src/evaluator/mcq.py), factored out of ``VoiceBenchScorer`` so more than one
e2e scorer can reuse it without duplicating the ~70-line template list.
"""

# mcq answer-extraction templates, verbatim from VoiceBench/src/evaluator/mcq.py
MCQ_TEMPLATES = [
    "答案是[CHOICE]",
    "答案是 [CHOICE]",
    "答案是选项[CHOICE]",
    "答案应该是[CHOICE]",
    "答案应该是 [CHOICE]",
    "答案就是选项[CHOICE]",
    "答案是‘[CHOICE]",
    "是[CHOICE]：",
    "答案选[CHOICE]",
    "[CHOICE]是正确",
    "选项[CHOICE]是最合适的",
    "answer is: **[CHOICE]",
    "answer is **[CHOICE]",
    "the answer to the question is: **[CHOICE]",
    "the answer to the multiple-choice question is **[CHOICE]",
    "the answer is '[CHOICE]'",
    "[CHOICE] is the best answer",
    "the answer is [CHOICE]",
    "the correct answer is [CHOICE]",
    "would select [CHOICE]",
    "would choose [CHOICE]",
    "would select option [CHOICE]",
    "would choose option [CHOICE]",
    'is "[CHOICE]"',
    'is "[CHOICE].',
    "is: **[CHOICE])",
    "is **[CHOICE],",
    "is **[CHOICE]:",
    "is **[CHOICE])",
    "is: **[CHOICE].",
    "is: **[CHOICE]:",
    "is **[CHOICE].",
    "be **[CHOICE],",
    "is: **[CHOICE]**",
    "answer is ([CHOICE])",
    "(option [CHOICE])",
    "option [CHOICE].",
    "option [CHOICE]:",
    "option [CHOICE],",
    "is [CHOICE]:",
    "is [CHOICE].",
    "is [CHOICE],",
    "is: [CHOICE].",
    "is ([CHOICE])",
    "is [CHOICE])",
    "is: [CHOICE])",
    "is: [CHOICE]",
    "is **[CHOICE]**",
    " [CHOICE].",
    " [CHOICE],",
    " [CHOICE]:",
    " [CHOICE])",
    "**[CHOICE].",
    "**[CHOICE])",
    '"[CHOICE].',
    '"[CHOICE],',
    '"[CHOICE]:',
    "([CHOICE])",
    '"[CHOICE]"',
]


def extract_choice(response: str) -> str | None:
    """Extract an A/B/C/D choice from a free-text response.

    Strips a leading ``<1>``/``<2>``/``<3>`` tag, then checks each of
    ``MCQ_TEMPLATES`` against the lowercased response for each choice
    letter; falls back to an exact-match or leading-punctuation check if no
    template matches.

    Args:
        response (str): The candidate's free-text response.

    Returns:
        choice (str | None): The extracted choice, uppercased
            ("A"/"B"/"C"/"D"), or None if no choice could be extracted.
    """
    response = (response or "").lower()
    if response[:3] in ("<1>", "<2>", "<3>"):
        response = response[3:].strip()
    for template in MCQ_TEMPLATES:
        for choice in ("a", "b", "c", "d"):
            if template.replace("[CHOICE]", choice) in response:
                return choice.upper()
    for choice in ("a", "b", "c", "d"):
        if response == choice:
            return choice.upper()
        for punc in (".", ",", ":", ")"):
            if response.startswith(choice + punc):
                return choice.upper()
    return None
