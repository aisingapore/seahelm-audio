"""SEA-HELM global_mmlu_lite (MCQ) scoring: exact match on the extracted choice."""

from .answer_extraction import extract_answer_choice_with_stage


def score_response_detailed(response: str, label: str) -> dict:
    """Score one MCQ response and report how the choice was read.

    Args:
        response (str): The candidate's free-text response.
        label (str): The correct choice letter (e.g. "B").

    Returns:
        result (dict): ``choice`` (the extracted letter or None), ``parse_status``
            (``template`` / ``tagged`` / ``ordinal`` / ``none``) and ``score`` (1.0 if the
            extracted choice matches ``label`` case-insensitively, else 0.0 -- including when
            no choice could be extracted; there is no abstain credit).
    """
    choice, stage = extract_answer_choice_with_stage(response)
    ref = (label or "").strip().upper()
    return {
        "choice": choice,
        "parse_status": stage,
        "score": 1.0 if choice == ref else 0.0,
    }


def score_response(response: str, label: str) -> float:
    """Score one MCQ response against the correct choice letter.

    Args:
        response (str): The candidate's free-text response.
        label (str): The correct choice letter (e.g. "B").

    Returns:
        score (float): 1.0 if the extracted choice matches ``label``, else 0.0.
    """
    return score_response_detailed(response, label)["score"]
