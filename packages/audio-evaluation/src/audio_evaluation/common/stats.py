"""Aggregate statistics (ported from ss-voice-toolkit@cac26e4 evaluation/scorers/base.py)."""


def aggregate_stats(values: list[float]) -> dict:
    """Compute avg/min/max over a list of metric values.

    Args:
        values (list[float]): Per-sample metric values to summarize.

    Returns:
        stats (dict): ``{"avg", "min", "max"}`` rounded to 4 decimal places,
            or all-None (``{"avg": None, "min": None, "max": None}``) when
            ``values`` is empty.
    """
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }
