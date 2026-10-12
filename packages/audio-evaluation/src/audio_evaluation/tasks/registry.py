"""The task table: everything the scorer needs to know about one SEA-HELM task.

Merges the toolkit's four registries (``SeaHelmScorer._SUBSET_KIND`` / ``_DATA_PREP`` /
``_SUBSET_LANGUAGES`` and ``stages._METRIC_FOR_TASK``) into one record per task. Task
names are the directory names the pipeline writes (``outputs/pipeline/<task>/``), which
are also the dataset's parquet config names.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    """Static description of one task.

    Attributes:
        name (str): Task name as used in paths (``ifeval``, ``mmlu_lite``, ...).
        kind (str): Scoring kind: ``ifeval`` / ``mcq`` / ``qa`` / ``harm``.
        metric (str): Summary metric name (toolkit convention).
        scale (float): Multiplier applied to per-sample scores in the summary
            (ifeval reports 0-100, everything else 0-1).
        data_dir (str): Parquet config directory under the dataset root.
        filename_prefix (str): Prefix of every sample ``filename`` for this task.
        gold_columns (tuple[str, ...]): Parquet columns that hold gold.
    """

    name: str
    kind: str
    metric: str
    scale: float
    data_dir: str
    filename_prefix: str
    gold_columns: tuple[str, ...]


TASKS: dict[str, TaskSpec] = {
    "ifeval": TaskSpec(
        name="ifeval",
        kind="ifeval",
        metric="overall_lang_normalized_acc",
        scale=100.0,
        data_dir="ifeval",
        filename_prefix="seahelm_ifeval_",
        gold_columns=("metadata", "kwargs"),
    ),
    "mmlu_lite": TaskSpec(
        name="mmlu_lite",
        kind="mcq",
        metric="accuracy",
        scale=1.0,
        data_dir="mmlu_lite",
        filename_prefix="seahelm_global_mmlu_lite_",
        gold_columns=("label", "subject"),
    ),
    # Not yet published as audio datasets; kept so the scorer is ready when they are.
    "question_answering": TaskSpec(
        name="question_answering",
        kind="qa",
        metric="f1",
        scale=1.0,
        data_dir="question_answering",
        filename_prefix="seahelm_question_answering_",
        gold_columns=("label",),
    ),
    "safeguard": TaskSpec(
        name="safeguard",
        kind="harm",
        metric="refusal_rate",
        scale=1.0,
        data_dir="safeguard",
        filename_prefix="seahelm_safeguard_",
        gold_columns=(),
    ),
}


def resolve_task(
    results_path: str | Path | None = None,
    filenames: list[str] | None = None,
) -> TaskSpec:
    """Pick the task from the results file's parent dir, else from the filename prefix.

    There is no explicit override: a results file is one task, and the task is always
    recoverable from where the file sits (``<run>/<task>/results.jsonl``) or from the
    dataset filenames inside it (``seahelm_<task>_...``).

    Args:
        results_path (str | Path | None): Path of the results file; its parent directory
            name is tried as the task name (``outputs/pipeline/<task>/results.jsonl``).
        filenames (list[str] | None): Sample filenames; the first one's prefix is tried.

    Returns:
        spec (TaskSpec): The resolved task.

    Raises:
        ValueError: If nothing identifies a known task.
    """
    if results_path is not None:
        parent = Path(results_path).resolve().parent.name
        if parent in TASKS:
            return TASKS[parent]
    if filenames:
        for spec in TASKS.values():
            if filenames[0].startswith(spec.filename_prefix):
                return spec
    raise ValueError(
        "could not infer the task: keep the results file under <run>/<task>/results.jsonl "
        f"with <task> one of {sorted(TASKS)}, and use the dataset's sample filenames"
    )
