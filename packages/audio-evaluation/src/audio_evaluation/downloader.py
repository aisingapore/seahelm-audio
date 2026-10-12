"""Download the SEA-HELM Audio dataset snapshot from the Hugging Face Hub."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

DATASET_ID = "aisingapore/seahelm-audio"


def download(
    dest: str | Path, revision: str | None = None, token: str | None = None
) -> Path:
    """Snapshot the dataset (parquet shards + README) into ``dest``.

    Args:
        dest (str | Path): Local directory to populate (``data/seahelm_audio_data``).
        revision (str | None): Hub commit/branch to pin; None means the Hub default.
            Pin it for reproducible scoring and record it alongside results.
        token (str | None): Hub token; defaults to ``HF_TOKEN`` from the environment
            or a ``.env`` file (needed while the dataset is private).

    Returns:
        path (Path): The populated directory.
    """
    load_dotenv()
    token = token or os.environ.get("HF_TOKEN")
    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(dest),
        revision=revision,
        token=token,
    )
    logger.info(f"downloaded {DATASET_ID} (revision={revision or 'default'}) -> {path}")
    return Path(path)
