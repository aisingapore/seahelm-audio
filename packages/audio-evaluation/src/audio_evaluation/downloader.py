# Download the SEAHELM Audio dataset from Huggingface
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

load_dotenv()

DATASET_ID = "aisingapore/seahelm-audio"
HF_TOKEN = os.environ.get("HF_TOKEN")  # None is fine for public datasets

path = snapshot_download(
    repo_id=DATASET_ID,
    repo_type="dataset",
    local_dir=Path(
        "/projects_vol/gp_david.ongtw/speech_spoke/yixian/repos/seahelm-audio/data/seahelm_audio_data"
    ),
    token=HF_TOKEN,
)

logger.info(f"Downloaded to {path}")
