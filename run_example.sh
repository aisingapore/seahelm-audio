#!/usr/bin/env bash
# Download the dataset, run the reference pipeline, then score each task.
# Needs API keys in packages/model-pipeline/.env. Run from the repo root.
#
#   bash run_example.sh                  # full run, every row of every task
#   LIMIT=3 bash run_example.sh          # smoke test, 3 rows per task
#   TTS=1 bash run_example.sh            # full run, also synthesise each reply to audio
set -euo pipefail

DATA_DIR="data/seahelm_audio_data"   # dataset download (model_pipeline reads this exact path)
RUN_DIR="outputs/pipeline"           # pipeline output: <task>/results.jsonl
EVAL_DIR="outputs/evaluation"        # scoring output: <task>/scored.jsonl, summary.json
LIMIT="${LIMIT:-0}"                  # rows per task; 0 (default) = all
TTS="${TTS:-0}"                      # 1 = add --tts

TTS_FLAG=""
if [[ "${TTS}" == "1" ]]; then TTS_FLAG="--tts"; fi

uv sync --all-packages
uv run audio-evaluation download --dest "${DATA_DIR}"

uv run python -m model_pipeline.main --out "${RUN_DIR}" --limit "${LIMIT}" ${TTS_FLAG}

for task in ifeval mmlu_lite; do
  uv run audio-evaluation score \
    --results "${RUN_DIR}/${task}/results.jsonl" \
    --data-root "${DATA_DIR}" \
    --out "${EVAL_DIR}/${task}"
done
