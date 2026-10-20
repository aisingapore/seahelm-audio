<div align="center">

# SEA-HELM Audio

<p align="center">
   <strong>An adaption of <a href="https://github.com/aisingapore/SEA-HELM">SEA-HELM</a> for speech agents</strong><br><br>
   <a href="https://arxiv.org/abs/2610.19869"><img src="https://img.shields.io/badge/TechnicalReport-arXiv-B31B1B?logo=arxiv&logoColor=white" alt="Technical Report"></a>
</p>

</div>

## Overview
SEA-HELM Audio is a multilingual voice benchmark designed to quantify the performance degradation introduced by the speech modality in end-to-end conversational voice stacks across Southeast Asian languages.

Adapted from AI Singapore's text-based SEA-HELM (Southeast Asian Holistic Evaluation of Language Models) benchmark, this dataset provides a standardized input speech corpus generated exclusively using OmniVoice. By pairing synthesized input audio with ground-truth prompt text, the dataset enables dual-branch evaluation to isolate the ASR modality gap.

The dataset card on Hugging Face describes the tasks and languages in detail. This README covers how to use the dataset to evaluate a voice stack.

## Links
- [Dataset](https://huggingface.co/datasets/aisingapore/seahelm-audio)
- [Technical Report](https://arxiv.org/abs/2610.19869)
- [Voice Agent Playground](https://voice-agent.sea-lion.ai/playground/voice-agent)

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Every command below runs from the repo root.

```bash
uv sync --all-packages    # one shared venv for both packages (a plain `uv sync` installs neither)
```

1. **Download the dataset.**

   ```bash
   uv run audio-evaluation download --dest <DESTINATION_FOLDER>

   # example
   uv run audio-evaluation download --dest data/seahelm_audio_data
   ```

2. **Produce results.** Run your voice stack over the prompt audio and write one `results.jsonl` per task in the [results format](#results-format) below. Either adapt the reference cascade in `packages/model-pipeline` by swapping its STT / LLM / TTS (optional) classes to models you are interested, or produce the file directly. Speech-to-text LLMs that skip a separate STT stage work too.

   ```bash
   uv run python -m model_pipeline.main --out <OUTPUT_FOLDER>

   # example (reference pipeline; API keys in packages/model-pipeline/.env)
   uv run python -m model_pipeline.main --out outputs/pipeline
   ```

3. **Scoring.** 

   ```bash
   uv run audio-evaluation score --results <RESULTS_FILE> --dry-run
   uv run audio-evaluation score --results <RESULTS_FILE>

   # example
   uv run audio-evaluation score --results outputs/pipeline/ifeval/results.jsonl --dry-run
   uv run audio-evaluation score --results outputs/pipeline/ifeval/results.jsonl
   ```

   This writes `scored.jsonl` (one record per sample) and `summary.json` (the aggregate metrics) to `<OUTPUT_FOLDER>`.

### Example script

[`run_example.sh`](run_example.sh) downloads the dataset, runs the reference pipeline, then scores every task. Needs the API keys in `packages/model-pipeline/.env` (see [Model Pipeline](#model-pipeline-rootpackagesmodel-pipeline)).

## Results format (from pipeline)
The results file provide the necessary interface for using the audio evaluation. Ensure that the following fields are present.

| field | meaning |
|---|---|
| `filename` | sample id from the dataset, e.g. `seahelm_ifeval_en_000000.wav` |
| `language` | ISO 639-1 code of the sample (`en`, `ms`, `zh`, ...)|
| `spoken_text` | the dataset's prompt text, used as the WER/CER reference |
| `transcript` | your STT output for the prompt audio, i.e. what the LLM saw |
| `reply` | your system's answer to the prompt; used for scoring |

Sample `JSONL` formatting

```json
{
  "filename": "seahelm_ifeval_en_000001.wav",
  "language": "en",
  "spoken_text": "Could you give me a short summary of The Lord of the Rings that is child-friendly?\nFirst, repeat \"Could you give me a short summary ...\" word for word, then give your answer.",
  "transcript": "Could you give me a short summary of The Lord of the Rings that is child-friendly? First, repeat. Could you give me a short summary ... then give your answer.",
  "reply": "Could you give me a short summary of The Lord of the Rings that is child-friendly? ... Frodo, a small hobbit, is given a magic ring ...",
  "audio_in": "outputs/pipeline/ifeval/audio_in/en/seahelm_ifeval_en_000001.wav",
  "audio_out": null
}
{
  "filename": "seahelm_global_mmlu_lite_zh_000001.wav",
  "language": "zh",
  "spoken_text": "仅使用以下格式回答。\n\"答案:\"\n将 答案 替换为您选择的答案。使用字母 第一、第二、第三 或 第四 作为答案。\n\n问题:\n在月球上，最重的是什么？\n第一: 五千克羽毛\n第二: 一千克羽毛\n第三: 在地球上测量重五磅的砖\n第四: 一千克砖",
  "transcript": "仅使用以下格式回答答案，将答案替换为您选择的答案。使用字母第一、第二、第三或第四作为答案。问题：这在月球上最重的是什么？第一五千克羽毛，第二一千克羽毛，第三在地球上测量重五磅的砖，第四一千克砖。",
  "reply": "第一",
  "audio_in": "outputs/pipeline/mmlu_lite/audio_in/zh/seahelm_global_mmlu_lite_zh_000001.wav",
  "audio_out": null
}
```

`audio_in` and `audio_out` are extras the reference pipeline adds (`audio_out` is filled only with `--tts`); they are optional.

### Where to put the results file

Each result file should be in a named folder after that task. 

```
<RUN_FOLDER>/
  ifeval/results.jsonl
  mmlu_lite/results.jsonl
```

Currently supported task names are `ifeval`, `mmlu_lite`.

## Packages
We include 2 packages (`root/packages`) within the repo; `model-pipeline` and `audio-evaluation`.
- `model-pipeline` provides a guideline on how to arrange the outputs, and a possible cascaded system flow to follow. However direct speech-to-text models will work as well.
- `audio-evaluation` consists of the actual evaluation code, scoring the outputs from model pipeline

### Model Pipeline `root/packages/model-pipeline`
Model pipeline provides the scaffolding for an end-to-end cascaded system. As a default, we use ElevenLabs for STT and SEALION as the underlying LLM. Optionally, we provide TTS as well if you wish to further evaluate the LLM's response as audio.

Credentials go in `packages/model-pipeline/.env`:

```
ELEVENLABS_API_KEY=...   # Scribe v2 for STT, and eleven_v4 for TTS when --tts is set
SEALION_API_KEY=...      # aisingapore/Qwen-SEA-LION-v4.5-27B-IT via the OpenAI-compatible endpoint
```

```bash
uv run python -m model_pipeline.main [--out <OUTPUT_FOLDER>] [--limit <N>] [--tts]

# examples
uv run python -m model_pipeline.main --limit 3                                  # smoke test: first 3 rows of each task, STT + LLM only
uv run python -m model_pipeline.main                                            # every row of every task, into outputs/pipeline
uv run python -m model_pipeline.main --out outputs/pipeline_sealion_v45 --tts   # named run, also synthesise each reply to audio
```

| option | required | meaning |
|---|---|---|
| `--out DIR` | optional | output directory, default `outputs/pipeline` |
| `--limit N` | optional | stop after N rows per task; 0 (the default) means all |
| `--tts` | optional | synthesise each reply into `audio_out/`, off by default |

There are no required flags: a bare run processes every row of every task under `data/seahelm_audio_data/`. The two API keys in `.env` are the only mandatory inputs.

Each task writes `<OUTPUT_FOLDER>/<task>/results.jsonl`, a `metadata.json` naming the models used, `audio_in/<language>/` with the extracted prompt clips and, with `--tts`, `audio_out/<language>/`. Runs are resumable: rows whose `filename` is already in `results.jsonl` are skipped.

To plug in your own model, implement the base class for that stage under `src/model_pipeline/models/<stt|llm|tts>/` and change the constructor call in `run()` in `main.py`.

### Audio Evaluation `root/packages/audio-evaluation`
This contains the actual evaluation code, taking in the results and scoring them against the dataset to get the final score. Ensure that all files you want to process follow the data format previously mentioned.

```bash
uv run audio-evaluation download --dest <DESTINATION_FOLDER> [--revision <COMMIT_SHA>]
uv run audio-evaluation score --results <RESULTS_FILE> [--dry-run] [--data-root <DATASET_FOLDER>] [--out <OUTPUT_FOLDER>]

# download example (optionl)
uv run audio-evaluation download --dest data/seahelm_audio_data --revision 627a26bfde4f9b12b287abf500208221bff1af3a

# scoring example
uv run audio-evaluation score --results outputs/pipeline/ifeval/results.jsonl --dry-run
uv run audio-evaluation score --results outputs/pipeline/mmlu_lite/results.jsonl --data-root data/seahelm_audio_data --out outputs/evaluation/mmlu_lite
```

| option | command | required | meaning |
|---|---|---|---|
| `--results` | `score` | **required** | the `results.jsonl` to score |
| `--dest` | `download` | optional | where to put the snapshot, default `data/seahelm_audio_data` |
| `--revision` | `download` | optional | Hub commit to pin; default is the Hub's current revision |
| `--dry-run` | `score` | optional | print the join report (rows matched to gold, missing rows, language or filename-prefix mismatches) and stop; exits non-zero on problems |
| `--data-root` | `score` | optional | dataset snapshot root, default `data/seahelm_audio_data` |
| `--out` | `score` | optional | output directory, default `outputs/evaluation/<task>` |

Outputs per run:

- `scored.jsonl`: one self-describing record per sample with the input row, the gold inline, WER/CER and the per-sample score with its task-specific extras.
- `summary.json`: the task metric, WER/CER stats, counts (rows, nulls, unparsed), per-subcategory breakdown for ifeval, the languages seen, normalizer config ids, library versions and a copy of the pipeline's `metadata.json` if one sits next to the results.

```bash
uv run pytest packages/audio-evaluation/tests -q    # scoring tests, including a frozen golden set
```

## Development

Formatting and linting are ruff via pre-commit:

```bash
uv tool install pre-commit    # once
pre-commit install            # once per clone, runs on every commit
pre-commit run --all-files    # on demand
```
## Technical Report and Citation

If you utilize this corpus or our methodology in your work, please cite the corresponding paper:

```
@article{seahelm_audio_bench_en_zh_ta_ms_2026,
    title={SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets}, 
    author={Aulia Adila and Yixian Tan and Jonathan Heng and Kunat Pipatanakul and Donghang Wu and Anocha Sutaveephamochanon},
    year={2026},
    eprint={2610.19869},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2610.19869}, 
}
```
## Team

AI Products Pillar, AI Singapore

## Acknowledgement

This project is supported by the National Research Foundation Singapore and Infocomm Media Development Authority (IMDA), Singapore under its National Large Language Model Funding Initiative.

## Contact

[**sealion@aisingapore.org**](mailto:sealion@aisingapore.org) 
