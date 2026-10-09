# model-pipeline

A thin reference cascade that turns the SEA-HELM Audio prompt clips into the results
file that `audio-evaluation` scores: prompt audio -> STT -> LLM -> reply -> TTS.

It is optional. Most users of this repo only need `audio-evaluation`. Use this package
if you do not yet have results to evaluate, or as a template for wiring up your own
models.

## How the repo is meant to be used

1. **Read the results format.** The file `audio-evaluation` consumes is the real
   contract; everything else here exists to produce or consume it.
2. **Evaluate.** Run `audio-evaluation` on your `results.jsonl`.
3. **Don't have the data?** The prompt audio and gold labels are the Hugging Face
   dataset `aisingapore/seahelm-audio`, snapshotted into `data/seahelm_audio_data/`
   (one parquet config per task, one split per language). A download step will live
   with the dataset code, not in this package.
4. **Don't have results?** Run this pipeline (below). It needs step 3 first.
5. **Authoring new prompt or audio sets** is a separate, rarely needed path and is
   documented with the dataset code.

Two consequences of that ordering:

- Evaluation still needs gold labels, which come from the dataset. Either the scorer
  reads them from the materialised prompts file, or each results row carries its gold
  fields inline. The format doc states which.
- Scoring never touches audio, so evaluation-only users should not have to pull the
  1.4 GB snapshot. The download step should offer a text-only mode that fetches just
  the prompt and gold columns.

## Running this pipeline

From the repo root:

```bash
uv run python -m model_pipeline.main --limit 3
```

- `--limit N` stops after N rows per task (0 = all). `--out` sets the output
  directory (default `outputs/pipeline`).
- Both tasks under `data/seahelm_audio_data/` are processed, each into
  `outputs/pipeline/<task>/` with `audio_in/` (the extracted prompt WAVs),
  `audio_out/<language>/` (the spoken replies, 24 kHz mono WAV) and `results.jsonl`.
- Runs are resumable: rows whose `filename` already appears in `results.jsonl` are
  skipped, so a rerun only does the missing samples.
- Credentials are read from `packages/model-pipeline/.env`: `ELEVENLABS_API_KEY`
  (Scribe STT and TTS) and `SEALION_API_KEY` (LLM). See `.example.env`.

The default models are ElevenLabs Scribe v2 (batch), SEA-LION v4.5 27B via the
OpenAI-compatible endpoint with thinking disabled, and ElevenLabs v4 TTS.

## Swapping models

Each stage is an abstract base class under `src/model_pipeline/models/`
(`stt/stt_base.py`, `llm/llm_base.py`, `tts/tts_base.py`) with a single `generate`
method. Implement the base for your model, then change the three constructor calls in
`run()` in `main.py`. The loop in `run_sample` is typed against the bases and does not
depend on any particular provider.
