---
dataset_info:
- config_name: ifeval
  features:
  - name: filename
    dtype: string
  - name: language
    dtype: string
  - name: text
    dtype: string
  - name: kwargs
    list:
    - name: end_phrase
      dtype: string
    - name: first_word
      dtype: string
    - name: forbidden_words
      list: string
    - name: frequency
      dtype: int64
    - name: keyword
      dtype: string
    - name: keywords
      list: string
    - name: num_bullets
      dtype: int64
    - name: num_highlights
      dtype: int64
    - name: num_paragraphs
      dtype: int64
    - name: num_placeholders
      dtype: int64
    - name: num_sections
      dtype: int64
    - name: num_sentences
      dtype: int64
    - name: num_words
      dtype: int64
    - name: number
      dtype: string
    - name: postscript_marker
      dtype: string
    - name: prompt_to_repeat
      dtype: string
    - name: relation
      dtype: string
    - name: response_language
      dtype: string
    - name: section_title
      dtype: string
  - name: metadata
    struct:
    - name: category
      dtype: string
    - name: language
      dtype: string
    - name: subcategory
      dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 24000
  - name: tts_model
    dtype: string
  - name: audio_duration_s
    dtype: float64
  splits:
  - name: english
    num_bytes: 44807176
    num_examples: 105
  - name: indonesian
    num_bytes: 60386040
    num_examples: 105
  - name: malay
    num_bytes: 63395174
    num_examples: 105
  - name: tamil
    num_bytes: 59040461
    num_examples: 105
  download_size: 227562208
  dataset_size: 227628851
- config_name: mmlu_lite
  features:
  - name: filename
    dtype: string
  - name: language
    dtype: string
  - name: text
    dtype: string
  - name: label
    dtype: string
  - name: subject
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 24000
  - name: tts_model
    dtype: string
  - name: audio_duration_s
    dtype: float64
  splits:
  - name: chinese
    num_bytes: 643670435
    num_examples: 400
  - name: english
    num_bytes: 641623682
    num_examples: 400
  download_size: 1285120301
  dataset_size: 1285294117
configs:
- config_name: ifeval
  data_files:
  - split: english
    path: ifeval/english-*
  - split: indonesian
    path: ifeval/indonesian-*
  - split: malay
    path: ifeval/malay-*
  - split: tamil
    path: ifeval/tamil-*
- config_name: mmlu_lite
  data_files:
  - split: chinese
    path: mmlu_lite/chinese-*
  - split: english
    path: mmlu_lite/english-*
license: mit
task_categories:
- automatic-speech-recognition
language:
- en
- id
- ta
- ms
- zh
tags:
- seahelm_audio
- aisingapore
- voice-evaluation
- southeast-asia
- speech-llm
- voice-agents
pretty_name: SEA-HELM Audio (OmniVoice Baseline)
---

# Dataset Card for SEA-HELM Audio (OmniVoice Baseline)

_[Last Update: 21 Oct 2026]_

## Dataset Summary

**SEA-HELM Audio** is a Southeast Asian multilingual voice benchmark designed to quantify the performance degradation introduced by the speech modality in end-to-end conversational voice stacks across Southeast Asian languages.

Adapted from AI Singapore's text-based **SEA-HELM** (Southeast Asian Holistic Evaluation of Language Models) benchmark, this dataset provides a standardized input speech corpus generated exclusively using **OmniVoice**. By pairing synthesized input audio with ground-truth prompt text, the dataset enables **dual-branch evaluation** to isolate the ASR modality gap

$$\Delta (Modality Gap) = \text{Score}_{\text{text}} - \text{Score}_{\text{speech}}$$

under deterministic, rule-based verification.

## Dataset Description

- **Developed by:** AI Products Pillar, AI Singapore
- **Funded by:** National Research Foundation Singapore
- **Shared by:** AI Products Pillar, AI Singapore
- **Language(s):** English, Indonesian, Malay, Mandarin and Tamil
- **License:** [MIT](https://mit-license.org/)
- **Generated from model:** [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)


## Supported Tasks and Languages

The corpus covers two primary evaluation tasks across five target languages:

| Task Subset | Task Type | Languages Covered | Evaluation Protocol |
| :--- | :--- | :--- | :--- |
| `seahelm_ifeval` | Instruction Following | English (`en`), Indonesian (`id`), Malay (`ms`), Tamil (`ta`) | Rule-based constraint verifier + language compliance check |
| `seahelm_global_mmlu_lite` | Multiple-Choice QA | Chinese (`zh`), English (`en`) | Deterministic option letter extraction (`A`–`D`) |

### Language Breakdown
* **Chinese (`zh`)**: Simplified Chinese
* **English (`en`)**: Regional and general English
* **Indonesian (`id`)**: Bahasa Indonesia
* **Malay (`ms`)**: Bahasa Melayu
* **Tamil (`ta`)**: Regional Tamil

## Data Generation Methodology

All input audio in this public release was generated using **OmniVoice** to establish a unified, open-weight acoustic baseline]. 

1. **Acoustic Standardization:** Prompts are read aloud using OmniVoice to simulate standardized user acoustic inputs across all regional target languages.
2. **Linguistic Adaptation:** Prompts are formatted to match natural spoken cadence. Instruction-following prompts (`seahelm_ifeval`) are spoken verbatim as written, while multiple-choice questions (`seahelm_global_mmlu_lite`) incorporate task preambles alongside question bodies and options.
3. **Deterministic Answer Isolation:** Ground-truth labels, constraint parameters, and answer keys are withheld from the audio signal and stored purely in the text sidecar for scoring time.

## Dataset Structure

### Data Instances

Each instance represents a single evaluation prompt, containing the ground-truth text, metadata, and the synthesized audio array.

An example from the dataset looks as follows:

```
{
  "sample_id": "ifeval_ms_001",
  "audio": {
    "path": "ifeval_ms_001.wav",
    "array": [0.001, 0.002, ...],
    "sampling_rate": 16000
  },
  "text": "Tuliskan sebuah karangan pendek. Pastikan ia kurang daripada 50 perkataan.",
  "language": "ms",
  "task_subset": "seahelm_ifeval"
}

```
### Data Fields

* `id` *(string)*: Unique sample identifier.
* `language` *(string)*: ISO language code (`zh`, `en`, `id`, `ms`, `ta`).
* `subset` *(string)*: SEA-HELM subtask (`seahelm_ifeval` or `seahelm_global_mmlu_lite`).
* `audio` *(datasets.Audio)*: Audio object containing synthesized speech waveform (24 kHz)) generated via OmniVoice.
* `prompt_text` *(string)*: The original ground-truth text prompt supplied to the text baseline branch.
* `tts_engine` *(string)*: Input acoustic synthesizer used (`OmniVoice`).
* `target_label` / `constraints` *(dict/string)*: Ground-truth target key used by the programmatic verification engine.


### Annotations

*No human annotations were added.* The dataset relies on the ground-truth labels and rule-based verification scripts provided by the parent SEA-HELM benchmark.


## Considerations for Using the Data

### Social Impact and Limitations

- **Synthetic Acoustic Distribution:** The audio in this dataset is entirely generated via OmniVoice. While this provides a highly controlled, reproducible environment for isolating ASR-LLM degradation, it does not perfectly replicate the acoustic noise, hesitation markers, and varied microphone quality of authentic human speech in real-world deployments.
- **Dialect Representation:** Users should note that the TTS models may impose specific dialectal or prosodic assumptions on the generated regional languages (e.g., specific accents for Tamil or Malay) that might not encompass all local variations.


## Quickstart Usage

You can load the dataset directly via the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the full SEA-HELM Audio benchmark
dataset = load_dataset("aisingapore/seahelm-audio")

# Load a specific language and subset
ifeval_ms = load_dataset(
    "aisingapore/seahelm-audio", 
    name="seahelm_ifeval", 
    split="malay"
)

# Inspect a single sample
sample = ifeval_ms[0]
print("Prompt:", sample["prompt_text"])
print("Audio Array Shape:", sample["audio"]["array"].shape)

```

## Evaluation Protocol

When evaluating a voice stack on this dataset, compare two parallel runs:

1. **Text Branch:** Supply `prompt_text` directly to your LLM to obtain Score<sub>text</sub>.

2. **Speech Branch:** Pass the `audio` signal through your ASR model, then feed the resulting transcript to your LLM to obtain Score<sub>speech</sub>.

**Repository:** [aisingapore/seahelm-audio](https://github.com/aisingapore/seahelm-audio/tree/main)

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
