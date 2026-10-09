---
dataset_info:
- config_name: ifeval
  features:
  - name: prompts
    list:
    - name: text
      dtype: string
  - name: metadata
    struct:
    - name: category
      dtype: string
    - name: language
      dtype: string
    - name: subcategory
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
  - name: filename
    dtype: string
  - name: language
    dtype: string
  - name: text
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
    num_bytes: 44824433
    num_examples: 105
  - name: indonesian
    num_bytes: 60406000
    num_examples: 105
  - name: tamil
    num_bytes: 59096627
    num_examples: 105
  download_size: 164218420
  dataset_size: 164327060
- config_name: mmlu_lite
  features:
  - name: filename
    dtype: string
  - name: label
    dtype: string
  - name: subject
    dtype: string
  - name: language
    dtype: string
  - name: text
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
  download_size: 1285120277
  dataset_size: 1285294117
configs:
- config_name: ifeval
  data_files:
  - split: english
    path: ifeval/english-*
  - split: indonesian
    path: ifeval/indonesian-*
  - split: tamil
    path: ifeval/tamil-*
- config_name: mmlu_lite
  data_files:
  - split: chinese
    path: mmlu_lite/chinese-*
  - split: english
    path: mmlu_lite/english-*
---
