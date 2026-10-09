"""Reference cascade: parquet row -> audio file -> STT -> LLM -> reply (-> TTS).

Reads the Hugging Face parquet shards under ``data/seahelm_audio_data/<task>/`` (each
row embeds the prompt WAV), writes every input clip to disk, and runs the stages in
sequence. One JSON line per sample is appended to ``<out>/<task>/results.jsonl``.

Run from the repo root (paths below are relative to it)::

    uv run python -m model_pipeline.main --limit 3          # STT + LLM only
    uv run python -m model_pipeline.main --limit 3 --tts    # also synthesize the reply

Output layout per task::

    <out>/<task>/audio_in/<language>/<filename>.wav    extracted prompt audio
    <out>/<task>/audio_out/<language>/<filename>.wav   spoken reply (only with --tts)
    <out>/<task>/results.jsonl                         one record per sample

Resumable: rows whose ``filename`` already appears in ``results.jsonl`` are skipped.
"""

import argparse
import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from .models.llm.llm_base import LLM
from .models.stt.stt_base import STT
from .models.tts.tts_base import TTS


def parquet_files(data: Path) -> list[Path]:
    """Resolve a data path to the parquet shards it refers to.

    Args:
        data (Path): A single ``.parquet`` file, or a directory containing shards
            (e.g. ``data/seahelm_audio_data/ifeval``).

    Returns:
        files (list[Path]): The shard paths in sorted order; a one-element list when
            ``data`` is a file.

    Raises:
        FileNotFoundError: If ``data`` does not exist. Resolved against the current
            working directory, so this fires when the script is not run from the repo
            root.
    """
    if not data.exists():
        raise FileNotFoundError(f"data path not found: {data.resolve()}")
    return sorted(data.glob("*.parquet")) if data.is_dir() else [data]


def count_rows(data: Path) -> int:
    """Count the rows across all shards without reading any row data.

    Args:
        data (Path): A parquet file or a directory of shards (see ``parquet_files``).

    Returns:
        total (int): Sum of ``num_rows`` from each shard's parquet metadata.
    """
    return sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files(data))


def load_rows(data: Path) -> Iterator[dict]:
    """Yield one dict per dataset row, shard by shard.

    Each row carries the dataset's columns verbatim: ``filename``, ``language`` (ISO
    code), ``text`` (the spoken prompt), the task's gold fields, and ``audio`` as a
    dict with ``bytes`` (the complete WAV file) and ``path`` (its original filename).

    Args:
        data (Path): A parquet file or a directory of shards (see ``parquet_files``).

    Yields:
        row (dict): One dataset row, in shard order then row order.
    """
    for f in parquet_files(data):
        yield from pq.read_table(f).to_pylist()


def extract_audio(row: dict, audio_dir: Path) -> Path:
    """Write a row's embedded WAV bytes to disk, keeping the original filename.

    Idempotent: an existing file is left untouched, so reruns do not rewrite audio.

    Args:
        row (dict): A dataset row from ``load_rows``; reads ``row["audio"]["bytes"]``
            and ``row["audio"]["path"]``.
        audio_dir (Path): Directory to write into; created if missing.

    Returns:
        path (Path): ``audio_dir / <original filename>``.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / row["audio"]["path"]
    if not path.exists():
        path.write_bytes(row["audio"]["bytes"])
    return path


def run_sample(
    row: dict, stt: STT, llm: LLM, out: Path, tts: TTS | None = None
) -> dict:
    """Run one row through the cascade: audio in -> transcript -> LLM reply -> audio out.

    Args:
        row (dict): A dataset row from ``load_rows``.
        stt (STT): STT model; its ``language_code`` is set from the row's ``language``
            before transcribing.
        llm (LLM): LLM that answers the transcript.
        out (Path): The task's output directory (``<out>/<task>``); audio is written
            under ``audio_in/<language>/`` and ``audio_out/<language>/`` inside it.
        tts (TTS | None): Optional TTS. When None the reply is not  synthesized and
            ``audio_out`` is None in the result.

    Returns:
        result (dict): ``filename``, ``language``, ``spoken_text`` (the dataset
            prompt), ``transcript`` (STT output), ``reply`` (LLM output), ``audio_in``
            (path of the extracted prompt WAV) and ``audio_out`` (path of the spoken
            reply, or None).
    """
    language = row.get("language") or "unknown"
    stt.language_code = language

    audio_in = extract_audio(row, out / "audio_in" / language)
    transcript = stt.generate(str(audio_in))
    reply = llm.generate(transcript)

    audio_out = None
    if tts is not None:
        audio_out = tts.generate(
            prompt=reply,
            out_path=out / "audio_out" / language / f"{audio_in.stem}.wav",
        )

    return {
        "filename": row["filename"],
        "language": language,
        "spoken_text": row["text"],
        "transcript": transcript,
        "reply": reply,
        "audio_in": str(audio_in),
        "audio_out": audio_out,
    }


def load_models(include_tts: bool = False) -> dict[str, STT | LLM | TTS | None]:
    """Instantiate the models for this run.

    This is the one place that names concrete providers; everything else in the
    module is typed against the abstract bases. Change the three classes here to run
    the cascade with your own models.

    Args:
        include_tts (bool): Also construct the TTS model. Off by default because
            synthesizing every reply is the expensive stage.

    Returns:
        models (dict[str, STT | LLM | TTS | None]): Keys ``"stt"``, ``"llm"`` and
            ``"tts"`` mapping to model instances; ``"tts"`` is None unless
            ``include_tts`` is set.
    """
    from .models.llm.sealion import SeaLionLLM
    from .models.stt.elevenlabs_scribe import ElevenLabsScribeSTT

    tts: TTS | None = None
    if include_tts:
        from .models.tts.elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS()

    return {"stt": ElevenLabsScribeSTT(), "llm": SeaLionLLM(), "tts": tts}


def run(data: Path, out: Path, limit: int = 0, include_tts: bool = False) -> None:
    """Run the cascade over one task's rows, appending results as they complete.

    Args:
        data (Path): The task's parquet directory (or a single shard).
        out (Path): The task's output directory; created if missing.
        limit (int): Process only the first N rows (0 = all). Counts rows read,
            including ones skipped as already done.
        include_tts (bool): Also synthesize each reply to ``audio_out/``.

    Returns:
        None: Writes ``<out>/results.jsonl`` and the audio directories as a side
            effect.
    """
    total = count_rows(data)  # validates the data path before any output is created
    if limit:
        total = min(total, limit)

    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done: set[str] = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            done = {json.loads(line)["filename"] for line in f if line.strip()}

    models = load_models(include_tts=include_tts)
    stt, llm, tts = models["stt"], models["llm"], models["tts"]

    rows = load_rows(data)
    if limit:
        rows = islice(rows, limit)

    with open(results_path, "a", encoding="utf-8") as f:
        pbar = tqdm(rows, total=total, desc=out.name, unit="row")
        for row in pbar:
            if row["filename"] in done:
                continue
            result = run_sample(row, stt=stt, llm=llm, out=out, tts=tts)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            pbar.set_postfix_str(result["filename"], refresh=False)
        pbar.close()


def main() -> None:
    """Parse CLI flags and run every task under ``data/seahelm_audio_data/``.

    Returns:
        None: Each task writes into ``<out>/<task>/``; nothing is returned.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/pipeline", help="Output directory.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N rows per task (0 = all)."
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Also synthesize each reply to audio (off by default).",
    )
    args = parser.parse_args()

    # Folder for each task
    task_data_folders = [
        Path("data/seahelm_audio_data/ifeval"),
        Path("data/seahelm_audio_data/mmlu_lite"),
    ]

    for task_data_folder in task_data_folders:
        task_name = task_data_folder.name
        run(
            data=task_data_folder,
            out=Path(args.out) / task_name,
            limit=args.limit,
            include_tts=args.tts,
        )


if __name__ == "__main__":
    main()
