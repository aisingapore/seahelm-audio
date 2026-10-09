"""ElevenLabs text-to-speech. Writes the spoken reply as 16-bit mono WAV and returns its path."""

import hashlib
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from .tts_base import TTS

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # same default voice as the toolkit wrapper
SAMPLE_RATE = 24000  # matches the OmniVoice input audio sets


class ElevenLabsTTS(TTS):
    def __init__(
        self,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = "eleven_v4",
        sample_rate: int = SAMPLE_RATE,
        output_dir: str | Path = "outputs/tts",
    ):
        load_dotenv()
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        # pcm_<rate> returns raw 16-bit little-endian mono PCM with no header.
        self.output_format = f"pcm_{sample_rate}"
        self.output_dir = Path(output_dir)
        self.client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def generate(self, prompt: str, out_path: str | Path | None = None) -> str:
        """Synthesize `prompt` and return the path of the written WAV file.

        Without `out_path`, the file is named by a hash of the text so the same
        text always maps to the same file.
        """
        if out_path is None:
            digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
            out_path = self.output_dir / f"{digest}.wav"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pcm = b"".join(
            self.client.text_to_speech.convert(
                text=prompt,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format,
            )
        )
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        return str(out_path)
