"""ElevenLabs Scribe speech-to-text (batch REST endpoint)."""

import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from .stt_base import STT


class ElevenLabsScribeSTT(STT):
    def __init__(self, model_id: str = "scribe_v2", language_code: str | None = None):
        load_dotenv()
        self.model_id = model_id
        self.language_code = language_code
        self.client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def generate(self, prompt: str) -> str:
        """Transcribe one audio file. `prompt` is the path to the audio file."""
        with open(prompt, "rb") as audio:
            response = self.client.speech_to_text.convert(
                file=audio,
                model_id=self.model_id,
                language_code=self.language_code,
                tag_audio_events=False,
                diarize=False,
            )
        return response.text
