from .base import LLM


class SeaLionLLM(LLM):
    def __init__(self, model: str = "aisingapore/Gemma-SEA-LION-v3-9B-IT"):
        self.model = model

    def generate(self, prompt: str) -> str: ...  # API call here
