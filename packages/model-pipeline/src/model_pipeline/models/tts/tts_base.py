from abc import ABC, abstractmethod


class TTS(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
