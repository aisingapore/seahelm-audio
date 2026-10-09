from abc import ABC, abstractmethod


class STT(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
