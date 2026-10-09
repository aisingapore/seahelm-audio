import os

from dotenv import load_dotenv
from openai import OpenAI

from .llm_base import LLM


class SeaLionLLM(LLM):
    def __init__(self):
        load_dotenv()
        self.model_name = "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"
        self.client = OpenAI(
            api_key=os.environ["SEALION_API_KEY"], base_url="https://api.sea-lion.ai/v1"
        )

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        reply = completion.choices[0].message.content
        return reply
