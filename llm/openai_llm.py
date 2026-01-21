# llm/openai_llm.py
from llm.base import BaseLLM
import openai

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model="gpt-4.1"):
        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
