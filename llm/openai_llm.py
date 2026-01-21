# llm/openai_llm.py
import os
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize OpenAI LLM

        Args:
            model: Model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            api_key: API key (or set OPENAI_API_KEY environment variable)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install OpenAI: pip install openai")

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

        self.client = OpenAI(api_key=self.api_key)
        print(f"[INFO] Initialized OpenAI with model: {model}")

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        print(f"[INFO] Calling OpenAI API ({self.model})...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            print(f"[OK] OpenAI responded (${response.usage.total_tokens} tokens)")
            return answer

        except Exception as e:
            return f"⚠ OpenAI API error: {type(e).__name__}: {str(e)}"