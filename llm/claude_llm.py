# llm/claude_llm.py
import os
from .base import BaseLLM


class ClaudeLLM(BaseLLM):
    def __init__(self, model="claude-3-5-sonnet-20241022", api_key=None):
        """
        Initialize Claude LLM

        Args:
            model: Model to use (claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, etc.)
            api_key: API key (or set ANTHROPIC_API_KEY environment variable)
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install Anthropic: pip install anthropic")

        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")

        self.client = Anthropic(api_key=self.api_key)
        print(f"[INFO] Initialized Claude with model: {model}")

    def generate(self, prompt: str) -> str:
        """Generate response using Claude API"""
        print(f"[INFO] Calling Claude API ({self.model})...")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.content[0].text
            print(f"[OK] Claude responded ({response.usage.input_tokens + response.usage.output_tokens} tokens)")
            return answer

        except Exception as e:
            return f"⚠ Claude API error: {type(e).__name__}: {str(e)}"