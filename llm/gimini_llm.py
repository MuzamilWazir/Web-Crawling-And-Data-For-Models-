# llm/gemini_llm.py
import os
from .base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model="gemini-1.5-flash", api_key=None):
        """
        Initialize Gemini LLM

        Args:
            model: Model to use (gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp)
            api_key: API key (or set GOOGLE_API_KEY environment variable)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install Google AI: pip install google-generativeai")

        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        print(f"[INFO] Initialized Gemini with model: {model}")

    def generate(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        print(f"[INFO] Calling Gemini API ({self.model_name})...")

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 1000,
                }
            )

            answer = response.text
            print(f"[OK] Gemini responded")
            return answer

        except Exception as e:
            return f"⚠ Gemini API error: {type(e).__name__}: {str(e)}"