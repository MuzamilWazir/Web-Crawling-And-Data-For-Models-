# llm/base.py
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for all LLM implementations"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM"""
        pass