# llm/ollama_llm.py
import subprocess
import sys
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model="qwen2.5:3b"):
        self.model = model
        print(f"[INFO] Initializing Ollama with model: {model}")

        # Test if Ollama is available
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                print("[WARNING] Ollama command failed. Make sure Ollama is running.")
            else:
                print("[OK] Ollama is available")
        except FileNotFoundError:
            print("[ERROR] Ollama not found. Please install from https://ollama.ai")
            sys.exit(1)
        except Exception as e:
            print(f"[WARNING] Could not verify Ollama: {e}")

    def generate(self, prompt: str, timeout=180) -> str:
        """
        Generate text from Ollama with proper error handling
        """
        print(f"[INFO] Generating response with {self.model}...")
        print("[INFO] This may take 30-180 seconds depending on prompt size...")

        try:
            # Run Ollama process
            process = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            # Send prompt and get response with timeout
            try:
                stdout, stderr = process.communicate(input=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return f"⚠ Ollama timed out after {timeout}s. Try:\n1. Using a smaller model (qwen2.5:1.5b)\n2. Reducing context size\n3. Increasing timeout"

            # Check for errors
            if process.returncode != 0:
                error_msg = stderr.strip() if stderr else "Unknown error"
                return f"⚠ Ollama error (code {process.returncode}):\n{error_msg}"

            # Return response
            response = stdout.strip()
            if not response:
                return "⚠ Ollama returned empty response. The model might not be loaded."

            return response

        except FileNotFoundError:
            return "⚠ Ollama command not found. Install from: https://ollama.ai"
        except Exception as e:
            return f"⚠ Unexpected error: {type(e).__name__}: {str(e)}"