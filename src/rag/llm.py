import os
import groq
from src.config import settings


class GroqLLM:
    """Ultra-fast LLM client powered by Groq LPUs."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        fallback_model: str = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.model = model or settings.GROQ_MODEL
        self.fallback_model = fallback_model or settings.FALLBACK_MODEL

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your .env file or environment variable."
            )

        self.client = groq.Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Generates response using Groq, with fallback to instant model on rate limits."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

    def chat(self, messages: list) -> str:
        """Sends chat completion request to Groq."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except groq.RateLimitError:
            if self.fallback_model and self.fallback_model != self.model:
                response = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            raise


class OllamaLLM:
    """Local LLM client using Ollama (kept for offline fallback)."""

    def __init__(self, model: str = None):
        self.model = model or settings.OLLAMA_MODEL

    def generate(self, prompt: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


def get_llm(provider: str = None):
    """Factory function returning the configured LLM client."""
    selected_provider = provider or settings.MODEL_PROVIDER

    if selected_provider.lower() == "groq":
        return GroqLLM()
    elif selected_provider.lower() == "ollama":
        return OllamaLLM()
    else:
        if settings.GROQ_API_KEY:
            return GroqLLM()
        return OllamaLLM()