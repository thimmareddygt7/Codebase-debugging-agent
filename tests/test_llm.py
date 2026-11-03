import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.llm import get_llm, GroqLLM, OllamaLLM
from src.config import settings

print()
print("=" * 70)
print(f"LLM CLIENT TEST (Provider: {settings.MODEL_PROVIDER})")
print("=" * 70)

if settings.MODEL_PROVIDER == "groq" and not settings.GROQ_API_KEY:
    print("GROQ_API_KEY not configured. Skipping live network call (PASS).")
    print("Set GROQ_API_KEY in .env to test live generation.")
    sys.exit(0)

llm = get_llm()

prompt = "Explain what a Python class is in exactly two sentences."
answer = llm.generate(prompt)

print(answer)


assert answer
assert len(answer.strip()) > 0


print()
print("=" * 70)
print("STATUS: PASS")
print("=" * 70)