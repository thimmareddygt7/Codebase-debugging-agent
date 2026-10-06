import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.llm import OllamaLLM


llm = OllamaLLM()


prompt = """
Explain what a Python class is in exactly two sentences.
"""


print()
print("=" * 70)
print("OLLAMA LLM TEST")
print("=" * 70)

print()
print("Prompt:")
print(prompt)


answer = llm.generate(prompt)


print()
print("=" * 70)
print("GENERATED ANSWER")
print("=" * 70)

print(answer)


assert answer
assert len(answer.strip()) > 0


print()
print("=" * 70)
print("STATUS: PASS")
print("=" * 70)