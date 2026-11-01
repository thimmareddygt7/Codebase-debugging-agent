import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import settings
from src.observability import is_tracing_enabled, get_langsmith_client, get_project_dashboard_url
from src.agent.graph import ask_codebase_agent


def main():
    print("=" * 80)
    print("LANGSMITH OBSERVABILITY & TRACING TEST")
    print("=" * 80)

    print(f"\nTracing Active: {is_tracing_enabled()}")
    print(f"Project Name:   {settings.LANGSMITH_PROJECT}")
    print(f"Model Engine:   {settings.MODEL_PROVIDER} ({settings.GROQ_MODEL})")
    print(f"Dashboard URL:  {get_project_dashboard_url()}")

    client = get_langsmith_client()
    if not client:
        print("\n[ERROR]: LangSmith client could not be initialized.")
        print("Please check LANGSMITH_API_KEY in your .env file.")
        sys.exit(1)

    print("\n" + "-" * 80)
    print("Executing traced query through LangGraph state machine...")
    print("-" * 80)

    question = "How does the AST chunker parse Python functions and classes?"
    thread_id = "langsmith_live_demo"

    result = ask_codebase_agent(question=question, thread_id=thread_id)

    print("\n[Agent Final Answer Summary]:")
    answer = result.get("final_answer", "")
    print(answer[:300] + ("..." if len(answer) > 300 else ""))

    print("\n" + "=" * 80)
    print("LANGSMITH TRACE SUCCESSFULLY LOGGED!")
    print("=" * 80)
    print(f"View this trace live in your LangSmith dashboard:")
    print(f">> {get_project_dashboard_url()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
