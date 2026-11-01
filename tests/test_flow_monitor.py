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
from src.observability import is_tracing_enabled, get_project_dashboard_url
from src.agent.graph import ask_codebase_agent


def main():
    print("=" * 80)
    print("END-TO-END PIPELINE FLOW & MONITORING TEST")
    print("=" * 80)
    print(f"Engine:     {settings.MODEL_PROVIDER} ({settings.GROQ_MODEL})")
    print(f"LangSmith:  {'ENABLED' if is_tracing_enabled() else 'DISABLED'}")
    print(f"Dashboard:  {get_project_dashboard_url()}")
    print("=" * 80)

    # A query specifically asking to find and explain a code symbol
    question = "Where is CodebaseChunker defined in src/ingestion/chunker.py and how does it chunk functions?"
    thread_id = "flow_monitoring_demo"

    result = ask_codebase_agent(question=question, thread_id=thread_id)

    print("\n" + "=" * 80)
    print("AGENT FINAL ANSWER:")
    print("=" * 80)
    print(result.get("final_answer", ""))
    print("=" * 80)


if __name__ == "__main__":
    main()
