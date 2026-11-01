import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ask_codebase_agent

def run_tests():
    print("=" * 60)
    print("TEST 1: Testing Documentation / Setup Branch")
    print("=" * 60)
    q1 = "What are the setup instructions or requirements mentioned in the project files?"
    res1 = ask_codebase_agent(q1)
    print(f"Classified query_type: {res1.get('query_type')}")
    print("Tool calls made:")
    for tc in res1.get("tool_calls", []):
        print(f" -> {tc['function']['name']}: {tc['function'].get('arguments')}")
    print("\nAnswer preview:", res1.get("final_answer")[:200], "...\n")

    print("=" * 60)
    print("TEST 2: Testing Code / Implementation Branch")
    print("=" * 60)
    q2 = "Where is CodebaseLoader defined and what class method loads files?"
    res2 = ask_codebase_agent(q2)
    print(f"Classified query_type: {res2.get('query_type')}")
    print("Tool calls made:")
    for tc in res2.get("tool_calls", []):
        print(f" -> {tc['function']['name']}: {tc['function'].get('arguments')}")
    print("\nAnswer preview:", res2.get("final_answer")[:200], "...\n")

if __name__ == "__main__":
    run_tests()