import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ask_codebase_agent

def main():
    # Use a fresh session ID
    session_id = "v6b_debug_context_verification"
    print("=" * 60)
    print(f"TESTING DEBUGGING CONTEXT MEMORY: {session_id}")
    print("=" * 60)

    # Turn 1: Investigate a specific component
    q1 = "Where is CodebaseChunker implemented and what is its main responsibility?"
    print(f"\n[Turn 1 Query]: {q1}\n" + "-" * 50)
    res1 = ask_codebase_agent(q1, thread_id=session_id)
    print("[Turn 1 Answer Preview]:", res1.get("final_answer")[:250], "...\n")
    print(f"Turn 1 Investigated Files: {res1.get('investigated_files')}")
    print(f"Turn 1 Active Symbols: {res1.get('active_symbols')}")

    # Turn 2: Follow-up that tests contextual recall of the same component
    q2 = "What AST visitor method in that class handles python functions?"
    print(f"\n[Turn 2 Follow-Up]: {q2}\n" + "-" * 50)
    res2 = ask_codebase_agent(q2, thread_id=session_id)
    print("[Turn 2 Answer]:\n", res2.get("final_answer"))
    print(f"\nTurn 2 Investigated Files: {res2.get('investigated_files')}")
    print(f"Turn 2 Active Symbols: {res2.get('active_symbols')}")
    print(f"Turn 2 Debug Notes: {res2.get('debug_notes')}")

    # Check assertions for recruiter-level verification
    assert len(res2.get("investigated_files", [])) > 0, "Agent should remember investigated files across turns"
    assert any("chunker" in f.lower() for f in res2.get("investigated_files", [])), "Chunker file should be in memory"
    print("\n" + "=" * 60)
    print("ALL ASSERTIONS PASSED: Agent successfully maintained debugging context across turns!")
    print("=" * 60)

if __name__ == "__main__":
    main() 