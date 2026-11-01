import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ask_codebase_agent

def main():
    session_id = "recruiter_eval_session_42"
    print("=" * 60)
    print(f"STARTING PERSISTENT MULTI-TURN SESSION: {session_id}")
    print("=" * 60)

    # Turn 1: Ask about CodebaseLoader
    q1 = "Where is CodebaseLoader defined and what directories does it ignore?"
    print(f"\n[Turn 1 Query]: {q1}\n" + "-" * 50)
    res1 = ask_codebase_agent(q1, thread_id=session_id)
    print("[Turn 1 Answer]:\n", res1.get("final_answer"))

    # Turn 2: Follow-up question relying on previous turn context
    q2 = "Why does it ignore those directories? Be brief."
    print(f"\n[Turn 2 Follow-Up]: {q2}\n" + "-" * 50)
    res2 = ask_codebase_agent(q2, thread_id=session_id)
    print("[Turn 2 Answer]:\n", res2.get("final_answer"))

    # Verify conversation message history growth in state
    total_messages = len(res2.get("messages", []))
    print("\n" + "=" * 60)
    print(f"VERIFICATION: Total messages tracked in persistent thread: {total_messages}")
    print("SQLite DB check: data/sessions.db exists ->", Path("data/sessions.db").exists())
    print("=" * 60)

if __name__ == "__main__":
    main()