import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ask_codebase_agent
from src.agent.context_verifier import ContextCrossChecker

def run_tests():
    print("=" * 60)
    print("TEST 1: Unit Test - ContextCrossChecker Relevance Scoring")
    print("=" * 60)
    verifier = ContextCrossChecker()

    sample_results = [{
        "file_path": "src/ingestion/loader.py",
        "chunk_name": "CodebaseLoader",
        "code": "class CodebaseLoader:\n    def get_source_files(self): pass"
    }]

    # Relevant query check
    ok, score, note = verifier.verify_relevance("Where is CodebaseLoader defined?", sample_results)
    print(f"Relevant Query Verified: {ok} (Score: {score}) -> {note}")
    assert ok is True and score > 0.3, "Context should be verified as relevant"

    # Irrelevant query check
    ok_bad, score_bad, note_bad = verifier.verify_relevance("How does user payment checkout process credit cards?", sample_results)
    print(f"Irrelevant Query Caught: {not ok_bad} (Score: {score_bad}) -> {note_bad}")
    assert ok_bad is False, "Context should be flagged as insufficient"

    print("\n" + "=" * 60)
    print("TEST 2: End-to-End State Machine Cross-Checking")
    print("=" * 60)
    query = "Where is CodebaseChunker defined and how does it chunk classes?"
    res = ask_codebase_agent(query, thread_id="test_cross_check_session")

    print(f"Context Verified: {res.get('context_verified')}")
    print(f"Confidence Score: {res.get('context_confidence')}")
    print(f"Verification Notes: {res.get('verification_notes')}")
    print("\nFinal Answer Preview:\n", res.get("final_answer")[:250], "...")

    assert res.get("context_verified") is True, "Expected valid context verification on codebase question"
    print("\nALL CROSS-CHECKING ASSERTIONS PASSED!")

if __name__ == "__main__":
    run_tests()