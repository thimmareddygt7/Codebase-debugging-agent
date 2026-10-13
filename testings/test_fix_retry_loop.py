import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.validator import FixValidator
from src.agent.graph import ask_codebase_agent

def test_validator_unit():
    print("=" * 60)
    print("TEST 1: Validator Unit Test")
    print("=" * 60)
    validator = FixValidator()

    bad_snippet = "def broken_code(\n    return 42"  # Missing closing paren & syntax error
    valid, msg = validator.validate_code_syntax(bad_snippet)
    print(f"Bad snippet caught correctly: {not valid} -> '{msg}'")
    assert not valid, "Validator should detect syntax error"

    good_snippet = "def fixed_code():\n    return 42"
    valid, msg = validator.validate_code_syntax(good_snippet)
    print(f"Good snippet accepted: {valid} -> '{msg}'")
    assert valid, "Validator should accept valid code"


def test_agent_with_fix_request():
    print("\n" + "=" * 60)
    print("TEST 2: End-to-End Fix Generation & Validation")
    print("=" * 60)
    query = (
        "Write a python helper function 'sanitize_path(path: str) -> str' that removes "
        "trailing slashes and handles empty strings. Provide the implementation in a python block."
    )
    result = ask_codebase_agent(query, max_retries=2)
    print(f"Fix Valid: {result.get('fix_valid')}")
    print(f"Retry Count: {result.get('retry_count')}")
    print(f"Validation Feedback: {result.get('validation_feedback')}")
    print("\nFinal Proposed Fix:\n")
    print(result.get("final_answer"))


if __name__ == "__main__":
    test_validator_unit()
    test_agent_with_fix_request()