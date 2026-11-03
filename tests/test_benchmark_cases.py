"""
Integrity of the bug benchmark. If these pass, a high resolve rate cannot come from
broken cases: every case fails for a real reason before the fix and passes after it.
"""
import json
import re
from pathlib import Path

from benchmarks.run_benchmark import load_cases, run_noop, run_oracle, summarize
from src.agent.workspace import DebugWorkspace, is_protected_path

CASES = load_cases()


def test_there_are_enough_cases_across_difficulties():
    assert len(CASES) >= 15
    assert {c["difficulty"] for c in CASES} == {"easy", "medium", "hard"}
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert all(c["id"] == c["dir"].name for c in CASES)


def test_every_case_fails_at_baseline_for_a_real_assertion_not_an_import_error():
    for case in CASES:
        with DebugWorkspace(str(case["dir"] / "repo")) as ws:
            run = ws.run_tests(case["test_command"])
        assert not run.success, f"{case['id']}: baseline must FAIL"
        assert "FAILED" in run.output, f"{case['id']}: expected assertion failures: {run.output[-300:]}"
        for marker in ("ImportError", "ModuleNotFoundError", "SyntaxError", "NameError"):
            assert marker not in run.output, f"{case['id']}: baseline fails because of {marker}, not the bug"


def test_noop_resolves_nothing_and_oracle_resolves_everything():
    assert all(not run_noop(c)["resolved"] for c in CASES)
    oracle = [run_oracle(c) for c in CASES]
    assert all(r["resolved"] for r in oracle), [r["id"] for r in oracle if not r["resolved"]]
    summary = summarize(oracle)
    assert summary["resolved"] == summary["cases"] and summary["resolve_rate"] == 1.0


def test_reference_fixes_touch_source_files_only_never_tests():
    for case in CASES:
        assert case["fixed_files"], case["id"]
        for rel in case["fixed_files"]:
            assert not is_protected_path(rel), f"{case['id']}: fix edits protected file {rel}"
            assert (case["dir"] / "repo" / rel).is_file() and (case["dir"] / "fixed" / rel).is_file()


def test_bug_reports_describe_symptoms_without_leaking_the_fix():
    leaks = re.compile(r"(\bfix\b|deepcopy|off-by-one|\+ 1\b|>=|<=|mutable default|range\()", re.I)
    for case in CASES:
        assert 40 <= len(case["description"]) <= 260, case["id"]
        assert not leaks.search(case["description"]), f"{case['id']}: description hints at the fix"


def test_case_metadata_is_complete_and_commands_are_allowlisted():
    from src.guardrails.command_safety import CommandSafetyGuardrail
    guard = CommandSafetyGuardrail()
    for case in CASES:
        meta = json.loads((case["dir"] / "case.json").read_text(encoding="utf-8"))
        assert {"id", "category", "difficulty", "description", "test_command", "fixed_files"} <= set(meta)
        assert guard.is_command_allowed(meta["test_command"])[0], case["id"]


def test_summarize_computes_rates_cost_and_groupings():
    recs = [
        dict(id="a", category="c1", difficulty="easy", resolved=True, steps=4, tool_calls=6,
             prompt_tokens=1000, completion_tokens=200, duration_s=2.0),
        dict(id="b", category="c1", difficulty="hard", resolved=False, steps=8, tool_calls=10,
             prompt_tokens=3000, completion_tokens=400, duration_s=6.0),
    ]
    s = summarize(recs, price_in=1.0, price_out=2.0)
    assert s["resolve_rate"] == 0.5 and s["mean_steps"] == 6.0
    assert s["by_difficulty"] == {"easy": "1/1", "hard": "0/1"}
    assert s["total_cost_usd"] == round((4000 * 1.0 + 600 * 2.0) / 1e6, 4)
    assert s["cost_per_resolved_usd"] == s["total_cost_usd"]   # one resolved case
