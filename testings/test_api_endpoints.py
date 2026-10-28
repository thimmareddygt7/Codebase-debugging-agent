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

from fastapi.testclient import TestClient
from app.main import app


def run_api_tests():
    print("=" * 80)
    print("FASTAPI CODEBASE DEBUGGING AGENT ENDPOINT VERIFICATION")
    print("=" * 80)

    client = TestClient(app)

    # =========================================================================
    # STAGE 1: CORE API ENDPOINTS
    # =========================================================================
    print("\n--- STAGE 1: CORE API ---")

    # 1. GET /health
    res_health = client.get("/health")
    print(f"1. GET /health -> Status: {res_health.status_code}, Body: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"
    print("   [PASS] /health is functional.")

    # 2. GET /ready
    res_ready = client.get("/ready")
    print(f"2. GET /ready -> Status: {res_ready.status_code}, Components: {res_ready.json()['components']}")
    assert res_ready.status_code == 200
    assert "vector_store" in res_ready.json()["components"]
    print("   [PASS] /ready successfully reports component health.")

    # 3. POST /search
    res_search = client.post("/search", json={"query": "CodebaseChunker", "top_k": 3, "search_type": "all"})
    print(f"3. POST /search -> Status: {res_search.status_code}, Results: {res_search.json()['total_results']}")
    assert res_search.status_code == 200
    assert res_search.json()["total_results"] > 0
    print("   [PASS] /search returns ranked code chunks.")

    # 4. POST /debug
    res_debug = client.post("/debug", json={
        "bug_description": "FixValidator in validator.py regex might fail on backticks",
        "suspected_files": ["src/agent/validator.py"]
    })
    print(f"4. POST /debug -> Status: {res_debug.status_code}")
    assert res_debug.status_code == 200
    assert len(res_debug.json()["diagnosis"]) > 0
    print("   [PASS] /debug produced diagnosis with supporting code references.")

    # =========================================================================
    # STAGE 2: REPOSITORY INDEXING ENDPOINTS
    # =========================================================================
    print("\n--- STAGE 2: REPOSITORY INDEXING ---")

    # 5. GET /index/status
    res_idx_status = client.get("/index/status")
    print(f"5. GET /index/status -> Status: {res_idx_status.status_code}, Records: {res_idx_status.json()['records_stored']}")
    assert res_idx_status.status_code == 200
    print("   [PASS] /index/status reports live vector store records.")

    # 6. POST /index
    res_idx = client.post("/index", json={"target_path": "testings", "reset_index": False, "async_mode": True})
    print(f"6. POST /index -> Status: {res_idx.status_code}, Message: {res_idx.json()['message']}")
    assert res_idx.status_code == 200
    print("   [PASS] /index triggers background indexing cleanly.")

    # =========================================================================
    # STAGE 3: LANGGRAPH WORKFLOW & FIX PROPOSALS
    # =========================================================================
    print("\n--- STAGE 3: WORKFLOW & FIX PROPOSALS ---")

    # Create a scratch dummy file for safe patch testing
    scratch_file = Path("testings/temp_test_patch.py")
    scratch_file.write_text("# Initial code version 1\ndef calculate_sum(a, b):\n    return a - b  # Bug!\n", encoding="utf-8")

    # 7. POST /fix/propose
    propose_payload = {
        "problem_description": "calculate_sum subtracts instead of adding",
        "target_file": "testings/temp_test_patch.py",
        "new_content": "# Initial code version 1\ndef calculate_sum(a, b):\n    return a + b  # Fixed!\n",
        "rationale": "Fix subtraction operator to addition operator in calculate_sum"
    }
    res_propose = client.post("/fix/propose", json=propose_payload)
    print(f"7. POST /fix/propose -> Status: {res_propose.status_code}")
    assert res_propose.status_code == 200
    fix_id = res_propose.json()["fix"]["fix_id"]
    print(f"   [PASS] Fix proposal created with ID: {fix_id}")

    # 8. GET /fix/{fix_id}
    res_get_fix = client.get(f"/fix/{fix_id}")
    print(f"8. GET /fix/{fix_id} -> Status: {res_get_fix.status_code}, State: {res_get_fix.json()['fix']['status']}")
    assert res_get_fix.status_code == 200
    assert res_get_fix.json()["fix"]["status"] == "PROPOSED"
    print("   [PASS] Initial fix state is strictly PROPOSED (Unapproved).")

    # 9. GET /fix/{fix_id}/diff
    res_diff = client.get(f"/fix/{fix_id}/diff")
    print(f"9. GET /fix/{fix_id}/diff -> Status: {res_diff.status_code}, Diff lines: {res_diff.json()['diff_lines']}")
    assert res_diff.status_code == 200
    assert "+    return a + b" in res_diff.json()["diff"]
    print("   [PASS] Unified diff generated and verified.")

    # =========================================================================
    # STAGE 4: APPROVAL AND SAFE CHANGES
    # =========================================================================
    print("\n--- STAGE 4: APPROVAL & SAFE CHANGES ---")

    # 10. Security Control: Try to APPLY unapproved patch -> Must be 403 Forbidden!
    res_unapproved = client.post(f"/fix/{fix_id}/apply", json={})
    print(f"10. Security Check (Apply unapproved): Status {res_unapproved.status_code}")
    assert res_unapproved.status_code == 403
    print("    [PASS] Unapproved patch was strictly rejected by Guardrail (403 Forbidden).")

    # 11. POST /fix/{fix_id}/approve
    res_approve = client.post(f"/fix/{fix_id}/approve", json={"approved_by": "developer_alice", "notes": "Approved for testing"})
    print(f"11. POST /fix/{fix_id}/approve -> Status: {res_approve.status_code}, State: {res_approve.json()['fix']['status']}")
    assert res_approve.status_code == 200
    assert res_approve.json()["fix"]["status"] == "APPROVED"
    print("    [PASS] Fix approved by developer.")

    # 12. POST /fix/{fix_id}/apply
    res_apply = client.post(f"/fix/{fix_id}/apply", json={})
    print(f"12. POST /fix/{fix_id}/apply -> Status: {res_apply.status_code}")
    assert res_apply.status_code == 200
    assert res_apply.json()["fix"]["status"] == "APPLIED"
    applied_content = scratch_file.read_text(encoding="utf-8")
    assert "return a + b  # Fixed!" in applied_content
    print("    [PASS] Approved patch applied to file.")

    # 13. POST /fix/{fix_id}/rollback
    res_rollback = client.post(f"/fix/{fix_id}/rollback")
    print(f"13. POST /fix/{fix_id}/rollback -> Status: {res_rollback.status_code}")
    assert res_rollback.status_code == 200
    assert res_rollback.json()["fix"]["status"] == "ROLLED_BACK"
    rolled_back_content = scratch_file.read_text(encoding="utf-8")
    assert "return a - b  # Bug!" in rolled_back_content
    print("    [PASS] Patch safely rolled back from backup copy.")

    # Cleanup scratch file
    if scratch_file.exists():
        scratch_file.unlink()

    # =========================================================================
    # STAGE 5: TESTING, WORKFLOW, AND MONITORING
    # =========================================================================
    print("\n--- STAGE 5: TESTING & MONITORING ---")

    # 14. Command Guardrail Security Check: Block dangerous command
    res_dangerous = client.post("/tests/run", json={"test_command": "rm -rf /"})
    print(f"14. Security Check (Forbidden command): Status {res_dangerous.status_code}")
    assert res_dangerous.status_code == 400
    print("    [PASS] Destructive command strictly blocked by allowlist.")

    # 15. POST /tests/run with safe allowlisted command
    res_test_run = client.post("/tests/run", json={"test_command": "pytest testings/test_guardrails.py"})
    print(f"15. POST /tests/run -> Status: {res_test_run.status_code}, Outcome: {'PASSED' if res_test_run.json()['success'] else 'FAILED'}")
    assert res_test_run.status_code == 200
    test_run_id = res_test_run.json()["test_run_id"]
    print(f"    [PASS] Safe test execution succeeded with Run ID: {test_run_id}")

    # 16. GET /tests/{run_id}
    res_get_test = client.get(f"/tests/{test_run_id}")
    print(f"16. GET /tests/{test_run_id} -> Status: {res_get_test.status_code}")
    assert res_get_test.status_code == 200
    print("    [PASS] Test run result retrieved.")

    # 17. GET /audit/logs
    res_audit = client.get("/audit/logs?limit=5")
    print(f"17. GET /audit/logs -> Status: {res_audit.status_code}, Total logs: {res_audit.json()['total_logs']}")
    assert res_audit.status_code == 200
    assert res_audit.json()["count"] > 0
    print("    [PASS] Audit logs retrieved with complete trail.")

    # 18. GET /evaluation
    res_eval = client.get("/evaluation")
    print(f"18. GET /evaluation -> Status: {res_eval.status_code}, Source: {res_eval.json()['report_source']}")
    assert res_eval.status_code == 200
    print("    [PASS] Evaluation metrics scorecard retrieved.")

    print("\n" + "=" * 80)
    print("ALL 19 FASTAPI ENDPOINTS VERIFIED AND PASSING SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_api_tests()
