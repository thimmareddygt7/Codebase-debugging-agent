import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.guardrails.secrets import redact_secrets
from src.config import settings


class AuditLogger:
    """
    Maintains a persistent, append-only, secret-redacted audit trail of all
    proposed modifications, approval decisions, commands, tests, and rollbacks.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (settings.PROJECT_ROOT / "data" / "audit_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        target_file: Optional[str] = None,
        rationale: Optional[str] = None,
        approval_status: Optional[str] = None,
        diff_summary: Optional[Dict[str, int]] = None,
        tests_executed: Optional[str] = None,
        test_outcome: Optional[str] = None,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Appends a structured event to the audit trail.
        All string parameters are passed through the secrets redactor.
        """
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": now,
            "event_type": event_type,
            "target_file": redact_secrets(str(target_file)) if target_file else None,
            "rationale": redact_secrets(str(rationale)) if rationale else None,
            "approval_status": approval_status,
            "diff_summary": diff_summary or {},
            "tests_executed": redact_secrets(str(tests_executed)) if tests_executed else None,
            "test_outcome": test_outcome,
            "details": {
                k: redact_secrets(str(v)) if isinstance(v, str) else v
                for k, v in (extra_details or {}).items()
            }
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[AUDIT LOGGING ERROR]: Failed to append to audit trail ({exc})")

        return record
