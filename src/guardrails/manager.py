from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable
from src.guardrails.secrets import redact_secrets, contains_secrets
from src.guardrails.filesystem import FileSystemGuardrail
from src.guardrails.command_safety import CommandSafetyGuardrail
from src.guardrails.patch_validator import PatchValidatorGuardrail
from src.guardrails.human_approval import HumanApprovalGuardrail
from src.guardrails.prompt_injection import PromptInjectionGuardrail
from src.guardrails.audit import AuditLogger
from src.config import settings


class GuardrailManager:
    """
    Central orchestration engine for all 8 mandatory codebase agent guardrails:
    1. Human approval before code changes
    2. File-system access control
    3. Command execution safety
    4. Patch validation
    5. Post-fix test validation
    6. Prompt-injection protection
    7. Secrets & privacy protection
    8. Audit trail and observability
    """

    def __init__(self, approval_callback: Optional[Callable[[str, str], bool]] = None):
        self.fs = FileSystemGuardrail()
        self.cmd = CommandSafetyGuardrail()
        self.patch = PatchValidatorGuardrail()
        self.human = HumanApprovalGuardrail(approval_callback=approval_callback)
        self.injection = PromptInjectionGuardrail()
        self.audit = AuditLogger()

    # --- 1. Human Approval & Patch Application Workflow ---
    def apply_patch_safely(
        self,
        target_file: Path | str,
        new_content: str,
        rationale: str = "Automated code patch",
        expected_sha256: Optional[str] = None,
        run_tests_command: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete secure patch pipeline:
        1. Validate patch (size, encoding, staleness, path confinement)
        2. Prompt for human approval with diff
        3. Create backup before touching disk
        4. Apply patch if approved
        5. Run post-patch verification tests (if configured)
        6. Automatic rollback if tests fail
        7. Record audit log
        """
        # Step 1: Validate patch
        is_valid, validation_msg, diff_text = self.patch.validate_patch(
            target_path=target_file,
            new_content=new_content,
            expected_sha256=expected_sha256
        )

        if not is_valid:
            self.audit.log_event(
                event_type="PATCH_VALIDATION_FAILED",
                target_file=str(target_file),
                rationale=rationale,
                approval_status="REJECTED_BY_GUARDRAIL",
                extra_details={"reason": validation_msg}
            )
            return {
                "success": False,
                "status": "VALIDATION_FAILED",
                "message": validation_msg,
                "diff": diff_text
            }

        # Step 2: Request Human Approval (Critical Guardrail 1)
        is_approved = self.human.request_approval(
            target_file=target_file,
            diff_text=diff_text,
            rationale=rationale
        )

        if not is_approved:
            self.audit.log_event(
                event_type="HUMAN_APPROVAL_DENIED",
                target_file=str(target_file),
                rationale=rationale,
                approval_status="DENIED_BY_USER",
                extra_details={"diff_lines": len(diff_text.splitlines())}
            )
            return {
                "success": False,
                "status": "APPROVAL_DENIED",
                "message": f"Modification rejected: Human approval was not granted for '{target_file}'.",
                "diff": diff_text
            }

        # Step 3: Create backup for Rollback Support
        target_path = Path(target_file).resolve()
        backup_path = self.patch.create_backup(target_path)

        # Step 4: Apply patch
        try:
            target_path.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            if backup_path:
                self.patch.rollback(backup_path, target_path)
            return {
                "success": False,
                "status": "WRITE_FAILED",
                "message": f"Failed to write patch to disk ({exc}). Restored original from backup."
            }

        # Step 5: Run Permitted Post-Fix Tests (Guardrail 5)
        test_results = None
        if run_tests_command:
            print(f"\n>>> [GUARDRAIL 5: POST-FIX TEST VERIFICATION] Running '{run_tests_command}'...")
            test_success, retcode, stdout, stderr = self.cmd.execute_safe_command(run_tests_command)
            test_results = {
                "command": run_tests_command,
                "passed": test_success,
                "returncode": retcode,
                "output": stdout or stderr
            }

            # If tests fail, automatically rollback to protect user codebase!
            if not test_success:
                print(f"[TEST FAILURE DETECTED]: Tests failed after patch. Initiating safe rollback...")
                if backup_path:
                    self.patch.rollback(backup_path, target_path)
                    print(f"[ROLLBACK COMPLETE]: Original contents restored for '{target_path.name}'.")

                self.audit.log_event(
                    event_type="PATCH_ROLLED_BACK",
                    target_file=str(target_file),
                    rationale=rationale,
                    approval_status="APPROVED_BUT_TEST_FAILED",
                    tests_executed=run_tests_command,
                    test_outcome="FAILED",
                    extra_details={"test_output": stdout or stderr}
                )

                return {
                    "success": False,
                    "status": "TESTS_FAILED_ROLLED_BACK",
                    "message": f"Patch caused test failure (exit {retcode}). File safely rolled back.",
                    "test_results": test_results
                }

        # Success Audit
        self.audit.log_event(
            event_type="PATCH_APPLIED_SUCCESSFULLY",
            target_file=str(target_file),
            rationale=rationale,
            approval_status="APPROVED_AND_APPLIED",
            tests_executed=run_tests_command,
            test_outcome="PASSED" if run_tests_command else "NO_TESTS_SPECIFIED",
            extra_details={"backup_path": str(backup_path) if backup_path else None}
        )

        return {
            "success": True,
            "status": "APPLIED",
            "message": f"Patch successfully applied to '{target_path.name}'.",
            "backup_path": str(backup_path) if backup_path else None,
            "test_results": test_results
        }

    # --- 2. Safe Content & Secret Redaction ---
    def redact(self, text: str) -> str:
        """Redacts all secrets, API keys, and sensitive tokens."""
        return redact_secrets(text)

    def sanitize_untrusted_data(self, content: str, source_label: str = "codebase_source") -> str:
        """Isolates untrusted codebase snippets to prevent prompt injection."""
        return self.injection.sanitize_untrusted_content(content, source_label)

    # --- 3. Path Access Validation ---
    def verify_read_path(self, path: Path | str) -> Tuple[bool, str]:
        return self.fs.is_safe_read_path(path)

    def verify_write_path(self, path: Path | str) -> Tuple[bool, str]:
        return self.fs.is_safe_write_path(path)

    # --- 4. Command Safety ---
    def execute_command_safely(self, command: str, timeout: int = 45) -> Tuple[bool, int, str, str]:
        return self.cmd.execute_safe_command(command, timeout=timeout)
