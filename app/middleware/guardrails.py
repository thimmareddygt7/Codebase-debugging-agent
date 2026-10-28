import json
from typing import Dict, Any, Optional
from src.guardrails import (
    GuardrailManager,
    redact_secrets,
    contains_secrets,
    FileSystemGuardrail,
    CommandSafetyGuardrail,
    PatchValidatorGuardrail,
    PromptInjectionGuardrail,
    AuditLogger
)

# Shared global guardrail instance for API and CLI layers
guardrail_manager = GuardrailManager()


class APIGuardrailsMiddleware:
    """
    Middleware service ensuring all API endpoints enforce repository confinement,
    prompt-injection filtering, secret redaction, and human approval tracking.
    """

    def __init__(self, manager: Optional[GuardrailManager] = None):
        self.manager = manager or guardrail_manager

    def sanitize_user_input(self, user_prompt: str) -> Dict[str, Any]:
        """Validates incoming user query against prompt injection."""
        is_suspicious, rationale = self.manager.injection.detect_injection(user_prompt)
        return {
            "is_suspicious": is_suspicious,
            "warning": rationale if is_suspicious else None,
            "sanitized_prompt": user_prompt
        }

    def sanitize_response(self, response_text: str) -> str:
        """Ensures outgoing response contains zero leaked API keys or passwords."""
        return self.manager.redact(response_text)

    def review_patch(
        self,
        target_file: str,
        new_content: str,
        expected_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """Pre-validates patch before requesting human approval."""
        is_valid, msg, diff_text = self.manager.patch.validate_patch(
            target_path=target_file,
            new_content=new_content,
            expected_sha256=expected_sha256
        )
        return {
            "valid": is_valid,
            "message": msg,
            "diff": diff_text,
            "diff_lines": len(diff_text.splitlines())
        }

    def apply_approved_patch(
        self,
        target_file: str,
        new_content: str,
        rationale: str,
        user_confirmed: bool,
        run_tests: Optional[str] = None
    ) -> Dict[str, Any]:
        """Applies patch with explicit confirmation check."""
        if not user_confirmed:
            return {
                "success": False,
                "status": "APPROVAL_REQUIRED",
                "message": "Human approval is required before modifying source files. Change rejected."
            }

        # Override human approval callback for API execution with user_confirmed status
        approval_manager = GuardrailManager(approval_callback=lambda f, d: user_confirmed)
        return approval_manager.apply_patch_safely(
            target_file=target_file,
            new_content=new_content,
            rationale=rationale,
            run_tests_command=run_tests
        )
