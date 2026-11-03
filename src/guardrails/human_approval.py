import sys
from typing import Callable, Optional
from pathlib import Path


class HumanApprovalGuardrail:
    """
    Enforces mandatory human confirmation before source code files can be modified.
    Displays unified diffs and halts execution unless explicit approval is granted.
    """

    def __init__(self, approval_callback: Optional[Callable[[str, str], bool]] = None):
        """
        :param approval_callback: Optional custom callback (target_file, diff_text) -> bool.
                                  If None, falls back to interactive CLI prompt.
        """
        self.approval_callback = approval_callback

    def request_approval(
        self,
        target_file: Path | str,
        diff_text: str,
        rationale: str = "Automated bugfix proposed by Codebase Debugging Agent"
    ) -> bool:
        """
        Presents unified diff and prompts for explicit human approval.
        Returns True only if user explicitly approves.
        """
        file_str = str(target_file)

        # If a programmatic callback is registered (e.g. for Web API or automated tests)
        if self.approval_callback:
            return bool(self.approval_callback(file_str, diff_text))

        # Interactive Terminal Prompt
        print("\n" + "=" * 80)
        print(">>> [CRITICAL GUARDRAIL: HUMAN APPROVAL REQUIRED]")
        print("=" * 80)
        print(f"Target Source File:  {file_str}")
        print(f"Proposed Rationale:  {rationale}")
        print("-" * 80)
        print("PROPOSED UNIFIED DIFF:")
        print("-" * 80)
        if diff_text.strip():
            print(diff_text)
        else:
            print("[No diff lines generated — new file creation or overwrite]")
        print("=" * 80)

        # Check if stdin is interactive
        if not sys.stdin.isatty():
            print("[GUARDRAIL DENIAL]: Non-interactive environment detected without approval callback.")
            print("Modification rejected by default. (No approval = No modification).")
            return False

        try:
            choice = input(f"Approve and apply this code change to '{file_str}'? [y/N]: ").strip().lower()
            approved = choice in ("y", "yes")
            if approved:
                print(f"[GUARDRAIL APPROVED]: Explicit approval granted for '{file_str}'.")
            else:
                print(f"[GUARDRAIL DENIED]: Modification rejected by user for '{file_str}'.")
            return approved
        except (KeyboardInterrupt, EOFError):
            print("\n[GUARDRAIL ABORTED]: Approval prompt interrupted. Change rejected.")
            return False
