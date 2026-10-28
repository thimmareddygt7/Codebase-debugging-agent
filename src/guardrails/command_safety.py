import subprocess
import shlex
from typing import Tuple, List, Optional
from src.config import settings

# Allowlist of safe, non-destructive read-only or testing commands
ALLOWED_COMMAND_PREFIXES: List[List[str]] = [
    ["pytest"],
    ["python", "-m", "pytest"],
    ["python", "-m", "unittest"],
    ["python", "testings/"],
    ["git", "status"],
    ["git", "diff"],
    ["git", "log", "-n"],
    ["git", "branch"],
]

# Strict blocklist of destructive commands that must never be executed automatically
FORBIDDEN_PATTERNS: List[str] = [
    "rm ", "del ", "rmdir", "shred", "format ", "dd if=",
    "sudo", "chmod", "chown", "mkfs",
    "git push --force", "git push -f", "git reset --hard", "git clean -fd",
    "powershell -enc", "curl ", "wget ", "nc ", "ncat",
    "DROP TABLE", "TRUNCATE", "DELETE FROM",
]

DEFAULT_TIMEOUT_SECONDS = 45
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB max stdout/stderr capture


class CommandSafetyGuardrail:
    """
    Enforces command execution safety with an allowlist for permitted testing commands,
    blocks destructive commands, and applies execution timeouts and output buffer limits.
    """

    def is_command_allowed(self, command_str: str) -> Tuple[bool, bool, str]:
        """
        Inspects command.
        Returns (is_permitted, requires_human_confirmation, rationale).
        """
        if not command_str or not command_str.strip():
            return False, False, "Empty command string."

        cmd_clean = command_str.strip()

        # Check blocklist first
        for forbidden in FORBIDDEN_PATTERNS:
            if forbidden.lower() in cmd_clean.lower():
                return (
                    False,
                    False,
                    f"Command Rejected: Contains prohibited destructive pattern '{forbidden}'."
                )

        try:
            tokens = shlex.split(cmd_clean, posix=False)
        except Exception:
            tokens = cmd_clean.split()

        if not tokens:
            return False, False, "Malformed command."

        # Check allowlist for automated test / read-only commands
        for prefix in ALLOWED_COMMAND_PREFIXES:
            if len(tokens) >= len(prefix):
                match = True
                for i in range(len(prefix)):
                    if prefix[i].endswith("/"):
                        if not tokens[i].replace("\\", "/").startswith(prefix[i]):
                            match = False
                            break
                    elif tokens[i].lower() != prefix[i].lower():
                        match = False
                        break
                if match:
                    return True, False, "Allowed safe command."

        # Package installations, migrations, or custom scripts require explicit human confirmation
        if any(tool in cmd_clean.lower() for tool in ("pip install", "npm install", "alembic", "manage.py migrate")):
            return True, True, "Dependency / Migration command requires explicit human confirmation."

        return False, True, f"Command '{cmd_clean[:50]}' is not in the approved safe allowlist and requires confirmation."

    def execute_safe_command(
        self,
        command_str: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        cwd: Optional[str] = None
    ) -> Tuple[bool, int, str, str]:
        """
        Executes an approved command within strict timeout and output resource limits.
        Returns (success, returncode, stdout, stderr).
        """
        is_allowed, requires_confirm, rationale = self.is_command_allowed(command_str)
        if not is_allowed:
            return False, -1, "", f"Execution Blocked by Guardrail: {rationale}"

        work_dir = cwd or str(settings.PROJECT_ROOT)

        try:
            process = subprocess.run(
                command_str,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                timeout=timeout,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            stdout = process.stdout or ""
            stderr = process.stderr or ""

            if len(stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
                stdout = stdout[:MAX_OUTPUT_BYTES] + "\n... [Output truncated by Guardrail: Max 50KB exceeded]"

            if len(stderr.encode("utf-8")) > MAX_OUTPUT_BYTES:
                stderr = stderr[:MAX_OUTPUT_BYTES] + "\n... [Stderr truncated by Guardrail: Max 50KB exceeded]"

            return (process.returncode == 0), process.returncode, stdout, stderr

        except subprocess.TimeoutExpired:
            return False, -1, "", f"Execution Blocked by Guardrail: Command timed out after {timeout} seconds."
        except Exception as exc:
            return False, -1, "", f"Execution Error: {exc}"
