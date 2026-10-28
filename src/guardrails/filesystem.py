import os
from pathlib import Path
from typing import Tuple, Union, Set
from src.config import settings

# Sensitive files and directories that must never be read or written by the agent
PROTECTED_WRITE_TARGETS: Set[str] = {
    ".env",
    ".env.example",
    ".git",
    ".gitignore",
    "id_rsa",
    "id_rsa.pub",
    "credentials",
    "service_account.json",
}

PROTECTED_EXTENSIONS: Set[str] = {
    ".pem",
    ".key",
    ".pfx",
    ".pkcs12",
}


class FileSystemGuardrail:
    """
    Enforces repository root confinement, prevents path traversal attacks,
    and protects sensitive configuration and git internal files from tampering.
    """

    def __init__(self, root_dir: Union[str, Path] = None):
        self.root_dir = Path(root_dir or settings.PROJECT_ROOT).resolve()

    def sanitize_and_resolve(self, target_path: Union[str, Path]) -> Tuple[bool, Path, str]:
        """
        Resolves path and verifies it strictly resides within self.root_dir.
        Detects and rejects directory traversal ('../') and unauthorized absolute paths.
        """
        try:
            raw_path = Path(target_path)
            # If relative, anchor to root_dir
            if not raw_path.is_absolute():
                resolved = (self.root_dir / raw_path).resolve()
            else:
                resolved = raw_path.resolve()

            # Confinement check: resolved path must start with root_dir
            try:
                resolved.relative_to(self.root_dir)
            except ValueError:
                return (
                    False,
                    resolved,
                    f"Path Traversal Rejected: Path '{target_path}' resolves outside repository root '{self.root_dir}'."
                )

            return True, resolved, "Path within repository root."
        except Exception as exc:
            return False, Path(target_path), f"Malformed path '{target_path}': {exc}"

    def is_safe_read_path(self, target_path: Union[str, Path]) -> Tuple[bool, str]:
        """Verifies if the file path is safe to read."""
        is_confined, resolved, msg = self.sanitize_and_resolve(target_path)
        if not is_confined:
            return False, msg

        # Disallow reading .git internal objects or direct credentials
        parts_lower = [p.lower() for p in resolved.parts]
        if ".git" in parts_lower:
            return False, "Access Denied: Reading Git internal directory (.git) is prohibited."

        if resolved.name.lower() in ("id_rsa", "credentials", "service_account.json"):
            return False, f"Access Denied: Reading sensitive credential file '{resolved.name}' is prohibited."

        if resolved.suffix.lower() in PROTECTED_EXTENSIONS:
            return False, f"Access Denied: Reading protected key extension '{resolved.suffix}' is prohibited."

        return True, "Safe read path."

    def is_safe_write_path(self, target_path: Union[str, Path]) -> Tuple[bool, str]:
        """Verifies if the file path is safe to write, modify, or delete."""
        is_confined, resolved, msg = self.sanitize_and_resolve(target_path)
        if not is_confined:
            return False, msg

        parts_lower = [p.lower() for p in resolved.parts]
        if ".git" in parts_lower:
            return False, "Access Denied: Modifying Git internals (.git) is strictly prohibited."

        # Protect .env files and credentials
        if resolved.name.startswith(".env") or resolved.name.lower() in PROTECTED_WRITE_TARGETS:
            return False, f"Access Denied: Modifying protected configuration file '{resolved.name}' is strictly prohibited."

        if resolved.suffix.lower() in PROTECTED_EXTENSIONS:
            return False, f"Access Denied: Modifying private key file '{resolved.name}' is strictly prohibited."

        return True, "Safe write path."
