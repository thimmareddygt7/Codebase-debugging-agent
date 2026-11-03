import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Set
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


# Directory names that must never be indexed or used as a workspace source.
SENSITIVE_DIR_NAMES: Set[str] = {".git", ".ssh", ".aws", ".gnupg", ".kube"}


def allowed_roots() -> List[Path]:
    """Roots the API may read from: this project, TARGET_REPO_DIR and INDEX_ALLOWED_ROOTS."""
    roots = [Path(settings.PROJECT_ROOT).resolve()]
    extra = [settings.TARGET_REPO_DIR] + settings.INDEX_ALLOWED_ROOTS.split(",")
    for raw in extra:
        raw = (raw or "").strip()
        if raw:
            roots.append(Path(raw).expanduser().resolve())
    return roots


def resolve_directory_within_roots(
    target_path: Union[str, Path],
    roots: Optional[List[Path]] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[bool, Path, str]:
    """
    Resolve a user-supplied directory and require it to be an existing directory
    inside one of the allowed roots (after resolving symlinks and ``..``).
    Relative paths are anchored to ``base_dir`` (default: project root), never to the
    process working directory.
    """
    roots = [Path(r).resolve() for r in (roots or allowed_roots())]
    base = Path(base_dir or settings.PROJECT_ROOT).resolve()
    try:
        raw = Path(str(target_path)).expanduser()
        resolved = (raw if raw.is_absolute() else base / raw).resolve()
    except Exception as exc:
        return False, Path("."), f"Malformed path: {type(exc).__name__}"

    if not any(_is_relative_to(resolved, root) for root in roots):
        return False, resolved, "Path is outside the directories this server is allowed to read."
    if any(part.lower() in SENSITIVE_DIR_NAMES for part in resolved.parts):
        return False, resolved, "Path contains a protected directory."
    if not resolved.is_dir():
        return False, resolved, "Path is not an existing directory."
    return True, resolved, "OK"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
