import os
import shutil
import hashlib
import difflib
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from src.guardrails.filesystem import FileSystemGuardrail
from src.config import settings

MAX_DIFF_LINES = 600
MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB limit for automated patches


class PatchValidatorGuardrail:
    """
    Validates patches before application, checks target path confinement,
    diff size, file encoding, and SHA-256 staleness, while generating automatic backups.
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        self.fs_guard = FileSystemGuardrail()
        self.backup_dir = backup_dir or (settings.PROJECT_ROOT / "data" / "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(content: str) -> str:
        """Computes SHA-256 hex digest for string content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def validate_patch(
        self,
        target_path: Path | str,
        new_content: str,
        expected_sha256: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Validates target path, file encoding, size, diff boundaries, and staleness.
        Returns (is_valid, validation_message, diff_text).
        """
        is_safe, resolved_path, msg = self.fs_guard.sanitize_and_resolve(target_path)
        if not is_safe:
            return False, f"Patch Rejected: {msg}", ""

        is_write_allowed, write_msg = self.fs_guard.is_safe_write_path(resolved_path)
        if not is_write_allowed:
            return False, f"Patch Rejected: {write_msg}", ""

        # Validate encoding
        try:
            new_content.encode("utf-8")
        except UnicodeEncodeError as exc:
            return False, f"Patch Rejected: New content contains non-UTF-8 characters ({exc}).", ""

        # Validate size
        if len(new_content.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
            return False, f"Patch Rejected: New content size ({len(new_content)} bytes) exceeds safety limit ({MAX_FILE_SIZE_BYTES} bytes).", ""

        original_content = ""
        if resolved_path.exists():
            try:
                original_content = resolved_path.read_text(encoding="utf-8")
            except Exception as exc:
                return False, f"Patch Rejected: Unable to read target file with UTF-8 encoding ({exc}).", ""

            # Concurrency / Staleness Check
            if expected_sha256:
                current_sha = self.compute_sha256(original_content)
                if current_sha != expected_sha256:
                    return (
                        False,
                        f"Patch Rejected: Target file '{resolved_path.name}' has been modified since patch generation. "
                        f"(Expected hash {expected_sha256[:8]}, found {current_sha[:8]}).",
                        ""
                    )

        # Generate Unified Diff
        diff_lines = list(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{resolved_path.name}",
                tofile=f"b/{resolved_path.name}",
            )
        )
        diff_text = "".join(diff_lines)

        if len(diff_lines) > MAX_DIFF_LINES:
            return (
                False,
                f"Patch Rejected: Diff size ({len(diff_lines)} lines) exceeds maximum allowed change limit ({MAX_DIFF_LINES} lines).",
                diff_text
            )

        if not diff_lines and resolved_path.exists():
            return False, "Patch Rejected: Proposed patch makes no modifications to the target file.", ""

        return True, "Patch validation passed.", diff_text

    def create_backup(self, target_path: Path | str) -> Optional[Path]:
        """Creates a timestamped backup copy of the target file in data/backups/."""
        resolved = Path(target_path).resolve()
        if not resolved.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{resolved.name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_filename

        shutil.copy2(resolved, backup_path)
        return backup_path

    def rollback(self, backup_path: Path | str, target_path: Path | str) -> bool:
        """Restores original file from backup in case of test failures or rejection."""
        try:
            b_path = Path(backup_path)
            t_path = Path(target_path)
            if b_path.exists():
                shutil.copy2(b_path, t_path)
                return True
            return False
        except Exception:
            return False
