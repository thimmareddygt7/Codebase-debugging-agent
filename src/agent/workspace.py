"""
Disposable debugging workspace.

The fix loop never edits the real repository. It copies the target repo into a
private temp directory, lets the agent read / edit / test there, and only hands
back a unified diff. Applying that diff to the real repo goes through the
existing human-approval workflow (/fix/propose -> approve -> apply).

Safety properties enforced here:
  * edits are confined to the workspace (relative, traversal-free, no .git/.env)
  * test files / test config cannot be edited (the agent must fix the SOURCE,
    not make the tests easier)
  * an edit that would leave a .py file with a syntax error is rejected
  * tests run through CommandSafetyGuardrail (no shell, scrubbed env, timeout),
    optionally inside a locked-down Docker container
"""
import ast
import difflib
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from src.config import settings
from src.guardrails.command_safety import CommandSafetyGuardrail, _path_is_safe

SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "vector_store",
    ".idea", ".vscode", "dist", "build", ".eggs",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".pfx", ".p12", ".log"}
TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini",
}
PROTECTED_NAMES = {"conftest.py", "tox.ini", "pytest.ini", "setup.cfg", "pyproject.toml"}

MAX_FILES = 5000
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_READ_LINES = 250
MAX_READ_CHARS = 24_000
MAX_EDIT_CHARS = 8_000


class WorkspaceError(Exception):
    """A workspace operation was rejected (message is safe to show to the model)."""


@dataclass
class TestRun:
    __test__ = False  # not a pytest test class

    command: str
    success: bool
    returncode: int
    output: str
    duration_s: float

    def tail(self, max_chars: int = 3500) -> str:
        text = self.output.strip()
        if len(text) <= max_chars:
            return text
        head = text[:800]
        return f"{head}\n...[{len(text) - max_chars - 800} chars omitted]...\n{text[-(max_chars - 800):]}"


def is_protected_path(rel: str) -> bool:
    """True for test files and test configuration the agent must not edit."""
    parts = PurePosixPath(rel).parts
    name = parts[-1] if parts else ""
    return (
        any(p in {"tests", "test", "testing"} for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name in PROTECTED_NAMES
    )


class DebugWorkspace:
    def __init__(self, source_dir: str, base_dir: Optional[str] = None,
                 max_files: int = MAX_FILES, max_bytes: int = MAX_TOTAL_BYTES):
        self.source = Path(source_dir).resolve()
        if not self.source.is_dir():
            raise WorkspaceError(f"Source directory does not exist: {source_dir}")
        self.root = Path(tempfile.mkdtemp(prefix="dbgws_", dir=base_dir)).resolve()
        self._originals: Dict[str, str] = {}
        self._crlf: Dict[str, bool] = {}
        self._guard = CommandSafetyGuardrail()
        try:
            self._copy_source(max_files, max_bytes)
        except Exception:
            shutil.rmtree(self.root, ignore_errors=True)
            raise

    # ----------------------------------------------------------------- lifecycle
    def _copy_source(self, max_files: int, max_bytes: int) -> None:
        files = total = 0
        for dirpath, dirnames, filenames in os.walk(self.source):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            rel_dir = Path(dirpath).relative_to(self.source)
            for fname in filenames:
                if Path(fname).suffix.lower() in SKIP_SUFFIXES:
                    continue
                if fname == ".env" or (fname.startswith(".env.") and fname != ".env.example"):
                    continue
                src = Path(dirpath) / fname
                if src.is_symlink():
                    continue
                size = src.stat().st_size
                files += 1
                total += size
                if files > max_files or total > max_bytes:
                    raise WorkspaceError(
                        f"Repository too large for a debugging workspace "
                        f"(limit {max_files} files / {max_bytes // (1024 * 1024)} MB)."
                    )
                dest = self.root / rel_dir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "DebugWorkspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # ----------------------------------------------------------------- helpers
    def _resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not _path_is_safe(rel.replace("\\", "/"), self.root):
            raise WorkspaceError(f"Path '{str(rel)[:80]}' is not allowed (use a relative path inside the project).")
        return self.root / rel.replace("\\", "/")

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise WorkspaceError("File is not valid UTF-8 text.")

    # ----------------------------------------------------------------- read-only tools
    def list_files(self, directory: str = ".", max_entries: int = 200, max_depth: int = 3) -> str:
        base = self.root if directory in (".", "", "./") else self._resolve(directory)
        if not base.is_dir():
            raise WorkspaceError(f"'{directory}' is not a directory.")
        entries: List[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
            depth = len(Path(dirpath).relative_to(base).parts)
            if depth >= max_depth:
                dirnames[:] = []
            for fname in sorted(filenames):
                entries.append(str((Path(dirpath) / fname).relative_to(self.root)).replace(os.sep, "/"))
                if len(entries) >= max_entries:
                    return "\n".join(entries) + f"\n... (truncated at {max_entries} entries)"
        return "\n".join(entries) if entries else "(empty directory)"

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"File not found: {path}")
        lines = self._read_text(target).replace("\r\n", "\n").split("\n")
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line)) if end_line else min(total, start + MAX_READ_LINES - 1)
        end = min(end, start + MAX_READ_LINES - 1)
        body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(start, end + 1))
        if len(body) > MAX_READ_CHARS:
            body = body[:MAX_READ_CHARS] + "\n... [truncated]"
        note = "" if end >= total else f"\n[more lines available: call read_file with start_line={end + 1}]"
        return f"FILE {path} (lines {start}-{end} of {total})\n{body}{note}"

    def grep(self, pattern: str, ignore_case: bool = True, max_results: int = 40) -> str:
        if not pattern or len(pattern) > 200:
            raise WorkspaceError("pattern must be 1-200 characters.")
        needle = pattern.lower() if ignore_case else pattern
        hits: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() not in TEXT_SUFFIXES or fpath.stat().st_size > 1_000_000:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                rel = str(fpath.relative_to(self.root)).replace(os.sep, "/")
                for lineno, line in enumerate(text.splitlines(), 1):
                    if needle in (line.lower() if ignore_case else line):
                        hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + f"\n... (stopped at {max_results} matches)"
        return "\n".join(hits) if hits else f"No matches for '{pattern}'."

    # ----------------------------------------------------------------- editing
    def replace_in_file(self, path: str, old_str: str, new_str: str) -> str:
        rel = path.replace("\\", "/")
        target = self._resolve(rel)
        if is_protected_path(rel):
            raise WorkspaceError(
                f"'{rel}' is a test or test-configuration file and cannot be edited. "
                "Fix the source code so the existing tests pass."
            )
        if not target.is_file():
            raise WorkspaceError(f"File not found: {rel}")
        if not isinstance(old_str, str) or not old_str.strip():
            raise WorkspaceError("old_str must be non-empty text copied exactly from the file.")
        if not isinstance(new_str, str):
            raise WorkspaceError("new_str must be a string.")
        if len(old_str) > MAX_EDIT_CHARS or len(new_str) > MAX_EDIT_CHARS:
            raise WorkspaceError(f"Edit too large (max {MAX_EDIT_CHARS} chars). Make a smaller, targeted change.")
        if old_str == new_str:
            raise WorkspaceError("old_str and new_str are identical; nothing to change.")

        raw = self._read_text(target)
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        old, new = old_str.replace("\r\n", "\n"), new_str.replace("\r\n", "\n")
        count = text.count(old)
        if count == 0:
            raise WorkspaceError(
                "old_str was not found. It must match the file exactly, including indentation; "
                "use read_file and copy the lines."
            )
        if count > 1:
            raise WorkspaceError(f"old_str appears {count} times; include more surrounding lines so it is unique.")

        updated = text.replace(old, new, 1)
        if target.suffix == ".py":
            try:
                ast.parse(updated)
            except SyntaxError as exc:
                raise WorkspaceError(
                    f"Edit rejected: it would leave {rel} with a syntax error "
                    f"(line {exc.lineno}: {exc.msg}). File was not changed."
                )
        self._originals.setdefault(rel, text)
        self._crlf.setdefault(rel, crlf)
        target.write_bytes((updated.replace("\n", "\r\n") if crlf else updated).encode("utf-8"))
        return f"Edited {rel}: replaced {old.count(chr(10)) + 1} line(s) with {new.count(chr(10)) + 1} line(s)."

    # ----------------------------------------------------------------- results
    def _current_normalized(self, rel: str) -> str:
        return self._read_text(self.root / rel).replace("\r\n", "\n")

    def changed_files(self) -> List[str]:
        return sorted(r for r, orig in self._originals.items() if self._current_normalized(r) != orig)

    def final_text(self, rel: str) -> str:
        """Current file content using the file's original newline style."""
        text = self._current_normalized(rel)
        return text.replace("\n", "\r\n") if self._crlf.get(rel) else text

    def diff(self) -> str:
        chunks = []
        for rel in self.changed_files():
            chunks.append("\n".join(difflib.unified_diff(
                self._originals[rel].split("\n"), self._current_normalized(rel).split("\n"),
                fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="",
            )))
        return "\n".join(chunks)

    # ----------------------------------------------------------------- tests
    def run_tests(self, command: str, timeout: Optional[int] = None) -> TestRun:
        timeout = timeout or settings.FIX_TEST_TIMEOUT
        docker_image = settings.SANDBOX_DOCKER_IMAGE if settings.SANDBOX_MODE == "docker" else None
        started = time.time()
        ok, code, out, err = self._guard.execute_safe_command(
            command, timeout=timeout, cwd=str(self.root), docker_image=docker_image
        )
        combined = (out + ("\n" + err if err.strip() else "")).strip()
        return TestRun(command=command, success=ok, returncode=code, output=combined,
                       duration_s=round(time.time() - started, 2))
