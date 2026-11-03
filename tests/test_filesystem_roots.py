"""Directory confinement used by /index and /debug/fix."""
import os
import tempfile
from pathlib import Path

from src.guardrails.filesystem import resolve_directory_within_roots


def _mk():
    root = Path(tempfile.mkdtemp(prefix="roots_")).resolve()
    (root / "repo").mkdir()
    (root / "repo" / "pkg").mkdir()
    outside = Path(tempfile.mkdtemp(prefix="outside_")).resolve()
    return root, outside


def test_directory_inside_root_is_accepted_absolute_and_relative():
    root, _ = _mk()
    ok, resolved, _ = resolve_directory_within_roots(str(root / "repo"), roots=[root], base_dir=root)
    assert ok and resolved == root / "repo"
    ok, resolved, _ = resolve_directory_within_roots("repo/pkg", roots=[root], base_dir=root)
    assert ok and resolved == root / "repo" / "pkg"


def test_traversal_and_outside_paths_are_rejected():
    root, outside = _mk()
    for bad in ["../", "repo/../..", str(outside), "/", "/etc", "~"]:
        ok, _, reason = resolve_directory_within_roots(bad, roots=[root], base_dir=root)
        assert not ok, f"{bad!r} must be rejected ({reason})"


def test_missing_dir_and_files_are_rejected():
    root, _ = _mk()
    (root / "repo" / "f.py").write_text("x = 1")
    assert not resolve_directory_within_roots("nope", roots=[root], base_dir=root)[0]
    assert not resolve_directory_within_roots("repo/f.py", roots=[root], base_dir=root)[0]


def test_symlink_escape_is_rejected():
    root, outside = _mk()
    link = root / "repo" / "escape"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        return  # symlinks unavailable on this platform
    ok, _, _ = resolve_directory_within_roots("repo/escape", roots=[root], base_dir=root)
    assert not ok


def test_git_directory_is_protected():
    root, _ = _mk()
    (root / "repo" / ".git").mkdir()
    assert not resolve_directory_within_roots("repo/.git", roots=[root], base_dir=root)[0]
