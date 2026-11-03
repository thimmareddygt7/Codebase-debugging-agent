"""DebugWorkspace: confinement, protected tests, safe editing, diffing, test running."""
import tempfile
from pathlib import Path

from src.agent.workspace import DebugWorkspace, WorkspaceError, is_protected_path


def _repo(files: dict) -> str:
    root = Path(tempfile.mkdtemp(prefix="wsrepo_"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return str(root)


BASE = {
    "pkg/__init__.py": "",
    "pkg/mod.py": "def add(a, b):\n    return a - b\n",
    "tests/test_mod.py": ("import unittest\nfrom pkg.mod import add\n\n"
                          "class T(unittest.TestCase):\n    def test_add(self):\n"
                          "        self.assertEqual(add(1, 2), 3)\n"),
}


def _raises(fn, *args):
    try:
        fn(*args)
    except WorkspaceError as exc:
        return str(exc)
    raise AssertionError(f"expected WorkspaceError from {fn.__name__}{args}")


def test_copy_skips_secrets_vcs_and_caches():
    src = _repo({**BASE, ".env": "GROQ_API_KEY=secret", ".env.example": "X=1", ".git/config": "x",
                 "data/app.db": "bin", "__pycache__/m.pyc": "x", "node_modules/a/b.js": "x"})
    with DebugWorkspace(src) as ws:
        files = ws.list_files().splitlines()
    assert "pkg/mod.py" in files and ".env.example" in files
    assert not any(f.startswith((".git", "node_modules", "__pycache__")) or f in (".env", "data/app.db") for f in files)


def test_real_repository_is_never_modified():
    src = _repo(BASE)
    before = (Path(src) / "pkg/mod.py").read_text()
    with DebugWorkspace(src) as ws:
        ws.replace_in_file("pkg/mod.py", "a - b", "a + b")
    assert (Path(src) / "pkg/mod.py").read_text() == before


def test_path_traversal_and_sensitive_paths_are_rejected():
    with DebugWorkspace(_repo(BASE)) as ws:
        for bad in ["../x.py", "/etc/passwd", "pkg/../../x", ".git/config", ".env", "C:/Windows/win.ini"]:
            _raises(ws.read_file, bad)
            _raises(ws.replace_in_file, bad, "a", "b")


def test_test_files_and_test_config_cannot_be_edited():
    assert all(is_protected_path(p) for p in
               ["tests/test_mod.py", "test_x.py", "pkg/mod_test.py", "conftest.py", "pytest.ini", "a/tests/helpers.py"])
    assert not is_protected_path("pkg/mod.py")
    with DebugWorkspace(_repo(BASE)) as ws:
        msg = _raises(ws.replace_in_file, "tests/test_mod.py", "3", "-1")
        assert "cannot be edited" in msg
        assert ws.changed_files() == []


def test_edit_requires_exact_unique_match():
    src = _repo({**BASE, "pkg/dup.py": "x = 1\ny = 1\n"})
    with DebugWorkspace(src) as ws:
        assert "not found" in _raises(ws.replace_in_file, "pkg/mod.py", "a * b", "a + b")
        assert "2 times" in _raises(ws.replace_in_file, "pkg/dup.py", "= 1", "= 2")
        assert "identical" in _raises(ws.replace_in_file, "pkg/mod.py", "a - b", "a - b")
        assert "non-empty" in _raises(ws.replace_in_file, "pkg/mod.py", "  ", "a")


def test_edit_that_breaks_syntax_is_rejected_and_file_unchanged():
    with DebugWorkspace(_repo(BASE)) as ws:
        msg = _raises(ws.replace_in_file, "pkg/mod.py", "return a - b", "return a +")
        assert "syntax error" in msg
        assert ws.changed_files() == [] and "a - b" in ws.read_file("pkg/mod.py")


def test_successful_edit_produces_diff_and_changed_files():
    with DebugWorkspace(_repo(BASE)) as ws:
        ws.replace_in_file("pkg/mod.py", "return a - b", "return a + b")
        assert ws.changed_files() == ["pkg/mod.py"]
        diff = ws.diff()
        assert "--- a/pkg/mod.py" in diff and "+++ b/pkg/mod.py" in diff
        assert "-    return a - b" in diff and "+    return a + b" in diff


def test_crlf_files_keep_their_line_endings():
    src = _repo({**BASE, "pkg/mod.py": "def add(a, b):\r\n    return a - b\r\n"})
    with DebugWorkspace(src) as ws:
        ws.replace_in_file("pkg/mod.py", "return a - b", "return a + b")
        final = ws.final_text("pkg/mod.py")
        assert final == "def add(a, b):\r\n    return a + b\r\n"
        assert "\r" not in ws.diff()


def test_read_file_numbers_lines_and_pages():
    big = "\n".join(f"line{i}" for i in range(1, 601))
    with DebugWorkspace(_repo({**BASE, "pkg/big.py": big})) as ws:
        first = ws.read_file("pkg/big.py")
        assert "lines 1-250 of 600" in first and "    1| line1" in first
        assert "start_line=251" in first
        assert "  300| line300" in ws.read_file("pkg/big.py", 300, 302)
        _raises(ws.read_file, "pkg/missing.py")


def test_grep_finds_matches_with_locations():
    with DebugWorkspace(_repo(BASE)) as ws:
        out = ws.grep("a - b")
        assert out.startswith("pkg/mod.py:2:")
        assert "No matches" in ws.grep("definitely_not_there")
        _raises(ws.grep, "")


def test_run_tests_reports_failure_then_success_after_fix():
    with DebugWorkspace(_repo(BASE)) as ws:
        cmd = "python -m unittest discover -s tests"
        assert not ws.run_tests(cmd).success
        ws.replace_in_file("pkg/mod.py", "return a - b", "return a + b")
        run = ws.run_tests(cmd)
        assert run.success and "OK" in run.output


def test_disallowed_test_command_is_blocked():
    with DebugWorkspace(_repo(BASE)) as ws:
        run = ws.run_tests("pytest && echo pwned")
        assert not run.success and run.returncode == -1 and "Blocked by Guardrail" in run.output


def test_oversized_repository_is_refused_and_cleaned_up():
    src = _repo({f"f{i}.py": "x = 1\n" for i in range(12)})
    try:
        DebugWorkspace(src, max_files=5)
    except WorkspaceError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("expected WorkspaceError")


def test_cleanup_removes_the_workspace_directory():
    ws = DebugWorkspace(_repo(BASE))
    root = ws.root
    assert root.exists()
    ws.cleanup()
    assert not root.exists()
