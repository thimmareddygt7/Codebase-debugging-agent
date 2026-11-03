"""
Security tests for CommandSafetyGuardrail.
No LLM, API key, or embedding model required -- safe to run in CI.
"""
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from src.guardrails.command_safety import CommandSafetyGuardrail, CommandRejected, build_docker_argv

guard = CommandSafetyGuardrail()


def _allowed(cmd: str) -> bool:
    return guard.is_command_allowed(cmd)[0]


def test_shell_injection_is_rejected():
    payloads = [
        "pytest tests/ && python -c 'print(1)'",
        "pytest; ls",
        "pytest | cat",
        "pytest `id`",
        "pytest $(id)",
        "pytest > out.txt",
        "pytest < /etc/passwd",
        "pytest tests/ & whoami",
        "pytest\nrm -rf /",
        "pytest\r\nwhoami",
    ]
    for cmd in payloads:
        assert not _allowed(cmd), f"should be rejected: {cmd!r}"


def test_arbitrary_python_is_rejected():
    for cmd in [
        "python -c 'print(1)'",
        'python -c "exit(1)"',
        "python script.py",
        "python tests/evil.py",
        "python3 -m http.server",
        "python -m pip install requests",
    ]:
        assert not _allowed(cmd), f"should be rejected: {cmd!r}"


def test_dangerous_pytest_options_are_rejected():
    for cmd in [
        "pytest -p evil_plugin",
        "pytest -c /tmp/x.ini",
        "pytest --rootdir=/",
        "pytest -o addopts=--foo",
        "pytest --basetemp=/",
        "pytest --pyargs os",
        "pytest ../../etc",
        "pytest /etc/passwd",
        "pytest C:/Windows",
        "pytest .git/config",
        "pytest tests/ --tb=evil",
        "pytest --maxfail=999999",
        "pytest -k",
    ]:
        assert not _allowed(cmd), f"should be rejected: {cmd!r}"


def test_destructive_and_unlisted_commands_are_rejected():
    for cmd in [
        "rm -rf /",
        "del /f /q C:\\Windows",
        "git push --force origin main",
        "git reset --hard",
        "git clean -fd",
        "git checkout main",
        "git branch -D main",
        "git branch newbranch",
        "git diff --ext-diff",
        "git diff --output=/tmp/x",
        "git log --output=/tmp/x",
        "git -c core.pager=evil status",
        "git",
        "format D:",
        "curl http://evil.example",
        "pip install requests",
        "",
        "   ",
    ]:
        assert not _allowed(cmd), f"should be rejected: {cmd!r}"


def test_legitimate_commands_are_allowed():
    for cmd in [
        "pytest",
        "pytest tests/ -v",
        "python -m pytest -q -x tests/test_a.py::test_b",
        "pytest -k 'foo and bar' tests/",
        "pytest --tb=short --maxfail=3",
        "python -m unittest discover -s tests -v",
        "git status",
        "git status --short",
        "git diff --stat",
        "git diff --cached -- src/config.py",
        "git log -n 5",
        "git log --oneline",
        "git branch",
    ]:
        ok, _, reason = guard.is_command_allowed(cmd)
        assert ok, f"should be allowed: {cmd!r} ({reason})"


def test_parse_builds_argv_without_a_shell():
    argv = guard.parse_command("pytest -k 'a and b' tests/")
    assert argv[0] == sys.executable
    assert "a and b" in argv  # kept as ONE argument, not re-split by a shell
    assert argv[:3] == [sys.executable, "-m", "pytest"]


def test_git_log_gets_a_bounded_default_count():
    argv = guard.parse_command("git log --oneline")
    assert "-n" in argv and argv[argv.index("-n") + 1] == "20"
    try:
        guard.parse_command("git log -n 100000")
    except CommandRejected:
        pass
    else:
        raise AssertionError("oversized git log count must be rejected")


def _make_project(test_source: str) -> str:
    root = tempfile.mkdtemp(prefix="cmdguard_")
    tests = Path(root) / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(textwrap.dedent(test_source), encoding="utf-8")
    return root


def test_execution_runs_without_shell_and_scrubs_secrets():
    root = _make_project(
        """
        import os, unittest

        class T(unittest.TestCase):
            def test_env_is_scrubbed(self):
                self.assertNotIn("GROQ_API_KEY", os.environ)
                self.assertNotIn("LANGSMITH_API_KEY", os.environ)
        """
    )
    os.environ["GROQ_API_KEY"] = "gsk_should_never_reach_child"
    os.environ["LANGSMITH_API_KEY"] = "lsv2_should_never_reach_child"
    try:
        ok, code, out, err = guard.execute_safe_command(
            "python -m unittest discover -s tests", timeout=30, cwd=root
        )
    finally:
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("LANGSMITH_API_KEY", None)
    assert ok, f"child saw secrets or failed: code={code} err={err}"
    assert "OK" in err


def test_failing_tests_report_failure():
    root = _make_project(
        """
        import unittest

        class T(unittest.TestCase):
            def test_fails(self):
                self.assertEqual(1, 2)
        """
    )
    ok, code, _, err = guard.execute_safe_command(
        "python -m unittest discover -s tests", timeout=30, cwd=root
    )
    assert not ok and code != 0
    assert "FAILED" in err


def test_timeout_kills_runaway_process():
    root = _make_project(
        """
        import time, unittest

        class T(unittest.TestCase):
            def test_hangs(self):
                time.sleep(60)
        """
    )
    started = time.time()
    ok, code, _, err = guard.execute_safe_command(
        "python -m unittest discover -s tests", timeout=2, cwd=root
    )
    assert not ok
    assert "timed out" in err
    assert time.time() - started < 20


def test_blocked_command_never_spawns_a_process():
    ok, code, out, err = guard.execute_safe_command("pytest && echo pwned")
    assert not ok and code == -1
    assert "Blocked by Guardrail" in err
    assert "pwned" not in out


def test_docker_wrapper_is_locked_down():
    argv = guard.parse_command("python -m unittest discover -s tests")
    docker_argv, name = build_docker_argv(argv, Path("/tmp/ws"), "python:3.11-slim")
    joined = " ".join(docker_argv)
    for required in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                     "--memory", "--pids-limit", "--rm"):
        assert required in joined, required
    assert docker_argv[:2] == ["docker", "run"] and "--privileged" not in joined
    assert docker_argv.count("-v") == 1                       # only the workspace is mounted
    image_idx = docker_argv.index("python:3.11-slim")
    assert docker_argv[image_idx + 1] == "python"              # host interpreter path replaced
    assert sys.executable not in docker_argv
    assert build_docker_argv(argv, Path("/tmp/ws"), "img")[1] != name   # unique container names
