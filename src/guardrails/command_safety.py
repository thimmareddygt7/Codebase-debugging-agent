"""
Command-execution guardrail.

Design (replaces the previous prefix-allowlist + ``shell=True`` approach, which
allowed ``pytest && <anything>`` to run arbitrary shell commands):

1. Reject any command containing shell metacharacters or control characters.
2. Parse the command into an argv list with ``shlex`` and validate it
   *structurally*: tool, sub-command, every flag and every path argument must
   be explicitly allowed. Anything unknown is rejected (default deny).
3. Execute the validated argv with ``shell=False``, a scrubbed environment
   (no API keys), no stdin, a hard timeout, output caps and process-group
   cleanup.

IMPORTANT: this guardrail cannot make running *untrusted repository code*
safe -- ``pytest`` executes whatever the repository's tests/conftest contain.
For untrusted repos, run this process inside a container with no network
and no secrets (see README "Security").
"""
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

from src.config import settings
from src.guardrails.secrets import redact_secrets

DEFAULT_TIMEOUT_SECONDS = 45
MAX_TIMEOUT_SECONDS = 180
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB max stdout/stderr returned
MAX_COMMAND_LENGTH = 300

# Characters that only have meaning to a shell (or that smuggle extra commands).
# Backslash is rejected too so Windows-style paths cannot be ambiguous; use "/".
_FORBIDDEN_CHARS = set(";&|`$<>()\\\x00")

_PYTHON_NAMES = {"python", "python3"}

_PYTEST_FLAGS = {
    "-q", "-qq", "-v", "-vv", "-x", "--exitfirst", "-l", "--showlocals",
    "--lf", "--ff", "--co", "--collect-only", "--strict-markers",
    "--disable-warnings", "-ra", "-rA", "-rf", "-rfE",
}
_PYTEST_TB = {"auto", "long", "short", "line", "native", "no"}
_EXPR_RE = re.compile(r"[\w\s.\[\]():,=-]{1,200}")
_NODE_SUFFIX_RE = re.compile(r"[\w.\[\]:,= /-]*")

_UNITTEST_FLAGS = {
    "-v", "-q", "-b", "-f", "--verbose", "--quiet", "--buffer", "--failfast",
}
_DOTTED_NAME_RE = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
_PATTERN_RE = re.compile(r"[\w*.-]{1,100}")

_GIT_STATUS_FLAGS = {"-s", "--short", "-b", "--branch", "--porcelain"}
_GIT_BRANCH_FLAGS = {"-a", "-r", "-v", "--list", "--all", "--remotes"}
_GIT_DIFF_FLAGS = {
    "--stat", "--shortstat", "--name-only", "--name-status",
    "--cached", "--staged", "--no-color",
}
_GIT_LOG_FLAGS = {"--oneline", "--stat", "--no-color"}
_GIT_LOG_MAX_COUNT = 50

# Only these variables reach the child process. API keys etc. are NOT passed.
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "VIRTUAL_ENV", "PYTHONPATH",
)


def build_docker_argv(
    argv: List[str],
    work_dir: Path,
    image: str,
    memory: str = "1g",
    cpus: str = "1.0",
    pids_limit: int = 256,
) -> Tuple[List[str], str]:
    """
    Wrap a validated argv so it runs inside a locked-down container:
    no network, read-only root fs, all capabilities dropped, memory/cpu/pid limits,
    only the workspace mounted. Returns (docker_argv, container_name).
    """
    name = f"dbgagent-{uuid.uuid4().hex[:12]}"
    inner = list(argv)
    if inner and inner[0] == sys.executable:
        inner[0] = "python"  # the host interpreter path does not exist in the container
    docker_argv = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", memory, "--cpus", cpus, "--pids-limit", str(pids_limit),
        "-v", f"{Path(work_dir).resolve()}:/work:rw", "-w", "/work",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "CI=1",
        image, *inner,
    ]
    return docker_argv, name


class CommandRejected(Exception):
    """Raised when a command fails validation."""

    def __init__(self, reason: str, hard: bool = True):
        super().__init__(reason)
        self.reason = reason
        # hard=True: never acceptable. hard=False: simply not on the allowlist.
        self.hard = hard


def _path_is_safe(arg: str, root: Path) -> bool:
    """Relative path, no traversal, no dot-dirs like .git/.env, stays inside root."""
    if not arg or arg.startswith("-") or arg.startswith("/") or re.match(r"^[A-Za-z]:", arg):
        return False
    parts = PurePosixPath(arg).parts
    if ".." in parts or any(p in {".git", ".env"} or p.startswith(".env.") for p in parts):
        return False
    try:
        (root / arg).resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def _check_path(arg: str, root: Path) -> None:
    if not _path_is_safe(arg, root):
        raise CommandRejected(f"Path argument '{arg[:60]}' is not permitted (must be a relative path inside the project).")


def _build_safe_env() -> dict:
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "CI": "1",
    })
    return env


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _clip(data: bytes, label: str) -> str:
    text = data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(data) > MAX_OUTPUT_BYTES:
        text += f"\n... [{label} truncated by Guardrail: Max 50KB exceeded]"
    return redact_secrets(text)


class CommandSafetyGuardrail:
    """Validates and safely executes a small allowlist of test / read-only git commands."""

    # ------------------------------------------------------------------ parse
    def parse_command(self, command_str: str, cwd: Optional[str] = None) -> List[str]:
        """Return a validated argv list (no shell) or raise CommandRejected."""
        root = Path(cwd or settings.PROJECT_ROOT).resolve()

        if not command_str or not command_str.strip():
            raise CommandRejected("Empty command string.")
        cmd = command_str.strip()
        if len(cmd) > MAX_COMMAND_LENGTH:
            raise CommandRejected(f"Command too long (max {MAX_COMMAND_LENGTH} characters).")

        bad = sorted({c for c in cmd if c in _FORBIDDEN_CHARS or (ord(c) < 32 and c != "\t")})
        if bad:
            shown = " ".join(repr(c) for c in bad)
            raise CommandRejected(f"Command contains forbidden shell/control characters: {shown}.")

        try:
            tokens = shlex.split(cmd)
        except ValueError:
            raise CommandRejected("Malformed command (unbalanced quotes).")
        if not tokens:
            raise CommandRejected("Malformed command.")

        head = tokens[0].lower()
        if head == "pytest":
            return self._build_pytest(tokens[1:], root)
        if head in _PYTHON_NAMES:
            if tokens[1:3] == ["-m", "pytest"]:
                return self._build_pytest(tokens[3:], root)
            if tokens[1:3] == ["-m", "unittest"]:
                return self._build_unittest(tokens[3:], root)
            raise CommandRejected(
                "Only 'python -m pytest' and 'python -m unittest' are permitted; "
                "arbitrary python execution is blocked."
            )
        if head == "git":
            return self._build_git(tokens[1:], root)

        raise CommandRejected(
            f"Command '{tokens[0][:30]}' is not in the approved allowlist.", hard=False
        )

    def _build_pytest(self, rest: List[str], root: Path) -> List[str]:
        args: List[str] = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in _PYTEST_FLAGS:
                args.append(a)
            elif a in ("-k", "-m"):
                i += 1
                if i >= len(rest) or rest[i].startswith("-") or not _EXPR_RE.fullmatch(rest[i]):
                    raise CommandRejected(f"Invalid or missing expression for pytest option '{a}'.")
                args += [a, rest[i]]
            elif a.startswith("--tb="):
                if a[5:] not in _PYTEST_TB:
                    raise CommandRejected(f"Invalid value for --tb: '{a[5:][:20]}'.")
                args.append(a)
            elif a.startswith("--maxfail="):
                val = a.split("=", 1)[1]
                if not (val.isdigit() and 1 <= int(val) <= 100):
                    raise CommandRejected("--maxfail must be an integer from 1 to 100.")
                args.append(a)
            elif a.startswith("-"):
                raise CommandRejected(f"pytest option '{a[:40]}' is not permitted.")
            else:
                path, _, node = a.partition("::")
                _check_path(path, root)
                if node and not _NODE_SUFFIX_RE.fullmatch(node):
                    raise CommandRejected("Invalid pytest node id.")
                args.append(a)
            i += 1
        return [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args]

    def _build_unittest(self, rest: List[str], root: Path) -> List[str]:
        args: List[str] = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in _UNITTEST_FLAGS or a == "discover":
                args.append(a)
            elif a in ("-s", "-p"):
                i += 1
                if i >= len(rest):
                    raise CommandRejected(f"Missing value for unittest option '{a}'.")
                if a == "-s":
                    _check_path(rest[i], root)
                elif not _PATTERN_RE.fullmatch(rest[i]):
                    raise CommandRejected("Invalid unittest -p pattern.")
                args += [a, rest[i]]
            elif a.startswith("-"):
                raise CommandRejected(f"unittest option '{a[:40]}' is not permitted.")
            elif _DOTTED_NAME_RE.fullmatch(a):
                args.append(a)
            else:
                _check_path(a, root)
                args.append(a)
            i += 1
        return [sys.executable, "-m", "unittest", *args]

    def _build_git(self, rest: List[str], root: Path) -> List[str]:
        if not rest:
            raise CommandRejected("Bare 'git' is not permitted.")
        sub, args = rest[0], rest[1:]
        base = ["git", "--no-pager", "-c", "core.fsmonitor=false"]

        if sub == "status":
            self._require_subset(args, _GIT_STATUS_FLAGS, "git status")
            return base + ["status", *args]
        if sub == "branch":
            self._require_subset(args, _GIT_BRANCH_FLAGS, "git branch")
            return base + ["branch", *args]
        if sub == "diff":
            if "--" in args:
                idx = args.index("--")
                flags, paths = args[:idx], args[idx + 1:]
            else:
                flags, paths = args, []
            self._require_subset(flags, _GIT_DIFF_FLAGS, "git diff")
            for p in paths:
                _check_path(p, root)
            tail = ["--", *paths] if paths else []
            return base + ["diff", "--no-ext-diff", "--no-textconv", *flags, *tail]
        if sub == "log":
            count, flags, i = 20, [], 0
            while i < len(args):
                a = args[i]
                if a in _GIT_LOG_FLAGS:
                    flags.append(a)
                elif a == "-n" and i + 1 < len(args):
                    i += 1
                    count = self._parse_count(args[i])
                elif re.fullmatch(r"-n\d+", a):
                    count = self._parse_count(a[2:])
                elif a.startswith("--max-count="):
                    count = self._parse_count(a.split("=", 1)[1])
                else:
                    raise CommandRejected(f"git log option '{a[:40]}' is not permitted.")
                i += 1
            return base + ["log", "--no-ext-diff", "--no-textconv", "-n", str(count), *flags]

        raise CommandRejected(f"git sub-command '{sub[:20]}' is not permitted (read-only commands only).")

    @staticmethod
    def _require_subset(args: List[str], allowed: set, label: str) -> None:
        for a in args:
            if a not in allowed:
                raise CommandRejected(f"{label} option '{a[:40]}' is not permitted.")

    @staticmethod
    def _parse_count(value: str) -> int:
        if not (value.isdigit() and 1 <= int(value) <= _GIT_LOG_MAX_COUNT):
            raise CommandRejected(f"git log count must be an integer from 1 to {_GIT_LOG_MAX_COUNT}.")
        return int(value)

    # ------------------------------------------------------------------ public
    def is_command_allowed(self, command_str: str) -> Tuple[bool, bool, str]:
        """
        Returns (is_permitted, requires_human_confirmation, rationale).
        Commands that are not allowlisted are rejected (never auto-executed).
        """
        try:
            self.parse_command(command_str)
        except CommandRejected as rej:
            return False, (not rej.hard), f"Command Rejected: {rej.reason}"
        return True, False, "Allowed safe command."

    def execute_safe_command(
        self,
        command_str: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        cwd: Optional[str] = None,
        docker_image: Optional[str] = None,
    ) -> Tuple[bool, int, str, str]:
        """
        Validates then executes a command WITHOUT a shell.
        With ``docker_image`` set, the validated command runs inside a locked-down container
        (no network, read-only root fs, resource limits) with only ``cwd`` mounted.
        Returns (success, returncode, stdout, stderr).
        """
        work_dir = Path(cwd or settings.PROJECT_ROOT).resolve()
        try:
            argv = self.parse_command(command_str, cwd=str(work_dir))
        except CommandRejected as rej:
            return False, -1, "", f"Execution Blocked by Guardrail: {rej.reason}"

        timeout = max(1, min(int(timeout), MAX_TIMEOUT_SECONDS))
        container_name = None
        if docker_image:
            argv, container_name = build_docker_argv(argv, work_dir, docker_image)
        popen_kwargs = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(work_dir),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_build_safe_env(),
                **popen_kwargs,
            )
        except Exception as exc:
            return False, -1, "", f"Execution Error: {type(exc).__name__}"

        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            if container_name:  # killing the CLI does not stop the container
                try:
                    subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=15)
                except Exception:
                    pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return False, -1, "", f"Execution Blocked by Guardrail: Command timed out after {timeout} seconds."

        return (proc.returncode == 0), proc.returncode, _clip(out, "Output"), _clip(err, "Stderr")
