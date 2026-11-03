"""Tool schemas + dispatcher for the fix loop. All tools act on a DebugWorkspace."""
import logging
from typing import Any, Dict, List, Optional

from src.agent.workspace import DebugWorkspace, WorkspaceError
from src.guardrails.prompt_injection import PromptInjectionGuardrail

logger = logging.getLogger(__name__)


def _tool(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SCHEMAS: List[dict] = [
    _tool("list_files", "List files in the project (relative paths).",
          {"directory": {"type": "string", "description": "Directory, default '.'"}}, []),
    _tool("grep_code", "Case-insensitive literal text search across the project. Returns path:line: text.",
          {"pattern": {"type": "string"}}, ["pattern"]),
    _tool("read_file", "Read a file with line numbers (max 250 lines per call; use start_line/end_line to page).",
          {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _tool("edit_file",
          "Replace ONE exact, unique snippet in a source file. old_str must match the file exactly "
          "(indentation included). Test files cannot be edited.",
          {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
          ["path", "old_str", "new_str"]),
    _tool("run_tests", "Run the test command and return PASSED/FAILED with output. Omit command to use the default.",
          {"command": {"type": "string", "description": "Optional narrower test command"}}, []),
    _tool("finish", "Call ONLY when run_tests passes. Give a 1-3 sentence root-cause explanation.",
          {"summary": {"type": "string"}}, ["summary"]),
]

_WRAPPED_TOOLS = {"list_files", "grep_code", "read_file", "run_tests"}


class FixToolbox:
    def __init__(self, workspace: DebugWorkspace, default_test_command: str, test_timeout: Optional[int] = None):
        self.ws = workspace
        self.default_test_command = default_test_command
        self.test_timeout = test_timeout
        self._guard = PromptInjectionGuardrail()
        self.calls = 0
        self.finished = False
        self.finish_summary = ""
        self.last_test_run = None  # most recent TestRun from the run_tests tool

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        self.calls += 1
        if not isinstance(args, dict) or "_invalid_json" in args:
            return "ERROR: tool arguments were not valid JSON. Retry with a valid JSON object."
        try:
            result = self._dispatch(name, args)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        except (TypeError, ValueError, KeyError) as exc:
            return f"ERROR: invalid arguments for {name} ({type(exc).__name__})."
        if name in _WRAPPED_TOOLS:
            # Repository text and test output are untrusted data, never instructions.
            return self._guard.sanitize_untrusted_content(result, source_label=name)
        return result

    def _dispatch(self, name: str, a: Dict[str, Any]) -> str:
        if name == "list_files":
            return self.ws.list_files(a.get("directory") or ".")
        if name == "grep_code":
            return self.ws.grep(str(a["pattern"]))
        if name == "read_file":
            return self.ws.read_file(str(a["path"]), int(a.get("start_line") or 1),
                                     int(a["end_line"]) if a.get("end_line") else None)
        if name == "edit_file":
            return self.ws.replace_in_file(str(a["path"]), a["old_str"], a["new_str"])
        if name == "run_tests":
            run = self.ws.run_tests(a.get("command") or self.default_test_command, self.test_timeout)
            self.last_test_run = run
            verdict = "PASSED" if run.success else "FAILED"
            return f"{verdict} (exit code {run.returncode}, {run.duration_s}s)\n{run.tail()}"
        if name == "finish":
            self.finished = True
            self.finish_summary = str(a.get("summary", ""))[:1000]
            return "Finish recorded. Verifying with the test command."
        raise WorkspaceError(f"Unknown tool '{name}'. Available: list_files, grep_code, read_file, edit_file, run_tests, finish.")
