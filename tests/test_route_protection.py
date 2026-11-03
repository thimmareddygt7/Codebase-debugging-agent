"""
Static guard: no route may silently escape authentication.
Parses source with `ast` -- needs no FastAPI, models, or API keys.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ALLOWED = {"/health", "/ready"}


def _route_decorators(tree: ast.AST):
    """Yield (router_name, http_method, path) for every @<router>.<method>("<path>") decorator."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name)
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                ):
                    yield dec.func.value.id, dec.func.attr, dec.args[0].value


def test_only_health_probes_are_public():
    tree = ast.parse((ROOT / "app/api/routes.py").read_text(encoding="utf-8"))
    public = {path for router, _m, path in _route_decorators(tree) if router == "public_router"}
    assert public == PUBLIC_ALLOWED, f"unexpected public routes: {public ^ PUBLIC_ALLOWED}"


def test_every_other_route_uses_the_protected_router():
    tree = ast.parse((ROOT / "app/api/routes.py").read_text(encoding="utf-8"))
    routers = {router for router, _m, _p in _route_decorators(tree)}
    assert routers <= {"router", "public_router"}, f"routes on unknown routers: {routers}"
    protected = [p for r, _m, p in _route_decorators(tree) if r == "router"]
    assert len(protected) >= 15, "expected the full API surface on the protected router"


def test_main_mounts_protected_router_behind_api_key_dependency():
    src = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert "include_router(api_router, dependencies=[Depends(require_api_key)])" in src
    assert "allow_origins=[\"*\"]" not in src
    assert "allow_credentials=False" in src


def test_command_endpoint_is_behind_the_kill_switch():
    src = (ROOT / "app/api/routes.py").read_text(encoding="utf-8")
    assert "dependencies=[Depends(require_command_execution_enabled)]" in src
    assert "ENABLE_COMMAND_EXECUTION" in src  # /fix/apply guard for run_tests_command


def test_auto_fix_endpoint_is_authenticated_and_behind_the_kill_switch():
    tree = ast.parse((ROOT / "app/api/routes.py").read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "auto_fix")
    dec = func.decorator_list[0]
    assert dec.func.value.id == "router", "auto_fix must be on the authenticated router"
    assert dec.args[0].value == "/debug/fix"
    deps = next(k for k in dec.keywords if k.arg == "dependencies")
    assert "require_command_execution_enabled" in ast.dump(deps.value)
