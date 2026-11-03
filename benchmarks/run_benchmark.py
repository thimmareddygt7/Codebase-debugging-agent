"""
Benchmark harness for the debugging agent.

    python -m benchmarks.run_benchmark --agent noop      # sanity floor: must resolve 0%
    python -m benchmarks.run_benchmark --agent oracle    # reference fixes: must resolve 100%
    python -m benchmarks.run_benchmark --agent llm       # the real agent (needs GROQ_API_KEY)

Each case is a tiny repository with a failing test suite. A case counts as RESOLVED only if
the same test command that failed at baseline passes afterwards AND source files changed
(see src/agent/fix_loop.py). The agent cannot edit test files.

`noop` and `oracle` need no LLM; they prove the harness measures what it claims to.
Optional cost: pass --price-in / --price-out (USD per 1M tokens) for your model.
"""
import argparse
import json
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.fix_loop import FixLoop
from src.agent.workspace import DebugWorkspace

CASES_DIR = Path(__file__).resolve().parent / "bugs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_cases(ids: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cases = []
    for case_dir in sorted(p for p in CASES_DIR.iterdir() if (p / "case.json").is_file()):
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        if ids and meta["id"] not in ids:
            continue
        meta["dir"] = case_dir
        cases.append(meta)
    return cases[:limit] if limit else cases


def _record(case: dict, **kw) -> Dict[str, Any]:
    base = {"id": case["id"], "category": case["category"], "difficulty": case["difficulty"],
            "status": "unresolved", "resolved": False, "steps": 0, "tool_calls": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "duration_s": 0.0,
            "changed_files": [], "diff_chars": 0, "error": None, "transcript": []}
    base.update(kw)
    return base


def run_noop(case: dict, **_) -> Dict[str, Any]:
    started = time.time()
    with DebugWorkspace(str(case["dir"] / "repo")) as ws:
        run = ws.run_tests(case["test_command"])
    status = "no_failure_reproduced" if run.success else "unresolved"
    return _record(case, status=status, duration_s=round(time.time() - started, 2))


def run_oracle(case: dict, **_) -> Dict[str, Any]:
    started = time.time()
    with DebugWorkspace(str(case["dir"] / "repo")) as ws:
        baseline = ws.run_tests(case["test_command"])
        if baseline.success:
            return _record(case, status="no_failure_reproduced", duration_s=round(time.time() - started, 2))
        for rel in case["fixed_files"]:
            shutil.copyfile(case["dir"] / "fixed" / rel, ws.root / rel)
        final = ws.run_tests(case["test_command"])
    return _record(case, status="resolved" if final.success else "unresolved", resolved=final.success,
                   changed_files=list(case["fixed_files"]), duration_s=round(time.time() - started, 2))


def run_llm(case: dict, llm=None, max_steps: Optional[int] = None, **_) -> Dict[str, Any]:
    result = FixLoop(llm).run(
        source_dir=str(case["dir"] / "repo"),
        problem_description=case["description"],
        test_command=case["test_command"],
        max_steps=max_steps,
    )
    return _record(
        case, status=result.status, resolved=result.resolved, steps=result.steps,
        tool_calls=result.tool_calls, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, duration_s=result.duration_s,
        changed_files=result.changed_files, diff_chars=len(result.diff), error=result.error,
        transcript=result.transcript,
    )


RUNNERS = {"noop": run_noop, "oracle": run_oracle, "llm": run_llm}


def summarize(records: List[Dict[str, Any]], price_in: Optional[float] = None,
              price_out: Optional[float] = None) -> Dict[str, Any]:
    n = len(records)
    resolved = sum(r["resolved"] for r in records)
    mean = lambda key: round(sum(r[key] for r in records) / n, 2) if n else 0.0
    summary: Dict[str, Any] = {
        "cases": n, "resolved": resolved,
        "resolve_rate": round(resolved / n, 4) if n else 0.0,
        "mean_steps": mean("steps"), "mean_tool_calls": mean("tool_calls"),
        "mean_duration_s": mean("duration_s"),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "total_completion_tokens": sum(r["completion_tokens"] for r in records),
    }
    if price_in is not None and price_out is not None and n:
        cost = (summary["total_prompt_tokens"] * price_in + summary["total_completion_tokens"] * price_out) / 1e6
        summary["total_cost_usd"] = round(cost, 4)
        summary["cost_per_case_usd"] = round(cost / n, 4)
        summary["cost_per_resolved_usd"] = round(cost / resolved, 4) if resolved else None
    for field in ("category", "difficulty"):
        groups = defaultdict(lambda: [0, 0])
        for r in records:
            groups[r[field]][1] += 1
            groups[r[field]][0] += int(r["resolved"])
        summary[f"by_{field}"] = {k: f"{a}/{b}" for k, (a, b) in sorted(groups.items())}
    return summary


def render_markdown(meta: dict, summary: dict, records: List[dict]) -> str:
    lines = [
        f"# Benchmark results: `{meta['agent']}`",
        "",
        f"- Date (UTC): {meta['timestamp']}",
        f"- Model: {meta.get('model') or 'n/a'}   |   max steps: {meta.get('max_steps') or 'default'}",
        f"- Python {meta['python']} on {meta['platform']}",
        "",
        f"**Resolved {summary['resolved']}/{summary['cases']} ({summary['resolve_rate']:.0%})**  "
        f"| mean steps {summary['mean_steps']} | mean tool calls {summary['mean_tool_calls']} "
        f"| mean time {summary['mean_duration_s']}s | tokens {summary['total_prompt_tokens']:,} in / "
        f"{summary['total_completion_tokens']:,} out",
    ]
    if "total_cost_usd" in summary:
        per = summary["cost_per_resolved_usd"]
        lines.append(f"\nCost: ${summary['total_cost_usd']} total, ${summary['cost_per_case_usd']} per case, "
                     f"{'$' + str(per) if per is not None else 'n/a'} per resolved case.")
    lines += ["", "| case | category | difficulty | status | steps | tools | time (s) |", "|---|---|---|---|---|---|---|"]
    for r in records:
        mark = "PASS" if r["resolved"] else "FAIL"
        lines.append(f"| {r['id']} | {r['category']} | {r['difficulty']} | {mark} ({r['status']}) | "
                     f"{r['steps']} | {r['tool_calls']} | {r['duration_s']} |")
    lines += ["", f"By difficulty: {summary['by_difficulty']}", f"By category: {summary['by_category']}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=sorted(RUNNERS), default="llm")
    ap.add_argument("--ids", nargs="*", help="Run only these case ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--price-in", type=float, help="USD per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, help="USD per 1M completion tokens")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    cases = load_cases(args.ids, args.limit)
    if not cases:
        print("No cases found.", file=sys.stderr)
        return 2

    llm, model = None, None
    if args.agent == "llm":
        from src.agent.llm_client import LLMUnavailable, get_default_llm
        from src.config import settings
        try:
            llm = get_default_llm()
        except LLMUnavailable as exc:
            print(f"Cannot run the LLM benchmark: {exc}", file=sys.stderr)
            return 2
        model = settings.OLLAMA_MODEL if settings.MODEL_PROVIDER == "ollama" else settings.GROQ_MODEL

    records = []
    for i, case in enumerate(cases, 1):
        rec = RUNNERS[args.agent](case, llm=llm, max_steps=args.max_steps)
        records.append(rec)
        print(f"[{i}/{len(cases)}] {case['id']:<34} {'PASS' if rec['resolved'] else 'FAIL'}  "
              f"({rec['status']}, {rec['steps']} steps, {rec['duration_s']}s)", flush=True)

    meta = {"agent": args.agent, "model": model, "max_steps": args.max_steps,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "python": platform.python_version(), "platform": platform.platform()}
    summary = summarize(records, args.price_in, args.price_out)
    report = render_markdown(meta, summary, records)
    print("\n" + report)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stem = f"{args.agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        (RESULTS_DIR / f"{stem}.json").write_text(
            json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2), encoding="utf-8")
        (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
        print(f"Saved to benchmarks/results/{stem}.(json|md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
