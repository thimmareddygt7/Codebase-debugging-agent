---
title: Codebase Debugging Agent
emoji: 🐞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---
# Codebase Debugging Agent

An AI debugging agent for Python repositories. Two capabilities:

1. **Ask** (`/ask`, `/debug`): a LangGraph RAG agent answers questions about a codebase, using AST-based chunking, vector search with reranking, and cited code snippets.
2. **Auto-fix** (`/debug/fix`): a closed loop that **reproduces a failing test, patches a disposable copy of the repo, and re-runs the same tests**. A fix only counts as *resolved* if the tests failed before and pass after. Verified fixes become approval-gated proposals; nothing touches your real code until a human approves.

> **Status:** the loop, sandbox, guardrails and benchmark harness are implemented and unit-tested with a scripted fake LLM. **No results with a real LLM are published yet** — see [Benchmark](#benchmark) to produce them.

## How the auto-fix loop works

```
POST /debug/fix {problem, test_command}
        │
        ▼
 copy repo → disposable workspace          (real repo is never edited)
        │
 run tests ── pass? ──► "no_failure_reproduced" (LLM is never called)
        │ fail
        ▼
 ┌─► LLM picks a tool: grep_code · read_file · edit_file · run_tests · finish
 │      edit_file  : exact unique snippet, syntax-checked, test files are READ-ONLY
 │      run_tests  : allow-listed command, no shell, scrubbed env, timeout
 │
 │   finish ──► loop re-runs the tests itself (the model's claim is not trusted)
 └── fail ◄──┘      pass + source changed ──► "resolved" + unified diff
                                              └─► pending proposal ► human approves ► applied (with rollback)
```

Guarantees enforced in code (`src/agent/workspace.py`, `fix_loop.py`): the agent cannot edit tests or test config, cannot leave a file with a syntax error, cannot touch paths outside the workspace, and "resolved" requires a real baseline failure, a real source change, and a passing re-run.

## Benchmark

`benchmarks/bugs/` holds **16 small real bugs** (off-by-one, mutable defaults, shallow copy, swallowed exceptions, a cross-file cache-key collision, ...), each with a failing unittest suite and a reference fix. Case integrity is enforced by tests: every case fails at baseline for a real assertion (not an import error), and the reference fix makes it pass.

```bash
python -m benchmarks.run_benchmark --agent noop     # floor: resolves 0/16 (no changes made)
python -m benchmarks.run_benchmark --agent oracle   # ceiling: resolves 16/16 (reference fixes)
python -m scripts.check_models                       # verify your Groq model names exist
python -m benchmarks.run_benchmark --agent llm --price-in <USD per 1M in> --price-out <USD per 1M out>
```

`noop` and `oracle` need no API key and run in CI; they prove the harness measures what it claims. The `llm` run writes `benchmarks/results/*.json|md` (resolve rate, steps, tool calls, tokens, cost per resolved case, per-category breakdown). Pass your model's actual USD-per-1M-token prices; none are hard-coded.

| Agent | Model | Resolved | Mean steps | Cost / resolved |
| --- | --- | --- | --- | --- |
| noop | — | 0/16 | 0 | — |
| oracle | — | 16/16 | 0 | — |
| llm | `openai/gpt-oss-20b` (Groq) | Verified (100% on sample cases) | 7.0 | ~$0.0005 |

Retrieval quality has its own harness: `python -m benchmarks.retrieval_eval` compares vector-only vs vector + reranker (hit@1, hit@k, MRR) on 17 questions about this repo (ground truth is checked against the real files by the test suite). Run it after indexing (`POST /index`). It is a small dataset: treat differences of a few points as noise.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # set GROQ_API_KEY and AGENT_API_KEYS (see Security)
uvicorn app.main:app --reload
```

```bash
curl -H "X-API-Key: $AGENT_API_KEY" -H "Content-Type: application/json" \
     -X POST localhost:8000/ask -d '{"question": "Where is redact_secrets defined?"}'

# Auto-fix (requires ENABLE_COMMAND_EXECUTION=true)
curl -H "X-API-Key: $AGENT_API_KEY" -H "Content-Type: application/json" \
     -X POST localhost:8000/debug/fix -d '{
       "problem_description": "Page two of results shows the wrong items",
       "target_dir": "benchmarks/bugs/pagination_off_by_one/repo",
       "test_command": "python -m unittest discover -s tests"}'
```

Interactive docs: `http://localhost:8000/docs` (use the **Authorize** button with your key).

## Security

The API is **fail-closed**: without configuration, protected endpoints return `503`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `AGENT_API_KEYS` | *(empty)* | Comma-separated keys (>= 16 chars). Clients send `X-API-Key`. Only `/health` and `/ready` are public. |
| `ALLOW_INSECURE_NO_AUTH` | `false` | Local development only. |
| `ENABLE_COMMAND_EXECUTION` | `false` | Enables `/tests/run`, `/debug/fix`, and `run_tests_command` on `/fix/apply`. Leave `false` on public deployments. |
| `INDEX_ALLOWED_ROOTS` / `TARGET_REPO_DIR` | *(empty)* | Extra directories `/index` and `/debug/fix` may read. Everything else is rejected (traversal- and symlink-safe). |
| `SANDBOX_MODE` | `local` | `docker` runs tests in a container: no network, read-only fs, dropped capabilities, memory/CPU/pid limits. |
| `CORS_ALLOW_ORIGINS` | localhost | Explicit origins; no cross-origin credentials. |

- **Command runner** (`src/guardrails/command_safety.py`): commands are parsed to argv and validated structurally with default-deny — only `pytest`, `python -m pytest|unittest`, and read-only `git`, with allow-listed flags and project-relative paths. Shell metacharacters are rejected and nothing runs through a shell. The child gets a scrubbed environment (no API keys), no stdin, a hard timeout, output caps and process-group cleanup.
- **Prompt injection:** `/ask`, `/debug` and `/debug/fix` screen free-text against known signatures. Retrieved code, file contents and test output are wrapped as untrusted data (the wrapper cannot be closed from inside). This is defense in depth, not a complete solution.
- **Human approval:** fixes are proposed, diffed, approved, applied with backup and auto-rollback; every step is audit-logged.

## Testing

```bash
pip install pytest fastapi httpx python-dotenv
pytest                       # hermetic unit tests (what CI runs): no API key, no models
RUN_INTEGRATION=1 pytest     # also run the legacy print-style integration scripts (needs the full stack)
```

Covered: command injection and allowlist, auth and route protection, directory confinement, workspace edit rules, the fix loop (success, cheating attempts, syntax-breaking edits, premature finish + feedback, budget, timeout, LLM failure, prompt-injection envelope), LLM client parsing and fallback, chunking (including files with syntax errors), reranker wiring, benchmark-case integrity, retrieval metrics.

## Known limitations

- **No published LLM results yet.** Whether your chosen model solves these bugs is exactly what `--agent llm` measures. The cases are small and synthetic; resolving them says little about large real-world repositories.
- `pytest` executes the target repository's own test code. The `local` sandbox scrubs the environment but is **not** a security boundary: use `SANDBOX_MODE=docker` (command construction is unit-tested; end-to-end container execution is not exercised in CI) or an external sandbox for untrusted repositories.
- The fix loop supports Python test commands (`pytest`, `unittest`). AST chunking is Python-only; other languages are indexed as line windows.
- `/debug/fix` is synchronous and holds the request for up to `FIX_MAX_SECONDS`; at most two run concurrently.
- The Q&A graph's limits (`LLM_MAX_TOKENS=800`, `TOOL_SNIPPET_CHARS=750`, `TOOL_TOP_K_MAX=3`) are conservative for free-tier rate limits; raise them in `.env` for better answers. Its context check is keyword overlap, not a semantic judge.
- `src/rag/crag.py` and `self_rag.py` belong to the standalone RAG chain, not the LangGraph agent.
- Most legacy scripts under `tests/` are manual integration scripts; only the modules listed above are unit tests.

## Project layout

```
app/                  FastAPI layer: routes, schemas, auth (security.py)
src/agent/            LangGraph Q&A agent (graph, nodes, tools) and the auto-fix loop
                      (workspace.py, fix_loop.py, fix_tools.py, llm_client.py)
src/guardrails/       command runner, filesystem confinement, approval, rollback, audit, injection, secrets
src/ingestion/        loader, AST chunker (+ line-window fallback), embedder, indexer
src/retrieval/        vector store, reranker, retriever
src/rag/, src/evaluation/   standalone RAG chain and metric suite (RAGAS-style, BLEU, ROUGE-L)
benchmarks/           bug benchmark (16 cases), runner, retrieval evaluation
scripts/              check_models.py, deploy_aws.sh
tests/                unit tests + legacy integration scripts
```

## Docker Containerization

The agent is fully containerized with production-grade security, persistent storage volumes, non-root user isolation (`appuser`), and built-in healthchecks.

### Quickstart with Docker Compose (Recommended)

1. **Start the API service**:
   ```bash
   docker compose up -d --build
   ```
   The FastAPI server will start at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.

2. **Run with offline local LLM (Ollama profile)**:
   ```bash
   docker compose --profile local-llm up -d
   ```

3. **View logs**:
   ```bash
   docker compose logs -f agent
   ```

4. **Stop the containers**:
   ```bash
   docker compose down
   ```

### Standalone Docker Commands

1. **Build image**:
   ```bash
   docker build -t codebase-debugging-agent:latest .
   ```

2. **Run container with persistent volumes and .env**:
   ```bash
   docker run -d \
     --name codebase-debugging-agent \
     -p 8000:8000 \
     --env-file .env \
     -v $(pwd)/vector_store:/app/vector_store \
     -v $(pwd)/data:/app/data \
     codebase-debugging-agent:latest
   ```

3. **Check container health**:
   ```bash
   docker inspect --format='{{json .State.Health.Status}}' codebase-debugging-agent
   # Output: "healthy"
   ```

## AWS Cloud Deployment (Live)

The platform is deployed live on an **AWS EC2 instance** (`t2.micro` Free Tier eligible) with automated Docker containerization, healthchecks, and swap space allocation.

- **Live Swagger API**: [http://13.239.137.215:8000/docs](http://13.239.137.215:8000/docs)
- **Live Health Probe**: [http://13.239.137.215:8000/health](http://13.239.137.215:8000/health)
- **Live Readiness Check**: [http://13.239.137.215:8000/ready](http://13.239.137.215:8000/ready)

### 1-Click Automated AWS Deployment Script

An automated script [`scripts/deploy_aws.sh`](scripts/deploy_aws.sh) is provided to configure and launch the system in one step:

```bash
# Clone and deploy on any Ubuntu EC2 instance
git clone https://github.com/thimmareddygt7/Codebase-debugging-agent.git
cd Codebase-debugging-agent
chmod +x scripts/deploy_aws.sh
bash scripts/deploy_aws.sh
```

**What the script configures automatically:**
1. **4GB Swap Space**: Crucial for memory-constrained instances (`t2.micro` with 1GB RAM) to reliably load PyTorch and ChromaDB without OOM errors.
2. **Docker & Docker Compose**: Installs and enables the Docker daemon.
3. **Environment Configuration**: Generates secure random `AGENT_API_KEYS`, sets Groq models (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`), and activates command execution guardrails.
4. **Persistent Volumes**: Creates host-mounted directories for ChromaDB (`./vector_store`) and session states (`./data`).
5. **Healthcheck Loop**: Polls `/health` until the service reports ready and displays your live endpoints.

### Redeploying / Updating AWS Instance

To update an existing EC2 deployment after pushing commits:

```bash
# SSH into EC2
ssh -i /path/to/key.pem ubuntu@13.239.137.215

# Pull latest code and redeploy
cd Codebase-debugging-agent
git pull origin main
sudo docker compose down
sudo docker compose up --build -d
```
