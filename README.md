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

An AI-powered codebase debugging agent utilizing RAG (Retrieval-Augmented Generation), LangGraph cyclic state machines, and multi-layered guardrails to analyze, diagnose, and fix bugs in target repositories.

## Directory Structure

```
codebase-debugging-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entrypoint (routes, CORS, startup)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # REST endpoints (/ask, /debug, /index, /fix, /tests, /health)
│   │   └── schemas.py               # Pydantic request/response models
│   └── middleware/
│       ├── __init__.py
│       └── guardrails.py            # Confirmation gate before destructive fixes
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Settings, model configuration, project paths
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                # Load/clone target codebase files
│   │   ├── chunker.py               # Python AST chunking (functions, classes, methods)
│   │   ├── embedder.py              # SentenceTransformer vector embedding generation
│   │   └── indexer.py               # Automated background indexing pipeline
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Persistent ChromaDB client and collection management
│   │   ├── retriever.py             # Vector similarity search engine
│   │   └── reranker.py              # Hybrid lexical & semantic reranking
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chain.py                 # Core RAG pipeline (question -> retrieve -> synthesize)
│   │   ├── citations.py             # File-level and line-range citation formatting
│   │   ├── citation_verifier.py     # Source citation verification
│   │   ├── context_builder.py       # Context window budgeting and snippet assembly
│   │   ├── crag.py                  # Corrective RAG (CRAG) relevance evaluation
│   │   ├── self_rag.py              # Self-RAG reflection and faithfulness fact-checking
│   │   └── llm.py                   # Groq LPU and Ollama LLM provider factory
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph cyclic state machine with SQLite checkpointing
│   │   ├── nodes.py                 # Core graph nodes (intent, LLM, tools, verifier, critic)
│   │   ├── tools.py                 # Tool interfaces (code_search, doc_search, read_file)
│   │   ├── memory.py                # SessionManager and AgentState definition
│   │   ├── validator.py             # AST syntax and integrity validation critic
│   │   ├── context_verifier.py      # Context cross-checker anti-hallucination guardrail
│   │   └── workflows/
│   │       ├── __init__.py
│   │       ├── conditional.py       # Error-type routing and tool execution branching
│   │       └── iterative.py         # Self-correction reflection loop and retry budgeting
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── manager.py               # Central GuardrailManager coordinator
│   │   ├── human_approval.py        # Explicit confirmation gate before file modifications
│   │   ├── filesystem.py            # Path traversal and protected file boundaries
│   │   ├── command_safety.py        # Strict execution allowlist & forbidden command blocker
│   │   ├── secrets.py               # Secret masking and credential leak prevention
│   │   ├── patch_validator.py       # Git diff syntax and file integrity validation
│   │   ├── prompt_injection.py      # Adversarial prompt injection defense
│   │   ├── fix_manager.py           # Patch proposal, application, and rollback tracker
│   │   └── audit.py                 # Tamper-evident JSONL audit trail logger
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── lexical_metrics.py       # BLEU, ROUGE-L, and Exact Match scoring
│   │   ├── semantic_metrics.py      # Cosine semantic similarity evaluation
│   │   ├── ragas_eval.py            # Faithfulness, answer relevance, and context precision
│   │   └── test_queries.py          # Benchmark dataset and evaluation test cases
│   │
│   └── observability/
│       ├── __init__.py
│       └── tracker.py               # LangSmith tracing and run telemetry
│
├── tests/
│   ├── __init__.py
│   ├── test_guardrails.py           # Multi-layered safety and security test suite
│   ├── test_api_endpoints.py        # FastAPI endpoint integration tests
│   ├── test_chunker.py              # AST code chunker tests
│   ├── test_nested_and_decorators.py # Nested class and decorator chunking tests
│   ├── test_vector_store.py         # ChromaDB vector store read/write tests
│   ├── test_retriever.py            # Vector retrieval and search tests
│   ├── test_rag_chain.py            # RAG pipeline end-to-end tests
│   ├── test_fix_retry_loop.py       # AST fix validator and self-correction tests
│   ├── test_agent_nodes.py          # LangGraph node execution tests
│   └── ...                          # 36+ comprehensive unit and integration tests
│
├── data/
│   ├── .gitkeep
│   ├── eval_dataset.json            # Curated evaluation dataset
│   └── backups/                     # Automatic pre-patch rollback backups (gitignored)
│
├── docs/
│   └── Codebase_Debugging_Agent.docx # System documentation and specification
│
├── notebooks/                       # Exploratory analysis and prototyping notebooks
│   ├── 01_codebase_exploration.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_rag_baseline.ipynb
│   └── 04_agent_development.ipynb
│
├── scripts/
│   └── deploy_aws.sh                # 1-Click AWS EC2 deployment automation script
│
├── docker/
│   ├── Dockerfile                   # Isolated container definition
│   └── docker-compose.yml           # Multi-container orchestration config
│
├── Dockerfile                       # Production root Dockerfile (Hugging Face / Cloud Run compatible)
├── docker-compose.yml               # Production root Docker Compose specification
├── .dockerignore                    # Docker build exclusion rules
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git exclusion rules
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md
```

## Architectural Highlights

| Component | Responsibility |
| --- | --- |
| **`app/` vs `src/`** | `app/` contains the web/API layer (FastAPI, schemas, routing); `src/` houses all domain logic, agents, RAG, and guardrails (testable independently of HTTP). |
| **`rag/` vs `agent/`** | Keeps standard RAG (retrieval, context building, citations) decoupled from the agentic graph (LangGraph state machine, tool reasoning, reflection loops). |
| **`rag/crag.py` & `rag/self_rag.py`** | Corrective RAG (CRAG) relevance scoring and Self-RAG reflection to eliminate hallucinations. |
| **`guardrails/`** | Comprehensive security layer: human approval gates, protected file paths, shell allowlists, secret redaction, and rollback backups. |
| **`evaluation/`** | Deep evaluation suite with RAGAS metrics, BLEU, ROUGE-L, and semantic similarity scoring. |

---

# Autonomous Codebase Debugging Agent

A stateful codebase debugging agent built with **LangGraph**, Python **AST structural chunking**, and ultra-fast LLMs via **Groq LPUs** (`llama-3.3-70b-versatile` / `qwen/qwen3.8-27b`) with optional local **Ollama** support.

## Architecture & Workflow

The agent uses a cyclic LangGraph state machine:

1. **Intent Classification**: Routes incoming queries to AST symbol search (`code_search`) or documentation search (`doc_search`).
2. **Tool Execution**: Retrieves structural code chunks (functions, classes, line numbers) and setup configurations.
3. **Context Cross-Checking (CRAG)**: Evaluates semantic alignment and keyword coverage between retrieved snippets and user queries before synthesis.
4. **Fix Validation Critic**: Parses proposed Python code snippets with `ast.parse`. Syntax errors automatically trigger a reflection loop with error tracebacks back to the LLM.
5. **Session Memory**: Uses SQLite checkpointing (`SqliteSaver`) to retain investigated files, active symbols, and debugging notes across multi-turn conversations.

## Setup

1. **Clone and create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY and optionally LANGCHAIN_API_KEY
   ```

3. **Run local FastAPI server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

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
