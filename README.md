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

An AI-powered codebase debugging agent utilizing RAG (Retrieval-Augmented Generation), LangGraph, and guardrails to analyze, diagnose, and fix bugs in target repositories.

## Directory Structure

```
codebase-debugging-agent/
â”‚
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ main.py                      # FastAPI entrypoint (routes, startup)
â”‚   â”œâ”€â”€ api/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ routes.py                # POST /ask, /debug, /health endpoints
â”‚   â”‚   â””â”€â”€ schemas.py               # Pydantic request/response models
â”‚   â””â”€â”€ middleware/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â””â”€â”€ guardrails.py            # Confirmation gate before destructive fixes
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ loader.py                # Load/clone target codebase
â”‚   â”‚   â”œâ”€â”€ chunker.py               # Chunk by function/class, not paragraph
â”‚   â”‚   â””â”€â”€ embedder.py              # Generate + store embeddings
â”‚   â”‚
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ vector_store.py          # Vector DB setup (Chroma/Pinecone/FAISS)
â”‚   â”‚   â”œâ”€â”€ retriever.py             # Basic + hybrid retrieval logic
â”‚   â”‚   â””â”€â”€ reranker.py              # Reranking retrieved chunks
â”‚   â”‚
â”‚   â”œâ”€â”€ rag/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ chain.py                 # Core RAG pipeline (question -> retrieve -> generate)
â”‚   â”‚   â”œâ”€â”€ citations.py             # File-level citation formatting
â”‚   â”‚   â”œâ”€â”€ crag.py                  # Corrective RAG - cross-check retrieved context
â”‚   â”‚   â””â”€â”€ self_rag.py              # Self-RAG - fact-check own answers
â”‚   â”‚
â”‚   â”œâ”€â”€ agent/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ graph.py                 # LangGraph state machine definition
â”‚   â”‚   â”œâ”€â”€ nodes.py                 # Individual graph nodes (diagnose, search, fix, verify)
â”‚   â”‚   â”œâ”€â”€ tools.py                 # Code search tool, doc search tool, etc.
â”‚   â”‚   â”œâ”€â”€ memory.py                # Short-term + long-term memory setup
â”‚   â”‚   â””â”€â”€ workflows/
â”‚   â”‚       â”œâ”€â”€ conditional.py       # Error-type branching logic
â”‚   â”‚       â””â”€â”€ iterative.py         # Retry/loop logic for wrong fixes
â”‚   â”‚
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ ragas_eval.py            # Faithfulness, relevance scoring
â”‚   â”‚   â””â”€â”€ test_queries.py          # Sample Q&A pairs for evaluation
â”‚   â”‚
â”‚   â””â”€â”€ config.py                    # Model names, chunk sizes, env settings
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ test_retrieval.py
â”‚   â”œâ”€â”€ test_agent.py
â”‚   â””â”€â”€ test_api.py
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_codebase_exploration.ipynb
â”‚   â”œâ”€â”€ 02_chunking_experiments.ipynb
â”‚   â”œâ”€â”€ 03_rag_baseline.ipynb
â”‚   â””â”€â”€ 04_agent_development.ipynb
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ .gitkeep                     # Target codebase cloned here (gitignored)
â”‚   â””â”€â”€ eval_dataset.json            # Evaluation questions + expected answers
â”‚
â”œâ”€â”€ vector_store/
â”‚   â””â”€â”€ .gitkeep                     # Persisted vector DB (gitignored)
â”‚
â”œâ”€â”€ docker/
â”‚   â”œâ”€â”€ Dockerfile
â”‚   â””â”€â”€ docker-compose.yml           # App + vector DB services together
â”‚
â”œâ”€â”€ .env.example                     # API keys template (OpenAI/Groq/etc.)
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â””â”€â”€ LICENSE
```

## Quick Notes on Key Folders

| Folder / Component | Why it's separated this way |
| --- | --- |
| **`app/` vs `src/`** | `app/` = FastAPI/web layer only; `src/` = all core logic (reusable, testable independent of the API) |
| **`rag/` vs `agent/`** | Keeps your "basic RAG" (v1-v2 from the build plan) cleanly separate from "agentic upgrades" (v4-v7) â€” mirrors your actual build stages |
| **`rag/crag.py` + `rag/self_rag.py`** | Dedicated files for the CRAG/Self-RAG upgrade stage â€” easy to point to in interviews |
| **`middleware/guardrails.py`** | Matches your Stage 12 patch (confirmation before destructive changes) |
| **`evaluation/ragas_eval.py`** | Matches your Stage 13 patch (RAGAS evaluation) |
| **`docker/`** | Separate folder keeps deployment config isolated from app logic |


# Autonomous Codebase Debugging Agent

A stateful codebase debugging agent built with **LangGraph**, Python **AST structural chunking**, and ultra-fast LLMs via **Groq LPUs** (`llama-3.3-70b-versatile`) with optional local **Ollama** support.

## Architecture & Workflow

The agent uses a cyclic LangGraph state machine rather than a simple loop:

1. **Intent Classification**: Routes queries to AST symbol search (`code_search`) or markdown/doc search (`doc_search`).
2. **Tool Execution**: Retrieves structural code chunks (functions, classes, line numbers) or setup files.
3. **Context Cross-Checking**: Evaluates semantic alignment and keyword coverage between retrieved snippets and the user query before synthesis.
4. **Fix Validation Critic**: Parses proposed Python code snippets with `ast.parse`. Syntax errors trigger an automated reflection loop that supplies tracebacks back to the LLM for correction.
5. **Session Memory**: Uses SQLite checkpointing (`SqliteSaver`) to retain investigated files, active symbols, and debugging notes across turns.

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

