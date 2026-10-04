# Codebase Debugging Agent

An AI-powered codebase debugging agent utilizing RAG (Retrieval-Augmented Generation), LangGraph, and guardrails to analyze, diagnose, and fix bugs in target repositories.

## Directory Structure

```
codebase-debugging-agent/
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entrypoint (routes, startup)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # POST /ask, /debug, /health endpoints
│   │   └── schemas.py               # Pydantic request/response models
│   └── middleware/
│       ├── __init__.py
│       └── guardrails.py            # Confirmation gate before destructive fixes
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                # Load/clone target codebase
│   │   ├── chunker.py               # Chunk by function/class, not paragraph
│   │   └── embedder.py              # Generate + store embeddings
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Vector DB setup (Chroma/Pinecone/FAISS)
│   │   ├── retriever.py             # Basic + hybrid retrieval logic
│   │   └── reranker.py              # Reranking retrieved chunks
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chain.py                 # Core RAG pipeline (question -> retrieve -> generate)
│   │   ├── citations.py             # File-level citation formatting
│   │   ├── crag.py                  # Corrective RAG - cross-check retrieved context
│   │   └── self_rag.py              # Self-RAG - fact-check own answers
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph state machine definition
│   │   ├── nodes.py                 # Individual graph nodes (diagnose, search, fix, verify)
│   │   ├── tools.py                 # Code search tool, doc search tool, etc.
│   │   ├── memory.py                # Short-term + long-term memory setup
│   │   └── workflows/
│   │       ├── conditional.py       # Error-type branching logic
│   │       └── iterative.py         # Retry/loop logic for wrong fixes
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_eval.py            # Faithfulness, relevance scoring
│   │   └── test_queries.py          # Sample Q&A pairs for evaluation
│   │
│   └── config.py                    # Model names, chunk sizes, env settings
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── test_api.py
│
├── notebooks/
│   ├── 01_codebase_exploration.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_rag_baseline.ipynb
│   └── 04_agent_development.ipynb
│
├── data/
│   ├── .gitkeep                     # Target codebase cloned here (gitignored)
│   └── eval_dataset.json            # Evaluation questions + expected answers
│
├── vector_store/
│   └── .gitkeep                     # Persisted vector DB (gitignored)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml           # App + vector DB services together
│
├── .env.example                     # API keys template (OpenAI/Groq/etc.)
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Notes on Key Folders

| Folder / Component | Why it's separated this way |
| --- | --- |
| **`app/` vs `src/`** | `app/` = FastAPI/web layer only; `src/` = all core logic (reusable, testable independent of the API) |
| **`rag/` vs `agent/`** | Keeps your "basic RAG" (v1-v2 from the build plan) cleanly separate from "agentic upgrades" (v4-v7) — mirrors your actual build stages |
| **`rag/crag.py` + `rag/self_rag.py`** | Dedicated files for the CRAG/Self-RAG upgrade stage — easy to point to in interviews |
| **`middleware/guardrails.py`** | Matches your Stage 12 patch (confirmation before destructive changes) |
| **`evaluation/ragas_eval.py`** | Matches your Stage 13 patch (RAGAS evaluation) |
| **`docker/`** | Separate folder keeps deployment config isolated from app logic |
