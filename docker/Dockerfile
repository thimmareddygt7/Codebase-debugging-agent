# ==============================================================================
# Production Dockerfile for Codebase Debugging Agent
# Python 3.11 Slim with Non-Root Security, Persistent Volumes, and Healthcheck
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install minimal OS dependencies required for git, compilation, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch first to prevent sentence-transformers/chromadb
# from pulling the full 2.5GB GPU/CUDA build unnecessarily
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torch>=2.6.0" \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create application directories for persistent state, sessions, and vector store
RUN mkdir -p /app/data /app/data/backups /app/vector_store

# Create a non-privileged user and grant ownership
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose FastAPI application port
EXPOSE 8000

# Health check probe against /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
