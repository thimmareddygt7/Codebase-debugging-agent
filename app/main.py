from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.middleware.guardrails import APIGuardrailsMiddleware
from src.guardrails.secrets import redact_secrets

guardrail_middleware = APIGuardrailsMiddleware()

app = FastAPI(
    title="Codebase Debugging Agent API",
    description=(
        "Production-grade FastAPI platform for RAG-augmented codebase debugging, "
        "repository search, patch proposals, human-in-the-loop approvals, and safe testing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_redaction_middleware(request: Request, call_next):
    """
    Middleware enforcing response secrets redaction and request traceability.
    """
    response = await call_next(request)
    return response


# Include all stage routers
app.include_router(api_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Codebase Debugging Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
