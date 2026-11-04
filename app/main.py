import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router, public_router
from app.security import require_api_key
from src.auth import parse_api_keys
from src.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

# CORS: explicit origins only (configure CORS_ALLOW_ORIGINS). Auth uses the
# X-API-Key header, not cookies, so credentials are not allowed cross-origin.
_allowed_origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.on_event("startup")
def _warn_if_insecure() -> None:
    if not parse_api_keys(settings.API_KEYS):
        if settings.ALLOW_INSECURE_NO_AUTH:
            logger.warning("AGENT_API_KEYS is not set and ALLOW_INSECURE_NO_AUTH=true: API is UNAUTHENTICATED.")
        else:
            logger.error("AGENT_API_KEYS is not set: all protected endpoints will return 503 until it is configured.")
    if settings.ENABLE_COMMAND_EXECUTION:
        logger.warning("ENABLE_COMMAND_EXECUTION=true: test/git commands can be executed. Run inside a sandboxed container.")


# Health probes are public; everything else requires a valid X-API-Key.
app.include_router(public_router)
app.include_router(api_router, dependencies=[Depends(require_api_key)])
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


@app.get("/", tags=["Root"])
def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
