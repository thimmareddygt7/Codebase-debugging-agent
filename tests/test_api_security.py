"""
HTTP-level tests of the security dependencies using a tiny FastAPI app
(does NOT import the agent, vector store, or any model).
Skipped automatically if fastapi/httpx are not installed.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.security as security  # noqa: E402

KEY = "k" * 32


def _client(monkeypatch, **overrides):
    cfg = SimpleNamespace(API_KEYS="", ALLOW_INSECURE_NO_AUTH=False, ENABLE_COMMAND_EXECUTION=False)
    cfg.__dict__.update(overrides)
    monkeypatch.setattr(security, "settings", cfg)

    api = FastAPI()

    @api.get("/protected", dependencies=[Depends(security.require_api_key)])
    def protected():
        return {"ok": True}

    @api.post("/run", dependencies=[Depends(security.require_command_execution_enabled)])
    def run():
        return {"ran": True}

    return TestClient(api)


def test_fail_closed_when_no_keys_configured(monkeypatch):
    assert _client(monkeypatch).get("/protected").status_code == 503


def test_insecure_mode_is_explicit_opt_in(monkeypatch):
    client = _client(monkeypatch, ALLOW_INSECURE_NO_AUTH=True)
    assert client.get("/protected").status_code == 200


def test_missing_and_wrong_key_get_401(monkeypatch):
    client = _client(monkeypatch, API_KEYS=KEY)
    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "wrong" * 8}).status_code == 401


def test_correct_key_is_accepted(monkeypatch):
    client = _client(monkeypatch, API_KEYS=KEY)
    assert client.get("/protected", headers={"X-API-Key": KEY}).status_code == 200


def test_weak_configured_key_does_not_enable_access(monkeypatch):
    client = _client(monkeypatch, API_KEYS="short")
    assert client.get("/protected", headers={"X-API-Key": "short"}).status_code == 503


def test_command_execution_disabled_by_default(monkeypatch):
    assert _client(monkeypatch).post("/run").status_code == 403


def test_command_execution_when_enabled(monkeypatch):
    assert _client(monkeypatch, ENABLE_COMMAND_EXECUTION=True).post("/run").status_code == 200
