"""FastAPI service: JSON and Server-Sent Events endpoints for the evidence agent."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from biomed_agent.agent.graph import EvidenceAgent
from biomed_agent.config import get_settings, load_env
from biomed_agent.factory import build_agent
from biomed_agent.llm import LLMError
from biomed_agent.schemas import AnswerResponse
from biomed_agent.tracing import flush

log = logging.getLogger("biomed_agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_env()
    agent, http = build_agent()
    app.state.agent = agent
    yield
    await http.aclose()
    flush()


app = FastAPI(title="Biomed Evidence Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _agent(request: Request) -> EvidenceAgent:
    return request.app.state.agent


@app.get("/api/health")
async def health() -> dict:
    s = get_settings()
    return {"status": "ok", "model": s.llm_model, "embedding_model": s.embedding_model}


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(body: AskRequest, request: Request) -> AnswerResponse:
    try:
        return await _agent(request).ask(body.question)
    except anthropic.APIStatusError as e:
        log.exception("Claude API error")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e.message}") from e
    except LLMError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> EventSourceResponse:
    """Stream graph progress as SSE: one `progress` event per node, then `result` (or `error`)."""
    agent = _agent(request)

    async def events():
        try:
            async for event in agent.stream(body.question):
                kind = "result" if event["node"] == "done" else "progress"
                payload = event.get("result", event)
                yield {"event": kind, "data": json.dumps(payload, default=str)}
        except (anthropic.APIError, LLMError) as e:
            log.exception("agent run failed")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return EventSourceResponse(events())


# Serve the built React app when present (Docker image / `npm run build`).
_web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")
