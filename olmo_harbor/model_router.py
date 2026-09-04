"""Small authenticated reverse proxy for three colocated vLLM engines."""

from __future__ import annotations

import gzip
import hmac
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.gzip import GZipMiddleware
from starlette.background import BackgroundTask
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODEL_PORTS: dict[str, int] = {
    str(model): int(port)
    for model, port in json.loads(os.environ["OLMO_MODEL_PORT_MAP"]).items()
}
TOKEN = os.environ["VLLM_API_KEY"]
GZIP_CHUNK_BYTES = 64 * 1024
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)


def authorize(request: Request) -> None:
    expected = f"Bearer {TOKEN}"
    observed = request.headers.get("authorization", "")
    if not hmac.compare_digest(observed, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def forwarded_headers(request: Request) -> dict[str, str]:
    blocked = {"authorization", "host", "content-length", "connection"}
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked
    }
    # vLLM reads VLLM_API_KEY and authenticates its own OpenAI-compatible API.
    # Replace, rather than blindly forwarding, the client credential.
    headers["authorization"] = f"Bearer {TOKEN}"
    return headers


def response_headers(response: httpx.Response) -> dict[str, str]:
    blocked = {"content-length", "transfer-encoding", "content-encoding", "connection"}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in blocked
    }


async def chunk_bytes(content: bytes) -> AsyncIterator[bytes]:
    for offset in range(0, len(content), GZIP_CHUNK_BYTES):
        yield content[offset : offset + GZIP_CHUNK_BYTES]


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        statuses = {}
        for model, port in MODEL_PORTS.items():
            response = await client.get(f"http://127.0.0.1:{port}/health")
            statuses[model] = response.status_code
    if any(status != 200 for status in statuses.values()):
        raise HTTPException(status_code=503, detail=statuses)
    return {"status": "ok", "models": statuses}


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    authorize(request)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "allenai"}
                for model in MODEL_PORTS
            ],
        }
    )


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def proxy(request: Request) -> Response:
    authorize(request)
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="request body must be JSON"
        ) from exc
    model = payload.get("model")
    if model not in MODEL_PORTS:
        raise HTTPException(status_code=404, detail=f"unknown model {model!r}")
    url = f"http://127.0.0.1:{MODEL_PORTS[model]}{request.url.path}"
    client = httpx.AsyncClient(timeout=None)
    upstream_request = client.build_request(
        "POST", url, content=body, headers=forwarded_headers(request)
    )
    upstream = await client.send(upstream_request, stream=bool(payload.get("stream")))
    headers = response_headers(upstream)
    if payload.get("stream"):
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(client.aclose),
        )
    content = await upstream.aread()
    await upstream.aclose()
    await client.aclose()
    # Modal may normalize Accept-Encoding before this app sees the request.
    # Compress every non-streaming JSON response and send it without a fixed
    # Content-Length. This preserves selected-token telemetry while preventing
    # edge truncation from surfacing as client IncompleteRead.
    content = gzip.compress(content, compresslevel=6)
    headers["content-encoding"] = "gzip"
    headers["cache-control"] = "no-transform"
    return StreamingResponse(
        chunk_bytes(content),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
