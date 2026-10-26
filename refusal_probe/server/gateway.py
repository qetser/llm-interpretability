"""Load balancer in front of one worker per GPU slice.

Each worker (refusal_probe.server.app) keeps one model resident. The gateway sends a chat request to a worker that
already has the model loaded and is idle; otherwise to an idle worker with enough free GPU memory (preferring one
with nothing loaded, so warm models are not evicted); otherwise it queues on the least-loaded candidate.
"""
import asyncio
import json
import os
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..config import ROOT, list_models, model_gb

WORKERS = [w.strip() for w in os.environ.get("WORKERS", "http://127.0.0.1:8001").split(",") if w.strip()]
client = httpx.AsyncClient(timeout=None)
inflight = defaultdict(int)

app = FastAPI(title="refusal-probe-gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _health(w):
    try:
        r = await client.get(f"{w}/api/health", timeout=3)
        return w, r.json()
    except Exception:
        return w, None


async def _all_health():
    return [(w, h) for w, h in await asyncio.gather(*[_health(w) for w in WORKERS]) if h]


def _load(w, h):
    return inflight[w] + (1 if h["busy"] else 0)


async def pick(model_id: str):
    alias, need = os.path.basename(model_id), model_gb(model_id)
    hs = await _all_health()  # no await below: choosing and counting must be atomic w.r.t. other requests
    warm = [x for x in hs if x[1]["loaded"] == alias]
    cold = [x for x in hs if x[1]["loaded"] != alias and x[1]["free_gb"] + x[1]["loaded_gb"] >= need]
    choice = None
    idle_warm = [x for x in warm if _load(*x) == 0]
    idle_cold = sorted((x for x in cold if _load(*x) == 0), key=lambda x: (x[1]["loaded"] is not None, -x[1]["free_gb"]))
    if idle_warm:
        choice = idle_warm[0]
    elif idle_cold:
        choice = idle_cold[0]
    elif warm or cold:
        choice = min(warm + cold, key=lambda x: _load(*x))
    if choice:
        inflight[choice[0]] += 1
        return choice[0]
    return None


def _sse(**kw):
    return f"data: {json.dumps(kw)}\n\n".encode()


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.body()
    try:
        model = json.loads(body)["model"]
    except Exception:
        raise HTTPException(400, "bad request")
    w = await pick(model)

    async def gen():
        if w is None:
            yield _sse(error=f"No GPU has the ~{model_gb(model):.1f} GB free that {os.path.basename(model)} needs right now. Try again shortly or pick a smaller model.")
            return
        try:
            async with client.stream("POST", f"{w}/api/chat", content=body, headers={"content-type": "application/json"}) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        except Exception as e:
            yield _sse(error=f"worker failed: {e}")
        finally:
            inflight[w] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.get("/api/workers")
async def workers():
    hs = dict(await asyncio.gather(*[_health(w) for w in WORKERS]))
    return [dict(url=w, up=h is not None, queued=inflight[w], **(h or {})) for w, h in hs.items()]


@app.get("/api/models")
async def models():
    hs = await _all_health()
    if not hs:
        raise HTTPException(503, "no workers are up")
    r = await client.get(f"{hs[0][0]}/api/models")
    rows = r.json()
    loaded = {h["loaded"] for _, h in hs if h["loaded"]}
    for m in rows:
        m["loaded"] = os.path.basename(m["id"]) in loaded
    return rows


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    """results, plots, unload: served from disk (shared by all workers) so any live worker will do."""
    hs = await _all_health()
    if not hs:
        raise HTTPException(503, "no workers are up")
    r = await client.request(request.method, f"{hs[0][0]}/api/{path}", content=await request.body(),
                             headers={"content-type": request.headers.get("content-type", "application/json")})
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


_dist = os.path.join(ROOT, "web", "dist")
if os.path.isdir(_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
