"""Chat backend: one probed model resident on the GPU at a time, intervention chosen per request."""
import json
import os
import threading
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from ..config import RUNS_DIR, list_models, model_gb
from ..hooks import add_hooks
from ..model import ProbedModel

app = FastAPI(title="refusal-probe")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()  # serialises model loading and generation (hooks are per-call, GPU is shared)
_state = {"pm": None, "alias": None, "layer": None, "direction": None}


def _run_dir(model_id: str) -> str:
    return os.path.join(RUNS_DIR, os.path.basename(model_id))


def _probed(model_id: str) -> bool:
    return os.path.exists(os.path.join(_run_dir(model_id), "direction.pt"))


def _read_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _gpu_memory():
    """Free/total GiB of this worker's GPU slice via NVML (no CUDA context, so it costs no GPU memory)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        uuid = os.environ.get("GPU_UUID") or pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(0))
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        m = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByUUID(uuid.encode()))
        return m.free / 2**30, m.total / 2**30, uuid
    except Exception:
        return 0.0, 0.0, os.environ.get("GPU_UUID", "?")


@app.get("/api/health")
def health():
    free, total, uuid = _gpu_memory()
    alias = _state["alias"]
    loaded_id = next((m for m in list_models() if os.path.basename(m) == alias), None)
    return dict(gpu=uuid, free_gb=free, total_gb=total, loaded=alias, loaded_gb=model_gb(loaded_id) if loaded_id else 0.0,
                busy=_lock.locked())


class _Cancel(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kw):
        return self.event.is_set()


@app.get("/api/models")
def models():
    out = []
    for mid in list_models():
        d = _run_dir(mid)
        meta = _read_json(f"{d}/direction_metadata.json")
        out.append(dict(
            id=mid, probed=_probed(mid), loaded=_state["alias"] == os.path.basename(mid),
            layer=meta and meta["layer"], pos=meta and meta["pos"], n_layers=meta and meta["n_layers"],
            has_safety=os.path.exists(f"{d}/safety/summary.json"), has_capability=os.path.exists(f"{d}/capability/lm_eval.json"),
        ))
    return out


@app.get("/api/results")
def results():
    """Aggregated benchmark numbers for every probed model."""
    rows = {}
    for mid in list_models():
        d = _run_dir(mid)
        if not _probed(mid):
            continue
        rows[mid] = dict(
            meta=_read_json(f"{d}/direction_metadata.json"),
            safety=_read_json(f"{d}/safety/summary.json"),
            loss=_read_json(f"{d}/loss_evals/ce_loss.json"),
            capability={k: v["headline"] for k, v in (_read_json(f"{d}/capability/lm_eval.json") or {}).items()},
        )
    return rows


PLOTS = {"ablation_scores", "actadd_scores", "kl_div_scores"}


@app.get("/api/plot/{alias}/{name}")
def plot(alias: str, name: str):
    from fastapi.responses import FileResponse

    path = os.path.join(RUNS_DIR, os.path.basename(alias), "select_direction", f"{name}.png")
    if name not in PLOTS or not os.path.exists(path):
        raise HTTPException(404, "no such plot")
    return FileResponse(path, media_type="image/png")


def _load(model_id: str):
    if _state["alias"] == os.path.basename(model_id):
        return
    if model_id not in list_models():
        raise HTTPException(404, "unknown model")
    if not _probed(model_id):
        raise HTTPException(409, "model has no refusal direction yet; run the pipeline first")
    if _state["pm"] is not None:
        _state["pm"].unload()
        _state.update(pm=None, alias=None)
    pm = ProbedModel(model_id)
    meta = json.load(open(f"{_run_dir(model_id)}/direction_metadata.json"))
    pm.refusal_toks = meta["refusal_toks"]
    _state.update(pm=pm, alias=os.path.basename(model_id), layer=meta["layer"],
                  direction=torch.load(f"{_run_dir(model_id)}/direction.pt"))


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str
    messages: List[Msg]
    mode: str = "baseline"  # baseline | ablate | steer
    strength: float = 1.0  # steer: multiple of the raw mean-diff vector (+ induces refusal, - suppresses it)
    max_new_tokens: int = 512
    temperature: float = 0.0
    system: Optional[str] = None


@app.post("/api/chat")
def chat(req: ChatReq):
    if req.mode not in ("baseline", "ablate", "steer"):
        raise HTTPException(400, "bad mode")
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "last message must be from the user")

    def stream():
        with _lock:
            try:
                _load(req.model)
            except HTTPException as e:
                yield f"data: {json.dumps(dict(error=e.detail))}\n\n"
                return
            pm, d = _state["pm"], _state["direction"]
            if req.mode == "ablate":
                pre, post = pm.ablation_hooks(d)
            elif req.mode == "steer":
                pre, post = pm.addition_hooks(_state["layer"], d, req.strength)
            else:
                pre, post = [], []

            history = [m.model_dump() for m in req.messages[:-1]]
            prompt = pm.format_prompt(req.messages[-1].content, history=history, system=req.system)
            enc = pm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(pm.device)
            streamer = TextIteratorStreamer(pm.tokenizer, skip_prompt=True, skip_special_tokens=True)
            sample = req.temperature > 0
            cfg = GenerationConfig(max_new_tokens=req.max_new_tokens, do_sample=sample, pad_token_id=pm.tokenizer.pad_token_id,
                                   temperature=req.temperature if sample else None, top_p=0.95 if sample else None, top_k=None)

            err = []
            cancel = threading.Event()
            cfg_stop = StoppingCriteriaList([_Cancel(cancel)])

            def run():
                try:
                    with torch.no_grad(), add_hooks(pre, post):
                        pm.model.generate(**enc, generation_config=cfg, streamer=streamer, stopping_criteria=cfg_stop)
                except Exception as e:  # surface OOM etc. to the client instead of hanging the stream
                    err.append(str(e))
                    streamer.end()

            t = threading.Thread(target=run)
            t.start()
            try:
                for piece in streamer:
                    yield f"data: {json.dumps(dict(token=piece))}\n\n"
            finally:
                cancel.set()  # client went away or finished: stop generating and free the GPU
            t.join()
            yield f"data: {json.dumps(dict(error=err[0]) if err else dict(done=True))}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/unload")
def unload():
    with _lock:
        if _state["pm"] is not None:
            _state["pm"].unload()
        _state.update(pm=None, alias=None)
    return {"ok": True}


# Serve the built React app (npm run build in web/) from the same origin when present.
_dist = os.path.join(os.path.dirname(RUNS_DIR), "web", "dist")
if os.path.isdir(_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
