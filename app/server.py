"""
server.py - FastAPI backend for the Free AI Studio.

Endpoints:
  GET  /                      -> Web UI
  GET  /favicon.ico
  GET  /api/health            -> health + which AI sources are live
  GET  /api/voices            -> list TTS voices
  POST /api/image             -> start image job (async, polled)
  POST /api/video             -> start video generation job (async)
  GET  /api/job/{id}          -> job status + progress
  GET  /api/history           -> list saved history entries (persisted)
  DELETE /api/history/{id}     -> delete a history entry (and its files)
  DELETE /api/history          -> clear all history
  GET  /api/download?path=...  -> force-download a generated file
  GET  /output/...            -> serve generated files (inline view)

100% free pipeline (Pollinations + Edge-TTS + FFmpeg). No API key required.
Full Persian support with an automatic Persian->English prompt translation layer.
"""
import os
import json
import uuid
import time
import threading
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.services import ai_text, ai_image, ai_tts, video_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_HOME = os.environ.get("AISTUDIO_HOME", BASE_DIR)
OUTPUT_DIR = os.path.join(APP_HOME, "output")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
VID_DIR = os.path.join(OUTPUT_DIR, "videos")
TMP_DIR = os.path.join(OUTPUT_DIR, "temp")
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")
for d in (IMG_DIR, VID_DIR, TMP_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Free AI Studio")

# In-memory job store
JOBS = {}
JOBS_LOCK = threading.Lock()

# Persisted history
HISTORY_LOCK = threading.Lock()


# ---------- History persistence ----------
def _load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(items):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


def add_history(entry):
    with HISTORY_LOCK:
        items = _load_history()
        items.insert(0, entry)
        items = items[:200]  # cap
        _save_history(items)


def delete_history(hid):
    with HISTORY_LOCK:
        items = _load_history()
        kept, removed = [], None
        for it in items:
            if it.get("id") == hid:
                removed = it
            else:
                kept.append(it)
        _save_history(kept)
    # delete files on disk
    if removed:
        for url in removed.get("files", []):
            _safe_delete_url(url)
    return removed is not None


def clear_history():
    with HISTORY_LOCK:
        items = _load_history()
        _save_history([])
    for it in items:
        for url in it.get("files", []):
            _safe_delete_url(url)
    return len(items)


def _safe_delete_url(url):
    try:
        rel = url.lstrip("/")
        fpath = os.path.join(APP_HOME, rel)
        # safety: only allow deletes inside OUTPUT_DIR
        if os.path.commonpath([os.path.abspath(fpath), OUTPUT_DIR]) == OUTPUT_DIR:
            if os.path.exists(fpath):
                os.unlink(fpath)
    except Exception:
        pass


# ---------- Models ----------
class ImageReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    model: str = "flux"
    count: int = 1
    quality: str = "ultra"


class VideoReq(BaseModel):
    topic: str
    duration: int = 60
    resolution: str = "1280x720"
    language: str = "fa"
    voice: str | None = None
    gender: str | None = None
    image_model: str = "flux"
    quality: str = "balanced"
    template: str = "auto"


# ---------- Job helpers ----------
def new_job(kind):
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid, "kind": kind, "status": "queued",
            "progress": 0, "stage": "queued", "message": "در صف",
            "results": [], "error": None, "created": time.time(),
        }
    return jid


def update_job(jid, **kw):
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


def get_job(jid):
    with JOBS_LOCK:
        return dict(JOBS.get(jid)) if jid in JOBS else None


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def index():
    idx = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Free AI Studio</h1><p>UI not found.</p>")


@app.get("/favicon.ico")
def favicon():
    ico = os.path.join(STATIC_DIR, "icon.ico")
    if os.path.exists(ico):
        return FileResponse(ico)
    return JSONResponse({}, status_code=204)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sources": {
            "image": ("Multi-AI / multi-API image pipeline "
                      "(multi-provider gen + crisp detail enhance + anatomy + "
                      "best-of QA) — sharp photorealistic faces, eyes & hands"),
            "research": "Free Deep-Research + Fact-Check AI agents (Persian)",
            "text": "Pollinations LLM (free) + built-in Persian writer",
            "translate": "Free fa->en translation layer (Google + LLM + offline)",
            "tts": ("Microsoft Edge Iranian Persian Neural TTS — emotional, "
                    "expressive per-sentence prosody + Google fallback"),
            "video": ("Multi-AI supervisor + professional multi-slide templates "
                      "with smooth cross-dissolve transitions (FFmpeg+PIL)"),
        },
        "resolutions": list(video_engine.RESOLUTIONS.keys()),
        "qualities": list(video_engine.QUALITY_PRESETS.keys()),
        "templates": ["auto"] + list(video_engine.TEMPLATES.keys()),
    }


@app.get("/api/voices")
def voices():
    try:
        vs = ai_tts.list_voices_sync()
        return {"voices": vs, "recommended": ai_tts.VOICE_MAP}
    except Exception as e:
        return {"voices": [], "recommended": ai_tts.VOICE_MAP, "error": str(e)}


# ----- IMAGE -----
def _run_image_job(jid, req: ImageReq):
    try:
        update_job(jid, status="running", stage="image",
                   message="در حال ساخت تصویر…")
        results = []
        count = max(1, min(req.count, 4))
        for i in range(count):
            fname = f"img_{jid}_{i}.jpg"
            fpath = os.path.join(IMG_DIR, fname)
            ai_image.generate_image(
                req.prompt, fpath, width=req.width, height=req.height,
                model=req.model, quality=req.quality)
            results.append({"url": f"/output/images/{fname}", "file": fname})
            update_job(jid, progress=int((i + 1) / count * 100),
                       message=f"تصویر {i+1}/{count} ساخته شد", results=results)
        update_job(jid, status="done", progress=100, stage="done",
                   message="انجام شد", results=results)
        add_history({
            "id": jid, "kind": "image", "title": req.prompt[:80],
            "prompt": req.prompt, "created": time.time(),
            "size": f"{req.width}x{req.height}", "quality": req.quality,
            "files": [r["url"] for r in results],
            "items": results,
        })
    except Exception as e:
        update_job(jid, status="error", error=str(e), message="خطا: " + str(e))
        traceback.print_exc()


@app.post("/api/image")
def gen_image(req: ImageReq):
    jid = new_job("image")
    threading.Thread(target=_run_image_job, args=(jid, req), daemon=True).start()
    return {"job_id": jid}


# ----- VIDEO -----
def _run_video_job(jid, req: VideoReq):
    try:
        update_job(jid, status="running", stage="script",
                   message="تیم هوش مصنوعی در حال پژوهش و نگارش سناریو…",
                   progress=3)

        # Multi-AI supervisor: research -> fact-check -> scriptwriting.
        def script_progress(msg):
            update_job(jid, stage="script", message=msg, progress=5)

        script = ai_text.generate_video_script(
            req.topic, req.duration, language=req.language,
            progress_cb=script_progress)
        n = len(script["scenes"])
        update_job(jid, message=f"سناریو آماده شد: {n} صحنه", progress=8,
                   results=[{"title": script.get("title", req.topic), "scenes": n}])

        # Adapt TTS speed so total duration ~ requested duration.
        total_words = sum(len(s["narration"].split()) for s in script["scenes"])
        est_sec = total_words / 2.4
        rate = "+0%"
        if req.duration and est_sec > 0:
            ratio = est_sec / req.duration
            pct = int(max(-25, min(45, (ratio - 1.0) * 100)))
            rate = ("+" if pct >= 0 else "") + str(pct) + "%"

        workdir = os.path.join(TMP_DIR, jid)
        os.makedirs(workdir, exist_ok=True)
        out_name = f"video_{jid}.mp4"
        out_path = os.path.join(VID_DIR, out_name)

        def progress_cb(stage, i, total, msg):
            per = 90.0 / max(1, total)
            base = 8
            if stage == "image":
                p = base + per * i
            elif stage == "tts":
                p = base + per * (i + 0.33)
            elif stage == "clip":
                p = base + per * (i + 0.66)
            elif stage == "concat":
                p = 96
            elif stage == "done":
                p = 99
            else:
                p = base
            update_job(jid, stage=stage, message=msg, progress=int(p))

        video_engine.build_video(
            script, resolution=req.resolution, language=req.language,
            voice=req.voice, workdir=workdir, out_path=out_path,
            progress_cb=progress_cb, image_model=req.image_model,
            tts_rate=rate, gender=req.gender, quality=req.quality,
            template=req.template)

        result = {
            "title": script.get("title", req.topic),
            "url": f"/output/videos/{out_name}",
            "file": out_name, "scenes": n, "script": script,
        }
        update_job(jid, status="done", progress=100, stage="done",
                   message="ویدیو آماده شد!", results=[result])
        add_history({
            "id": jid, "kind": "video", "title": script.get("title", req.topic),
            "prompt": req.topic, "created": time.time(),
            "size": req.resolution, "quality": req.quality, "scenes": n,
            "language": req.language, "files": [result["url"]],
            "items": [result],
        })
        # cleanup temp working dir to save space
        try:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
    except Exception as e:
        update_job(jid, status="error", error=str(e), message="خطا: " + str(e))
        traceback.print_exc()


@app.post("/api/video")
def gen_video(req: VideoReq):
    jid = new_job("video")
    threading.Thread(target=_run_video_job, args=(jid, req), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    job = get_job(jid)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


# ----- HISTORY -----
@app.get("/api/history")
def history():
    return {"items": _load_history()}


@app.delete("/api/history/{hid}")
def history_delete(hid: str):
    ok = delete_history(hid)
    return {"deleted": ok}


@app.delete("/api/history")
def history_clear():
    n = clear_history()
    return {"cleared": n}


# ----- DOWNLOAD (force attachment) -----
@app.get("/api/download")
def download(path: str):
    rel = path.lstrip("/")
    fpath = os.path.join(APP_HOME, rel)
    try:
        if os.path.commonpath([os.path.abspath(fpath), OUTPUT_DIR]) != OUTPUT_DIR:
            return JSONResponse({"error": "forbidden"}, status_code=403)
    except Exception:
        return JSONResponse({"error": "bad path"}, status_code=400)
    if os.path.exists(fpath):
        return FileResponse(fpath, filename=os.path.basename(fpath),
                            media_type="application/octet-stream")
    return JSONResponse({"error": "file missing"}, status_code=404)


# serve outputs + static
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
