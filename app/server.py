"""
server.py - FastAPI backend for the Free AI Studio.

Endpoints:
  GET  /                     -> Web UI
  GET  /api/health           -> health + which AI sources are live
  POST /api/image            -> generate image (returns job, polled)  [sync small]
  POST /api/image/quick      -> generate single image synchronously, returns URL
  POST /api/video            -> start video generation job (async)
  GET  /api/job/{id}         -> job status + progress
  GET  /api/voices           -> list TTS voices
  GET  /output/...           -> serve generated files
  GET  /api/download/{id}    -> download finished file

100% free pipeline (Pollinations + Edge-TTS + FFmpeg). No API key required.
"""
import os
import uuid
import time
import threading
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.services import ai_text, ai_image, ai_tts, video_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# When packaged as exe, write outputs next to the executable (AISTUDIO_HOME).
OUTPUT_DIR = os.path.join(os.environ.get("AISTUDIO_HOME", BASE_DIR), "output")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
VID_DIR = os.path.join(OUTPUT_DIR, "videos")
TMP_DIR = os.path.join(OUTPUT_DIR, "temp")
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
for d in (IMG_DIR, VID_DIR, TMP_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Free AI Studio")

# In-memory job store
JOBS = {}
JOBS_LOCK = threading.Lock()


# ---------- Models ----------
class ImageReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    model: str = "flux"
    count: int = 1


class VideoReq(BaseModel):
    topic: str
    duration: int = 60
    resolution: str = "1280x720"
    language: str = "en"
    voice: str | None = None
    gender: str | None = None   # "male" | "female" | None
    image_model: str = "flux"


# ---------- Helpers ----------
def new_job(kind):
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid, "kind": kind, "status": "queued",
            "progress": 0, "stage": "queued", "message": "Queued",
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
            "image": "Pollinations (free, no key)",
            "text": "Pollinations Text (free, no key)",
            "tts": "Microsoft Edge TTS + Google TTS fallback (free)",
            "video": "FFmpeg local render (free)",
        },
        "resolutions": list(video_engine.RESOLUTIONS.keys()),
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
        update_job(jid, status="running", stage="image", message="Generating image(s)")
        results = []
        count = max(1, min(req.count, 4))
        for i in range(count):
            fname = f"img_{jid}_{i}.jpg"
            fpath = os.path.join(IMG_DIR, fname)
            ai_image.generate_image(
                req.prompt, fpath,
                width=req.width, height=req.height, model=req.model)
            results.append({"url": f"/output/images/{fname}", "file": fname})
            update_job(jid, progress=int((i + 1) / count * 100),
                       message=f"Image {i+1}/{count} done", results=results)
        update_job(jid, status="done", progress=100,
                   stage="done", message="Done", results=results)
    except Exception as e:
        update_job(jid, status="error", error=str(e),
                   message="Failed: " + str(e))
        traceback.print_exc()


@app.post("/api/image")
def gen_image(req: ImageReq):
    jid = new_job("image")
    t = threading.Thread(target=_run_image_job, args=(jid, req), daemon=True)
    t.start()
    return {"job_id": jid}


# ----- VIDEO -----
def _run_video_job(jid, req: VideoReq):
    try:
        update_job(jid, status="running", stage="script",
                   message="Writing script with AI...", progress=3)

        script = ai_text.generate_video_script(
            req.topic, req.duration, language=req.language)
        n = len(script["scenes"])
        update_job(jid, message=f"Script ready: {n} scenes", progress=8,
                   results=[{"title": script.get("title", req.topic),
                             "scenes": n}])

        # Adapt TTS speed so total duration ~ requested duration.
        total_words = sum(len(s["narration"].split()) for s in script["scenes"])
        est_sec = total_words / 2.6  # ~2.6 words/sec at normal rate
        rate = "+0%"
        if req.duration and est_sec > 0:
            ratio = est_sec / req.duration
            # speed up if narration is too long, slow down if too short
            pct = int(max(-25, min(45, (ratio - 1.0) * 100)))
            rate = ("+" if pct >= 0 else "") + str(pct) + "%"

        workdir = os.path.join(TMP_DIR, jid)
        os.makedirs(workdir, exist_ok=True)
        out_name = f"video_{jid}.mp4"
        out_path = os.path.join(VID_DIR, out_name)

        def progress_cb(stage, i, total, msg):
            # map stage to overall progress (script=8 .. done=100)
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
            tts_rate=rate, gender=req.gender)

        update_job(jid, status="done", progress=100, stage="done",
                   message="Video ready!",
                   results=[{
                       "title": script.get("title", req.topic),
                       "url": f"/output/videos/{out_name}",
                       "file": out_name,
                       "scenes": n,
                       "script": script,
                   }])
    except Exception as e:
        update_job(jid, status="error", error=str(e),
                   message="Failed: " + str(e))
        traceback.print_exc()


@app.post("/api/video")
def gen_video(req: VideoReq):
    jid = new_job("video")
    t = threading.Thread(target=_run_video_job, args=(jid, req), daemon=True)
    t.start()
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    job = get_job(jid)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


@app.get("/api/download/{jid}")
def download(jid: str):
    job = get_job(jid)
    if not job or job["status"] != "done" or not job["results"]:
        return JSONResponse({"error": "not ready"}, status_code=404)
    res = job["results"][-1]
    url = res.get("url", "")
    fpath = os.path.join(BASE_DIR, url.lstrip("/"))
    if os.path.exists(fpath):
        return FileResponse(fpath, filename=os.path.basename(fpath))
    return JSONResponse({"error": "file missing"}, status_code=404)


# serve outputs + static
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
