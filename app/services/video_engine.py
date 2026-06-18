"""
video_engine.py - Assemble a full educational video from a script.

Pipeline:
  1. For each scene: generate image (ai_image) + narration audio (ai_tts).
  2. Apply Ken Burns (pan/zoom) animation to each still image -> a video clip
     whose length matches its narration audio.
  3. Burn captions/subtitles onto clips.
  4. Concatenate all scene clips, mux full narration track.
  5. Export at the requested resolution (720p / 1080p / 1080x1920 vertical).

Everything is done locally with FFmpeg -> 100% free.
"""
import os
import json
import math
import shutil
import subprocess
import tempfile
import textwrap

from . import ai_image, ai_tts, ai_text

RESOLUTIONS = {
    "1280x720":  (1280, 720),
    "1920x1080": (1920, 1080),
    "1080x1920": (1080, 1920),
}


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg error:\nCMD: " + " ".join(cmd) +
            "\nSTDERR:\n" + proc.stderr[-1500:])
    return proc


def _escape_drawtext(text):
    """Escape text for ffmpeg drawtext filter."""
    if not text:
        return ""
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")  # curly apostrophe avoids quoting issues
    text = text.replace("%", "\\%")
    return text


def _font_dir():
    # When packaged by PyInstaller, data is unpacked under sys._MEIPASS.
    base = getattr(__import__("sys"), "_MEIPASS", None)
    if base:
        p = os.path.join(base, "app", "fonts")
        if os.path.isdir(p):
            return p
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

_FONT_DIR = _font_dir()
_RTL_LANGS = {"fa", "ar", "ur", "he", "ps"}

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_RTL = True
except Exception:
    _HAS_RTL = False


def _find_font(language="en"):
    """Pick a font that supports the target language."""
    if language in _RTL_LANGS:
        cands = [
            os.path.join(_FONT_DIR, "Vazirmatn-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        cands = [
            os.path.join(_FONT_DIR, "Latin-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _shape_rtl(text, language="en"):
    """Reshape + bidi-reorder Arabic/Persian text so drawtext renders it
    correctly (joined letters, right-to-left)."""
    if language in _RTL_LANGS and _HAS_RTL and text:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


def _make_scene_clip(image_path, audio_path, out_clip, w, h, caption,
                     fontfile, progress_cb=None, language="en"):
    """Create one animated scene clip (Ken Burns + caption) matching audio length."""
    duration = ai_tts.get_audio_duration(audio_path)
    if duration <= 0.1:
        duration = 4.0
    duration = max(2.0, duration + 0.4)  # small tail padding

    fps = 25
    total_frames = int(duration * fps)

    # Ken Burns zoom-in (1.0 -> 1.10). To keep CPU low on limited hardware we
    # pre-scale the still ONCE to the target frame (cover+crop), then run
    # zoompan at the target size instead of on a giant 2x canvas.
    zoom_end = 1.10
    zinc = (zoom_end - 1.0) / max(1, total_frames)

    # Pre-scale to exactly fill the frame (cover) once.
    scale_cover = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}"
    )

    zoompan = (
        f"zoompan=z='min(zoom+{zinc:.6f},{zoom_end})':"
        f"d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={w}x{h}:fps={fps}"
    )

    vf = scale_cover + "," + zoompan + ",format=yuv420p"

    # Caption overlay (semi-transparent bar at bottom)
    if caption and fontfile:
        # wrap first (on raw text), then shape each line for RTL languages
        lines = textwrap.wrap(caption, width=max(20, int(w / 22))) or [caption]
        shaped_lines = [_escape_drawtext(_shape_rtl(ln, language)) for ln in lines]
        wrapped = "\\n".join(shaped_lines)
        cap = wrapped
        fontsize = max(28, int(h * 0.045))
        drawtext = (
            f",drawbox=x=0:y=ih-{int(h*0.14)}:w=iw:h={int(h*0.14)}:"
            f"color=black@0.45:t=fill,"
            f"drawtext=fontfile='{fontfile}':text='{wrapped}':"
            f"fontcolor=white:fontsize={fontsize}:"
            f"x=(w-text_w)/2:y=h-{int(h*0.11)}:"
            f"shadowcolor=black@0.8:shadowx=2:shadowy=2"
        )
        vf += drawtext

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        # Force consistent GOP so concat with stream-copy is seamless
        "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
        "-shortest",
        out_clip,
    ]
    _run(cmd)
    return out_clip, duration


def _concat_clips(clip_paths, out_path):
    """Concatenate scene clips. All clips share the same codec/params, so we
    use stream-copy (-c copy) which is near-instant (no re-encode)."""
    listfile = out_path + ".concat.txt"
    with open(listfile, "w") as f:
        for c in clip_paths:
            f.write(f"file '{os.path.abspath(c)}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-c", "copy", "-movflags", "+faststart",
        out_path,
    ]
    try:
        _run(cmd)
    except Exception:
        # Fallback: re-encode if copy fails (mismatched params)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
            "-movflags", "+faststart", out_path,
        ]
        _run(cmd)
    try:
        os.unlink(listfile)
    except Exception:
        pass
    return out_path


def build_video(script, resolution="1280x720", language="en", voice=None,
                workdir=None, out_path=None, progress_cb=None, image_model="flux",
                tts_rate="+0%"):
    """
    script: dict {title, scenes:[{narration, image_prompt, caption}]}
    Returns final video path.
    """
    if resolution not in RESOLUTIONS:
        resolution = "1280x720"
    w, h = RESOLUTIONS[resolution]
    fontfile = _find_font(language)

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="vidgen_")
    os.makedirs(workdir, exist_ok=True)

    scenes = script["scenes"]
    n = len(scenes)

    def report(stage, i, msg):
        if progress_cb:
            progress_cb(stage, i, n, msg)

    # ---- Phase 1: download all images + synthesize all narration IN PARALLEL.
    # These are network/IO-bound, so threading gives a big speed-up.
    from concurrent.futures import ThreadPoolExecutor

    img_paths = [os.path.join(workdir, f"img_{i:03d}.jpg") for i in range(n)]
    aud_paths = [os.path.join(workdir, f"aud_{i:03d}.mp3") for i in range(n)]
    done_counter = {"img": 0, "tts": 0}
    lock = __import__("threading").Lock()

    def do_image(i):
        try:
            ai_image.generate_image(scenes[i]["image_prompt"], img_paths[i],
                                    width=w, height=h, model=image_model)
        except Exception:
            _make_placeholder(img_paths[i], w, h,
                              scenes[i].get("caption", "") or script.get("title", ""))
        with lock:
            done_counter["img"] += 1
            report("image", done_counter["img"] - 1,
                   f"Generating images {done_counter['img']}/{n}")

    def do_tts(i):
        try:
            ai_tts.synthesize(scenes[i]["narration"], aud_paths[i],
                              language=language, voice=voice, rate=tts_rate)
        except Exception:
            _make_silence(aud_paths[i], 4.0)
        with lock:
            done_counter["tts"] += 1
            report("tts", done_counter["tts"] - 1,
                   f"Generating narration {done_counter['tts']}/{n}")

    # Limit concurrency to be gentle on the free services + 2-core CPU.
    with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
        list(ex.map(do_image, range(n)))
    with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
        list(ex.map(do_tts, range(n)))

    # ---- Phase 2: render each scene clip (CPU-bound -> keep sequential).
    clips = []
    for i in range(n):
        report("clip", i, f"Rendering scene {i+1}/{n}")
        clip_path = os.path.join(workdir, f"clip_{i:03d}.mp4")
        _make_scene_clip(img_paths[i], aud_paths[i], clip_path, w, h,
                         scenes[i].get("caption", ""), fontfile,
                         language=language)
        clips.append(clip_path)

    report("concat", n, "Combining scenes")
    if out_path is None:
        out_path = os.path.join(workdir, "final.mp4")
    _concat_clips(clips, out_path)
    report("done", n, "Video ready")
    return out_path


def _make_placeholder(path, w, h, text=""):
    fontfile = _find_font()
    vf = f"color=c=0x1a2238:s={w}x{h}"
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", vf, "-frames:v", "1"]
    if text and fontfile:
        t = _escape_drawtext(text[:40])
        cmd[-1:-1] = []
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", vf,
               "-vf", f"drawtext=fontfile='{fontfile}':text='{t}':fontcolor=white:fontsize={int(h*0.06)}:x=(w-text_w)/2:y=(h-text_h)/2",
               "-frames:v", "1", path]
    else:
        cmd.append(path)
    _run(cmd)
    return path


def _make_silence(path, seconds):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
           f"anullsrc=channel_layout=stereo:sample_rate=44100",
           "-t", f"{seconds}", "-c:a", "libmp3lame", path]
    _run(cmd)
    return path
