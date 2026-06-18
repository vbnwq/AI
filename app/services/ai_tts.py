"""
ai_tts.py - Free Text-to-Speech (no API key)
Primary:  Microsoft Edge TTS (edge-tts) - natural neural voices, multilingual.
Fallback: Google Translate TTS.
Returns spoken audio (mp3) and the measured duration in seconds.
"""
import asyncio
import os
import subprocess
import urllib.parse
import requests

import edge_tts

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Curated good neural voices per language
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "en-male": "en-US-GuyNeural",
    "fa": "fa-IR-FaridNeural",
    "fa-female": "fa-IR-DilaraNeural",
    "ar": "ar-SA-HamedNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "tr": "tr-TR-AhmetNeural",
    "ru": "ru-RU-DmitryNeural",
    "hi": "hi-IN-MadhurNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
}

# Per-language male/female neural voices for explicit gender selection.
GENDER_VOICES = {
    "en": {"male": "en-US-GuyNeural", "female": "en-US-AriaNeural"},
    "fa": {"male": "fa-IR-FaridNeural", "female": "fa-IR-DilaraNeural"},
    "ar": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "tr": {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "zh": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
}


def pick_voice(language="en", voice=None, gender=None):
    """Resolve a concrete neural voice.

    Priority: explicit voice id > gender selection for language > default.
    """
    if voice:
        # treat 'male'/'female' passed in the voice slot as gender too
        if voice in ("male", "female"):
            gender = voice
        else:
            return voice
    if gender in ("male", "female"):
        g = GENDER_VOICES.get(language)
        if g:
            return g[gender]
    return VOICE_MAP.get(language, "en-US-AriaNeural")


async def _edge_save(text, voice, out_path, rate="+0%", pitch="+0Hz"):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(out_path)


def _gtts_fallback(text, out_path, language="en"):
    """Google Translate TTS fallback. Splits long text into <200 char chunks."""
    chunks = []
    words = text.split()
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 180:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)

    part_files = []
    for i, ch in enumerate(chunks):
        part = out_path + f".part{i}.mp3"
        url = ("https://translate.google.com/translate_tts?ie=UTF-8&q="
               + urllib.parse.quote(ch)
               + f"&tl={language}&client=tw-ob&total={len(chunks)}&idx={i}&textlen={len(ch)}")
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and len(r.content) > 500:
            with open(part, "wb") as f:
                f.write(r.content)
            part_files.append(part)
    if not part_files:
        raise RuntimeError("gTTS fallback produced no audio")

    # concat parts
    if len(part_files) == 1:
        os.replace(part_files[0], out_path)
    else:
        listfile = out_path + ".txt"
        with open(listfile, "w") as f:
            for p in part_files:
                f.write(f"file '{os.path.abspath(p)}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", out_path],
            check=True, capture_output=True)
        for p in part_files:
            try: os.unlink(p)
            except Exception: pass
        try: os.unlink(listfile)
        except Exception: pass
    return out_path


def get_audio_duration(path):
    """Return audio duration in seconds using ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def synthesize(text, out_path, language="en", voice=None, rate="+0%",
               pitch="+0Hz", gender=None):
    """
    Convert text to speech mp3 at out_path.
    Returns (out_path, duration_seconds).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for TTS")

    chosen_voice = pick_voice(language, voice, gender)

    # Try edge-tts first
    try:
        asyncio.run(_edge_save(text, chosen_voice, out_path, rate=rate, pitch=pitch))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path, get_audio_duration(out_path)
    except Exception:
        pass

    # Fallback to Google TTS
    _gtts_fallback(text, out_path, language=language)
    return out_path, get_audio_duration(out_path)


def list_voices_sync():
    """Return available edge-tts voices (cached friendly)."""
    try:
        voices = asyncio.run(edge_tts.list_voices())
        return [{"name": v["ShortName"], "gender": v["Gender"],
                 "locale": v["Locale"]} for v in voices]
    except Exception:
        return [{"name": v, "gender": "", "locale": ""} for v in VOICE_MAP.values()]
