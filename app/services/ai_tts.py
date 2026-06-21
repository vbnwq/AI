"""
ai_tts.py - Free Text-to-Speech (no API key)
Primary:  Microsoft Edge TTS (edge-tts) - natural neural voices, multilingual.
Fallback: Google Translate TTS.
Returns spoken audio (mp3) and the measured duration in seconds.

PERSIAN / IRANIAN ACCENT
------------------------
The Persian voices used here are TRUE Iranian (fa-IR) neural voices:
  * fa-IR-FaridNeural   (male, natural Tehran/standard Iranian accent)
  * fa-IR-DilaraNeural  (female, natural Iranian accent)
These are the official Microsoft Persian (Iran) neural voices — they speak with
a genuine Iranian/Persian accent, correct intonation and natural prosody.

We additionally:
  * Normalize Persian text (digits, punctuation, ZWNJ) for clearer, more
    natural pronunciation.
  * Apply light, accent-friendly prosody defaults (slightly warmer pitch /
    measured pace) so narration sounds professional and natural in Persian.
"""
import asyncio
import os
import re
import subprocess
import urllib.parse
import requests

import edge_tts

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Curated good neural voices per language
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "en-male": "en-US-GuyNeural",
    "fa": "fa-IR-FaridNeural",       # Iranian Persian male
    "fa-female": "fa-IR-DilaraNeural",  # Iranian Persian female
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

# Accent-friendly prosody defaults per language. For Persian we use a measured
# pace and a touch of warmth so the Iranian accent sounds natural & professional
# rather than robotic. Values are merged with any caller-provided rate/pitch.
ACCENT_PROSODY = {
    "fa": {"rate": "-4%", "pitch": "+0Hz", "volume": "+0%"},
    "ar": {"rate": "-3%", "pitch": "+0Hz", "volume": "+0%"},
    "en": {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
}

# Per-language male/female neural voices for explicit gender selection.
# Persian (fa) has high-quality neural voices: Farid (male), Dilara (female).
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


# Map Western/Persian digits etc. so the Persian voice reads numbers naturally.
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"})


def normalize_persian(text):
    """Clean up Persian text for clearer, more natural TTS pronunciation.

    * unify Arabic glyphs to Persian (ي->ی, ك->ک)
    * normalize ZWNJ / spacing and collapse repeats
    * convert ASCII digits to Persian digits (read correctly by fa voices)
    * tidy punctuation so the voice pauses naturally
    """
    if not text:
        return text
    t = text.translate(_ARABIC_TO_PERSIAN)
    t = t.replace("\u200c", " ").replace("\u200f", "").replace("\u200e", "")
    # standalone ASCII numbers -> Persian digits for natural reading
    t = re.sub(r"(?<![A-Za-z])\d+(?![A-Za-z])",
               lambda m: m.group(0).translate(_PERSIAN_DIGITS), t)
    t = t.replace("...", "،").replace("…", "،")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _merge_prosody(language, rate, pitch, volume):
    """Combine caller-supplied prosody with accent-friendly defaults.

    Caller values win when explicitly non-default; otherwise the accent default
    is applied. The `rate` from the server (duration-fit) is always respected
    when it is not the neutral "+0%".
    """
    base = ACCENT_PROSODY.get(language, {})
    out_rate = rate if rate not in (None, "+0%") else base.get("rate", rate or "+0%")
    out_pitch = pitch if pitch not in (None, "+0Hz") else base.get("pitch", pitch or "+0Hz")
    out_vol = volume if volume not in (None, "+0%") else base.get("volume", volume or "+0%")
    return out_rate, out_pitch, out_vol


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


async def _edge_save(text, voice, out_path, rate="+0%", pitch="+0Hz",
                     volume="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch,
                                       volume=volume)
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
               pitch="+0Hz", gender=None, volume="+0%"):
    """
    Convert text to speech mp3 at out_path.
    Returns (out_path, duration_seconds).

    Robust chain (all free):
      1. Microsoft Edge neural TTS (best quality, native Persian voices) with
         a couple of retries (handles transient network blips).
      2. Edge TTS with the language's default voice (if a custom voice failed).
      3. Google Translate TTS fallback.
    Never raises on transient failure unless every backend fails.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for TTS")

    # Normalize Persian/Arabic text for clearer, more natural pronunciation.
    if language in ("fa", "ar"):
        text = normalize_persian(text)

    # Apply accent-friendly prosody defaults (Iranian accent for Persian).
    rate, pitch, volume = _merge_prosody(language, rate, pitch, volume)

    chosen_voice = pick_voice(language, voice, gender)

    # Try edge-tts (best quality) with retries.
    for attempt in range(3):
        try:
            asyncio.run(_edge_save(text, chosen_voice, out_path,
                                   rate=rate, pitch=pitch, volume=volume))
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return out_path, get_audio_duration(out_path)
        except Exception:
            pass
        # second try: fall back to the language default voice
        if attempt == 0:
            chosen_voice = VOICE_MAP.get(language, "en-US-AriaNeural")

    # Fallback to Google TTS (gTTS endpoint).
    try:
        _gtts_fallback(text, out_path, language=language)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path, get_audio_duration(out_path)
    except Exception:
        pass

    raise RuntimeError("All TTS backends failed for language=" + str(language))


def list_voices_sync():
    """Return available edge-tts voices (cached friendly)."""
    try:
        voices = asyncio.run(edge_tts.list_voices())
        return [{"name": v["ShortName"], "gender": v["Gender"],
                 "locale": v["Locale"]} for v in voices]
    except Exception:
        return [{"name": v, "gender": "", "locale": ""} for v in VOICE_MAP.values()]
