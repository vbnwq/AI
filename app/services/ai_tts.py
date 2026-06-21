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
# A slightly raised pitch + warm volume makes the Iranian narrator sound more
# engaged and human (less flat/robotic).
ACCENT_PROSODY = {
    "fa": {"rate": "-3%", "pitch": "+2Hz", "volume": "+8%"},
    "ar": {"rate": "-3%", "pitch": "+0Hz", "volume": "+4%"},
    "en": {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
}

# Emotional prosody profiles. Each sentence in the narration is classified by
# its punctuation / content and rendered with its own expressive prosody so the
# voiceover has natural emotional inflection, rises and falls — like a real
# human narrator instead of a flat, robotic reading.
#   delta_rate / delta_pitch are RELATIVE nudges applied on top of the base
#   accent prosody; pause_ms is the silent gap inserted AFTER the sentence.
EMOTION_PROFILES = {
    "neutral":     {"d_rate": 0,  "d_pitch": 0,  "pause_ms": 260},
    "excited":     {"d_rate": 6,  "d_pitch": 14, "pause_ms": 240},   # ! emphasis
    "question":    {"d_rate": -2, "d_pitch": 12, "pause_ms": 320},   # ? rising
    "calm":        {"d_rate": -5, "d_pitch": -3, "pause_ms": 380},   # explanatory
    "emphatic":    {"d_rate": -3, "d_pitch": 6,  "pause_ms": 300},   # key point
    "closing":     {"d_rate": -7, "d_pitch": -5, "pause_ms": 120},   # wrap-up
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


# Persian cue words that signal emphasis / excitement / closing, used to pick a
# fitting emotional profile per sentence for natural, human-like narration.
_FA_EXCITED = ("شگفت", "باورنکردنی", "فوق‌العاده", "عالی", "جذاب", "هیجان",
               "مهم‌ترین", "بهترین", "خطرناک", "اولین", "حتماً", "قطعاً")
_FA_EMPHATIC = ("توجه", "نکته", "مهم", "کلید", "یادتان باشد", "دقت", "اصلی",
                "در واقع", "باید", "هرگز", "همیشه")
_FA_CLOSING = ("در پایان", "خلاصه", "جمع‌بندی", "نتیجه", "در نهایت", "پس",
               "بنابراین", "امیدوارم", "ممنون", "سپاس")


def _classify_emotion(sentence, is_last=False):
    """Pick an emotional profile for a sentence (punctuation + Persian cues)."""
    s = (sentence or "").strip()
    if not s:
        return "neutral"
    if is_last or any(c in s for c in _FA_CLOSING):
        return "closing"
    if "!" in s or "؟!" in s or any(w in s for w in _FA_EXCITED):
        return "excited"
    if s.endswith("؟") or s.endswith("?"):
        return "question"
    if any(w in s for w in _FA_EMPHATIC):
        return "emphatic"
    # Longer explanatory sentences read calmer; short ones stay neutral.
    if len(s) > 90:
        return "calm"
    return "neutral"


def _split_sentences(text):
    """Split narration into sentences on Persian/Latin sentence punctuation,
    keeping the trailing punctuation so emotion classification still works."""
    if not text:
        return []
    # split but keep delimiters
    parts = re.split(r"(?<=[\.!\?؟।])\s+|(?<=[\.!\?؟])(?=\S)", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # further break very long clauses on Persian comma for breathing room
        if len(p) > 160 and "،" in p:
            sub = [x.strip() for x in p.split("،") if x.strip()]
            out.extend(sub)
        else:
            out.append(p)
    return out or [text]


def _apply_delta(base_pct_str, delta, unit="%", lo=-50, hi=60):
    """Apply an integer delta to a '+N%'/'+NHz' style string, clamped."""
    try:
        n = int(re.sub(r"[^\-0-9]", "", base_pct_str or "0"))
    except Exception:
        n = 0
    n = max(lo, min(hi, n + delta))
    return ("+" if n >= 0 else "") + str(n) + unit


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


def _silence_mp3(path, ms):
    """Create a short silent mp3 (used as a natural pause between sentences)."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "anullsrc=channel_layout=mono:sample_rate=24000",
             "-t", f"{max(1, ms)/1000.0:.3f}", "-q:a", "9", path],
            check=True, capture_output=True)
        return path
    except Exception:
        return None


def _concat_mp3(parts, out_path):
    """Concatenate mp3 parts into one file (re-encode for safe concat)."""
    listfile = out_path + ".list.txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             "-c:a", "libmp3lame", "-q:a", "3", out_path],
            check=True, capture_output=True)
    finally:
        try: os.unlink(listfile)
        except Exception: pass
    return out_path


def _synthesize_expressive(text, out_path, language, chosen_voice,
                           base_rate, base_pitch, base_volume):
    """Render narration sentence-by-sentence with per-sentence EMOTIONAL prosody
    and natural pauses, then stitch into one expressive voiceover.

    This is what gives the Persian narrator real emotional inflection (rising
    questions, excited emphasis, calm explanations, gentle closings) instead of
    a flat, robotic monotone. Returns out_path on success or None to fall back.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return None  # nothing to gain; let the simple path handle it

    workdir = out_path + "_parts"
    os.makedirs(workdir, exist_ok=True)
    parts = []
    try:
        for i, sent in enumerate(sentences):
            profile = EMOTION_PROFILES[_classify_emotion(
                sent, is_last=(i == len(sentences) - 1))]
            r = _apply_delta(base_rate, profile["d_rate"], "%")
            p = _apply_delta(base_pitch, profile["d_pitch"], "Hz", lo=-30, hi=30)
            seg = os.path.join(workdir, f"seg_{i:03d}.mp3")
            ok_seg = False
            for attempt in range(2):
                try:
                    asyncio.run(_edge_save(sent, chosen_voice, seg,
                                           rate=r, pitch=p, volume=base_volume))
                    if os.path.exists(seg) and os.path.getsize(seg) > 400:
                        ok_seg = True
                        break
                except Exception:
                    pass
            if not ok_seg:
                return None  # bail -> simple path / fallbacks
            parts.append(seg)
            # natural pause after the sentence
            pause = os.path.join(workdir, f"pause_{i:03d}.mp3")
            if _silence_mp3(pause, profile["pause_ms"]):
                parts.append(pause)

        if not parts:
            return None
        _concat_mp3(parts, out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path
        return None
    except Exception:
        return None
    finally:
        # cleanup segment parts
        try:
            for f in os.listdir(workdir):
                try: os.unlink(os.path.join(workdir, f))
                except Exception: pass
            os.rmdir(workdir)
        except Exception:
            pass


def synthesize(text, out_path, language="en", voice=None, rate="+0%",
               pitch="+0Hz", gender=None, volume="+0%", expressive=True):
    """
    Convert text to speech mp3 at out_path.
    Returns (out_path, duration_seconds).

    Robust chain (all free):
      0. EXPRESSIVE per-sentence neural TTS (emotional inflection + natural
         pauses) — gives a human, professional-narrator feel (default ON).
      1. Microsoft Edge neural TTS single-shot (native Persian voices) with
         retries (handles transient network blips).
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

    # 0) Expressive, emotion-aware multi-sentence rendering (best quality feel).
    if expressive:
        try:
            res = _synthesize_expressive(text, out_path, language, chosen_voice,
                                         rate, pitch, volume)
            if res and os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return out_path, get_audio_duration(out_path)
        except Exception:
            pass

    # 1) Edge-tts single-shot (best quality) with retries.
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

    # 2/3) Fallback to Google TTS (gTTS endpoint).
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
