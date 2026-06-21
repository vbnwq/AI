"""
translate.py - Persian (Farsi) -> English translation layer.

Many free AI models (especially image models) understand English far better
than Persian. To keep the app fully usable in Persian while still getting
top-tier results, every prompt that goes to an image model is passed through
this layer first.

Strategy (all free, no API key, never fails):
  1. Detect whether the text actually contains Persian/Arabic script.
  2. If it does, try multiple FREE translation backends in order:
        a. Google translate (unofficial, free `gtx` endpoint)
        b. Pollinations LLM (instructed to translate only)
        c. A small built-in Persian->English glossary (offline guarantee)
  3. Return the best translation; if everything fails, return the original
     text (the image model will still try its best).

Results are cached in-memory so repeated prompts are instant.
"""
import re
import json
import time
import threading
import urllib.parse
import requests

_CACHE = {}
_CACHE_LOCK = threading.Lock()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
}

# Persian / Arabic Unicode ranges
_PERSIAN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def has_persian(text):
    """True if the text contains any Persian/Arabic characters."""
    if not text:
        return False
    return bool(_PERSIAN_RE.search(text))


def looks_english(text):
    """Heuristic: text is already (mostly) latin / english."""
    if not text:
        return True
    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text)
    if not letters:
        return True
    latin = sum(1 for c in letters if c.isascii())
    return latin / max(1, len(letters)) > 0.6


# ----------------------------------------------------------- backend 1: google
def _google_translate(text, src="fa", dest="en", timeout=12):
    """Free unofficial Google translate endpoint (gtx). Returns text or None."""
    try:
        url = ("https://translate.googleapis.com/translate_a/single"
               "?client=gtx&sl=" + src + "&tl=" + dest + "&dt=t&q="
               + urllib.parse.quote(text))
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                out = "".join(seg[0] for seg in data[0] if seg and seg[0])
                out = out.strip()
                if out:
                    return out
    except Exception:
        pass
    return None


def _google_translate_alt(text, src="fa", dest="en", timeout=12):
    """A second google host as a backup mirror."""
    try:
        url = ("https://clients5.google.com/translate_a/t"
               "?client=dict-chrome-ex&sl=" + src + "&tl=" + dest + "&q="
               + urllib.parse.quote(text))
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            # format: [["translated","src"]] OR {"sentences":[...]}
            if isinstance(data, list) and data:
                if isinstance(data[0], list):
                    return "".join(x[0] for x in data if x and x[0]).strip() or None
                if isinstance(data[0], str):
                    return " ".join(data).strip() or None
    except Exception:
        pass
    return None


# --------------------------------------------------------- backend 2: LLM
def _llm_translate(text, timeout=40):
    """Use the free Pollinations text LLM to translate. Returns text or None."""
    try:
        payload = {
            "model": "openai",
            "messages": [
                {"role": "system",
                 "content": "You are a professional Persian-to-English translator. "
                            "Translate the user's text to natural, vivid English. "
                            "Return ONLY the translation, no quotes, no notes."},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
            "referrer": "freeaistudio.app",
        }
        r = requests.post("https://text.pollinations.ai/openai",
                          json=payload, headers=_HEADERS, timeout=timeout)
        if r.status_code == 200 and r.text.strip():
            try:
                data = r.json()
                if isinstance(data, dict) and data.get("choices"):
                    out = data["choices"][0]["message"]["content"].strip()
                    if out:
                        return out
            except Exception:
                if not r.text.lstrip().startswith("{"):
                    return r.text.strip()
    except Exception:
        pass
    return None


# ------------------------------------------------- backend 3: offline glossary
# Small but useful glossary so even fully offline we improve the prompt a lot.
_GLOSSARY = {
    "عکس": "photo", "تصویر": "image", "ویدیو": "video", "فیلم": "movie",
    "شهر": "city", "آینده": "future", "آینده‌نگرانه": "futuristic",
    "ماشین": "car", "ماشین‌های پرنده": "flying cars", "پرنده": "flying",
    "نور": "light", "نورهای نئونی": "neon lights", "نئون": "neon",
    "شب": "night", "روز": "day", "سینمایی": "cinematic", "باکیفیت": "high quality",
    "خیابان": "street", "جنگل": "forest", "کوه": "mountain", "دریا": "sea",
    "اقیانوس": "ocean", "آسمان": "sky", "ابر": "cloud", "خورشید": "sun",
    "ماه": "moon", "ستاره": "star", "گل": "flower", "درخت": "tree",
    "حیوان": "animal", "گربه": "cat", "سگ": "dog", "اسب": "horse",
    "شیر": "lion", "ببر": "tiger", "پرنده‌ای": "bird", "اژدها": "dragon",
    "زن": "woman", "مرد": "man", "کودک": "child", "دختر": "girl", "پسر": "boy",
    "پرتره": "portrait", "منظره": "landscape", "طبیعت": "nature",
    "غروب": "sunset", "طلوع": "sunrise", "بارانی": "rainy", "برفی": "snowy",
    "زمستان": "winter", "تابستان": "summer", "بهار": "spring", "پاییز": "autumn",
    "قلعه": "castle", "خانه": "house", "ساختمان": "building", "پل": "bridge",
    "فضا": "space", "سیاره": "planet", "کهکشان": "galaxy", "ربات": "robot",
    "هوش مصنوعی": "artificial intelligence", "علمی تخیلی": "science fiction",
    "فانتزی": "fantasy", "واقع‌گرایانه": "realistic", "سه بعدی": "3D",
    "نقاشی": "painting", "طراحی": "drawing", "انیمه": "anime",
    "زیبا": "beautiful", "بزرگ": "large", "کوچک": "small", "قرمز": "red",
    "آبی": "blue", "سبز": "green", "زرد": "yellow", "سیاه": "black",
    "سفید": "white", "طلایی": "golden", "نقره‌ای": "silver", "رنگارنگ": "colorful",
    "قهرمان": "hero", "جنگجو": "warrior", "شوالیه": "knight", "جادوگر": "wizard",
    "لینوکس": "Linux", "ویندوز": "Windows", "کامپیوتر": "computer",
    "گوشی": "smartphone", "اینترنت": "internet", "بازی": "game",
    "خواب": "sleep", "سلامتی": "health", "ورزش": "exercise", "غذا": "food",
    "فواید": "benefits", "آموزش": "education", "درباره": "about",
}


def _offline_translate(text):
    """Word-by-word glossary translation as a last resort."""
    out = text
    # longer keys first to catch multi-word phrases
    for fa in sorted(_GLOSSARY, key=len, reverse=True):
        if fa in out:
            out = out.replace(fa, " " + _GLOSSARY[fa] + " ")
    # strip any remaining persian chars and collapse spaces
    out = _PERSIAN_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip(" ,،.")
    return out.strip()


# --------------------------------------------------------------- public API
def to_english(text, force=False):
    """
    Translate `text` to English if it contains Persian/Arabic script.
    If the text is already English (and not forced), returns it unchanged.
    Always returns a usable (non-empty) string.
    """
    text = (text or "").strip()
    if not text:
        return text
    if not force and not has_persian(text):
        return text

    key = text
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    result = None
    for backend in (_google_translate, _google_translate_alt, _llm_translate):
        try:
            out = backend(text)
        except Exception:
            out = None
        if out and out.strip() and not has_persian(out):
            result = out.strip()
            break
        time.sleep(0.2)

    if not result:
        result = _offline_translate(text)

    if not result or not result.strip():
        result = text  # ultimate fallback: original

    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def translate(text, src="auto", dest="en"):
    """General translate helper used by other modules (e.g. en->fa for captions)."""
    text = (text or "").strip()
    if not text:
        return text
    if dest == "en":
        return to_english(text)
    # other direction (e.g. en -> fa)
    out = _google_translate(text, src="en" if src == "auto" else src, dest=dest)
    if not out:
        out = _google_translate_alt(text, src="en" if src == "auto" else src, dest=dest)
    return out or text
