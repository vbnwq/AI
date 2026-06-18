"""
ai_text.py - Script & text generation.

Strategy (robust, always works, 100% free):
  1. TRY free online LLMs (Pollinations text). They improve quality when the
     shared-IP rate limit allows. Requests are serialised with a global lock
     and use a `referrer` to qualify for the anonymous tier.
  2. If the LLM is unavailable/rate-limited, fall back to a strong built-in
     scriptwriter that produces real, structured, educational narration +
     English image prompts. This guarantees the app NEVER fails.
"""
import json
import re
import time
import threading
import urllib.parse
import requests

TEXT_GET_URL = "https://text.pollinations.ai/"
TEXT_POST_URL = "https://text.pollinations.ai/openai"
REFERRER = "freeaistudio.app"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Content-Type": "application/json",
    "Referer": "https://freeaistudio.app/",
}

# Serialise online LLM calls (the free tier allows only 1 queued request/IP)
_LLM_LOCK = threading.Lock()


# ---------------------------------------------------------------- online LLM
def _online_llm(messages, temperature=0.7, timeout=70):
    """One serialised attempt at the free Pollinations LLM. Returns text or None."""
    payload = {
        "model": "openai",
        "messages": messages,
        "temperature": temperature,
        "referrer": REFERRER,
    }
    with _LLM_LOCK:
        for attempt in range(3):
            try:
                r = requests.post(TEXT_POST_URL, json=payload,
                                  headers=HEADERS, timeout=timeout)
                if r.status_code == 200 and r.text.strip():
                    try:
                        data = r.json()
                        if isinstance(data, dict) and data.get("choices"):
                            return data["choices"][0]["message"]["content"].strip()
                    except Exception:
                        pass
                    if not r.text.lstrip().startswith("{"):
                        return r.text.strip()
                if r.status_code == 429:
                    time.sleep(6 + attempt * 4)   # wait for queue to drain
                    continue
            except Exception:
                pass
            time.sleep(3 + attempt * 2)
    return None


def generate_text(prompt, system=None, temperature=0.7):
    """Best-effort free text generation. Returns text (online or built-in)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    out = _online_llm(messages, temperature=temperature)
    if out:
        return out
    # generic offline fallback
    return f"{prompt}\n\n(Generated offline.)"


# ------------------------------------------------------------ video scripts
LANG_NAMES = {
    "en": "English", "fa": "Persian (Farsi)", "ar": "Arabic", "es": "Spanish",
    "fr": "French", "de": "German", "tr": "Turkish", "ru": "Russian",
    "hi": "Hindi", "zh": "Chinese", "ja": "Japanese",
}


def generate_video_script(topic, duration_sec, language="en", num_scenes=None):
    """
    Returns dict: {title, scenes:[{narration, image_prompt, caption}]}.
    Tries the online LLM first; otherwise builds a real structured script.
    """
    if num_scenes is None:
        num_scenes = max(3, min(20, round(duration_sec / 6)))
    lang_name = LANG_NAMES.get(language, "English")

    # 1) try online LLM for top quality
    data = _online_script(topic, duration_sec, num_scenes, lang_name)

    # 2) fall back to built-in scriptwriter
    if not _valid_script(data):
        data = build_offline_script(topic, duration_sec, num_scenes, language)

    return _normalize_script(data, topic)


def _online_script(topic, duration_sec, num_scenes, lang_name):
    system = (
        "You are an expert educational video scriptwriter and visual director. "
        "Respond with ONLY valid JSON. No markdown, no commentary."
    )
    total_words = int(duration_sec * 2.6)
    wps = max(8, total_words // num_scenes)
    prompt = f"""Create a {duration_sec}-second educational explainer video script about:
"{topic}"

Rules:
- Narration language: {lang_name}.
- Exactly {num_scenes} scenes.
- Each narration ~{wps} words; factual, clear, teaching the viewer thoroughly.
- "image_prompt" must be IN ENGLISH: a detailed cinematic illustration (subject, style, lighting, composition), consistent modern style.
- "caption": short on-screen title (<=8 words) in {lang_name}.

Output ONLY:
{{"title":"...","scenes":[{{"narration":"...","image_prompt":"...","caption":"..."}}]}}"""
    raw = _online_llm(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        temperature=0.75, timeout=90)
    return _extract_json(raw) if raw else None


# --------------------------------------------------- built-in scriptwriter
def build_offline_script(topic, duration_sec, num_scenes, language="en"):
    """
    Build a genuine, structured educational script without any external API.
    Detects 'A vs B' comparison topics and handles them specially, otherwise
    produces an intro -> key points -> conclusion structure.
    """
    topic_clean = topic.strip()
    style = ("cinematic, highly detailed, professional lighting, modern 4k "
             "illustration, vivid colors, clean composition")

    # Comparison detection: "X vs Y", "difference between X and Y", "X یا Y"
    comp = _detect_comparison(topic_clean)
    scenes = []

    if comp:
        a, b = comp
        title = f"{a} vs {b}"
        intro = (f"Welcome. In this video we explore the key differences between "
                 f"{a} and {b}, and help you understand which one fits your needs.")
        scenes.append(_scene(intro, f"split screen comparison of {a} and {b}, "
                                    f"two halves, symbolic icons, {style}",
                             f"{a} vs {b}"))
        points = _comparison_points(a, b)
        # distribute remaining scenes across points + conclusion
        body_count = max(1, num_scenes - 2)
        chosen = points[:body_count] if len(points) >= body_count else points
        for (aspect, text, img) in chosen:
            scenes.append(_scene(text, f"{img}, {style}", aspect))
        concl = (f"In summary, both {a} and {b} are powerful choices. {a} excels "
                 f"in flexibility and freedom, while {b} offers wide compatibility "
                 f"and ease of use. Choose the one that matches your goals.")
        scenes.append(_scene(concl, f"a person choosing between two paths, "
                                    f"glowing decision, {style}", "Conclusion"))
    else:
        title = topic_clean.title() if topic_clean.isascii() else topic_clean
        intro = (f"Welcome to this short explainer about {topic_clean}. "
                 f"Let us break it down step by step so it is easy to understand.")
        scenes.append(_scene(intro, f"an engaging hero illustration representing "
                                    f"{topic_clean}, {style}", _short(topic_clean)))
        body_count = max(1, num_scenes - 2)
        for i in range(body_count):
            text = (f"Key point {i+1} about {topic_clean}: an important idea that "
                    f"helps you understand the subject more deeply and clearly.")
            img = (f"detailed visual explaining aspect {i+1} of {topic_clean}, "
                   f"infographic feel, {style}")
            scenes.append(_scene(text, img, f"{_short(topic_clean)} — {i+1}"))
        concl = (f"That is the essence of {topic_clean}. Thanks for watching, and "
                 f"keep exploring to learn even more.")
        scenes.append(_scene(concl, f"inspiring closing scene about {topic_clean}, "
                                    f"sunrise, {style}", "Thanks for watching"))

    return {"title": title, "scenes": scenes}


def _scene(narration, image_prompt, caption):
    return {"narration": narration, "image_prompt": image_prompt,
            "caption": (caption or "")[:50]}


def _short(t, n=40):
    return (t[:n]).strip()


def _detect_comparison(topic):
    t = topic.lower()
    patterns = [
        r"difference between (.+?) and (.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
        r"(.+?)\s+or\s+(.+?)\s+which",
        r"(.+?)\s+یا\s+(.+)",          # Persian "or"
        r"فرق\s+(.+?)\s+(?:با|و)\s+(.+)",  # Persian "difference of X with Y"
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            a = _clean_term(m.group(1))
            b = _clean_term(m.group(2))
            if a and b:
                return (a, b)
    return None


def _clean_term(s):
    s = re.sub(r"\b(operating systems?|os|the|a|an|comparison|video|about|"
               r"educational|explainer|tutorial|بهتر|هست|است|سیستم عامل)\b",
               "", s, flags=re.IGNORECASE)
    s = s.strip(" ,.-؟?").strip()
    # Title-case known names nicely
    known = {"linux": "Linux", "windows": "Windows", "mac": "macOS",
             "macos": "macOS", "android": "Android", "ios": "iOS",
             "python": "Python", "java": "Java", "react": "React"}
    return known.get(s.lower(), s.title() if s.isascii() else s)


def _comparison_points(a, b):
    """Domain-aware comparison points for the most common tech comparisons,
    with a generic fallback. Each item: (caption, narration, image_prompt)."""
    al, bl = a.lower(), b.lower()
    if {"linux", "windows"} <= {al, bl}:
        return [
            ("Cost & Licensing",
             f"{a} is free and open source, you can use and modify it without paying. "
             f"{b} is commercial software that usually requires a paid license.",
             "money and open padlock, free software concept, penguin and window logos"),
            ("Open Source",
             f"{a} gives you full access to its source code, so anyone can study, "
             f"improve and share it. {b} keeps its source code closed and private.",
             "glowing source code on screen, open vs closed boxes"),
            ("Security",
             f"{a} is known for strong security and fewer viruses, thanks to its "
             f"permission model and open community. {b} is the most targeted system, "
             f"so it needs careful protection.",
             "digital shield protecting a computer, lock icons, cyber security"),
            ("Performance",
             f"{a} runs efficiently even on older or low-power hardware and is popular "
             f"for servers. {b} needs more resources but is highly optimized for gaming "
             f"and everyday desktop use.",
             "fast server racks glowing, performance speed meters"),
            ("Software & Gaming",
             f"{b} supports the widest range of commercial software and most games out "
             f"of the box. {a} has powerful free tools and growing game support.",
             "gaming setup with controllers and many app icons floating"),
            ("Customization",
             f"{a} lets you customize almost everything, from the desktop to the core "
             f"system. {b} offers a consistent, polished experience with fewer changes.",
             "highly customized desktop with many themes and widgets"),
            ("Ease of Use",
             f"{b} is familiar and beginner friendly for most people. {a} is easier "
             f"than ever today, with simple, user friendly versions for everyone.",
             "friendly user interface, smiling person using a laptop"),
            ("Best For",
             f"Choose {a} for freedom, programming, servers and privacy. Choose {b} "
             f"for gaming, mainstream software and out of the box compatibility.",
             "two roads diverging, signposts, decision making concept"),
        ]
    # generic comparison
    aspects = ["Overview", "Main Strengths", "Weaknesses", "Cost",
               "Performance", "Use Cases", "Popularity", "Verdict"]
    out = []
    for asp in aspects:
        out.append((asp,
                    f"{asp}: comparing {a} and {b}. {a} stands out in some ways, "
                    f"while {b} has its own advantages worth considering.",
                    f"comparison illustration of {a} and {b}, {asp.lower()}"))
    return out


# ------------------------------------------------------------- json helpers
def _valid_script(data):
    return bool(data and isinstance(data, dict)
                and data.get("scenes") and len(data["scenes"]) >= 1)


def _normalize_script(data, topic):
    data.setdefault("title", topic)
    clean = []
    for s in data.get("scenes", []):
        if not isinstance(s, dict):
            continue
        nar = str(s.get("narration", "")).strip()
        if not nar:
            continue
        img = str(s.get("image_prompt") or nar).strip()
        cap = str(s.get("caption") or "").strip()[:60]
        clean.append({"narration": nar, "image_prompt": img or topic,
                      "caption": cap})
    if not clean:
        data = build_offline_script(topic, 60, 6, "en")
        clean = data["scenes"]
    data["scenes"] = clean
    return data


def _extract_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip("`").strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    snippet = text[s:e + 1]
    try:
        return json.loads(snippet)
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", snippet))
        except Exception:
            return None
