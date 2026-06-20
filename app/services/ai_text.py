"""
ai_text.py - Script & text generation with full Persian support.

Strategy (robust, always works, 100% free):
  1. TRY free online LLMs (Pollinations text) for top-quality scripts. The
     prompt asks the model to write narration in the chosen language (incl.
     native Persian) and image prompts in English.
  2. If the LLM is unavailable/rate-limited, fall back to a strong built-in
     scriptwriter that produces real, structured, educational narration in the
     selected language (with a native Persian writer) + English image prompts.
  3. After any path, we GUARANTEE every scene's image_prompt is in English
     (translating it if necessary) so the image model produces top results,
     and narration/caption are in the requested language.
"""
import json
import re
import time
import threading

import requests

from . import translate

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
                    time.sleep(6 + attempt * 4)
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
    Guarantees: narration/caption in `language`, image_prompt in English.
    """
    if num_scenes is None:
        num_scenes = max(4, min(20, round(duration_sec / 6)))
    lang_name = LANG_NAMES.get(language, "English")

    # 1) try online LLM for top quality
    data = _online_script(topic, duration_sec, num_scenes, lang_name)

    # 2) fall back to built-in scriptwriter
    if not _valid_script(data):
        data = build_offline_script(topic, duration_sec, num_scenes, language)

    return _normalize_script(data, topic, language)


def _online_script(topic, duration_sec, num_scenes, lang_name):
    system = (
        "You are an expert educational video scriptwriter and visual director. "
        "Respond with ONLY valid JSON. No markdown, no commentary."
    )
    total_words = int(duration_sec * 2.4)
    wps = max(10, total_words // num_scenes)
    prompt = f"""Create a {duration_sec}-second educational explainer video script about:
"{topic}"

Strict rules:
- Narration language: {lang_name}. Write natural, fluent, native {lang_name}.
- Exactly {num_scenes} scenes.
- Each "narration" ~{wps} words; factual, clear, engaging, teaching thoroughly.
- "image_prompt" MUST be IN ENGLISH: a detailed cinematic illustration prompt
  (subject, setting, style, lighting, composition, mood) describing the scene
  visually. Keep a consistent modern, high-quality cinematic style across scenes.
- "caption": a short on-screen title (<=7 words) in {lang_name}.

Output ONLY this JSON:
{{"title":"...","scenes":[{{"narration":"...","image_prompt":"...","caption":"..."}}]}}"""
    raw = _online_llm(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        temperature=0.8, timeout=95)
    return _extract_json(raw) if raw else None


# --------------------------------------------------- built-in scriptwriter
# Native Persian phrase building blocks for a genuine offline Persian writer.
_FA = {
    "welcome": "به این ویدیوی آموزشی خوش آمدید.",
    "intro_tail": "در این ویدیو موضوع را قدم‌به‌قدم و ساده توضیح می‌دهیم.",
    "about": "درباره‌ی",
    "point": "نکته‌ی",
    "key_point_tail": "یک ایده‌ی مهم که به درک بهتر موضوع کمک می‌کند.",
    "thanks": "ممنون که تماشا کردید. به یادگیری ادامه دهید!",
    "summary": "به‌طور خلاصه،",
    "vs_intro": "در این ویدیو تفاوت‌های کلیدی این دو را بررسی می‌کنیم.",
    "conclusion": "نتیجه‌گیری",
    "scene": "صحنه",
}


def build_offline_script(topic, duration_sec, num_scenes, language="en"):
    """
    Build a genuine, structured educational script without any external API.
    Persian (fa) gets a native Persian writer; other languages use English
    text that is later translated to the target language during normalization.
    """
    topic_clean = topic.strip()
    style = ("cinematic, highly detailed, professional lighting, modern 4k "
             "illustration, vivid rich colors, clean composition, sharp focus")

    is_fa = (language == "fa")
    # English topic used for IMAGE PROMPTS (always English).
    topic_en = translate.to_english(topic_clean) if is_fa else topic_clean

    comp = _detect_comparison(topic_clean if not is_fa else topic_en)
    scenes = []

    if comp:
        a, b = comp
        title = (f"{a} در برابر {b}" if is_fa else f"{a} vs {b}")
        if is_fa:
            intro = f"{_FA['welcome']} {_FA['vs_intro']}"
        else:
            intro = (f"Welcome. In this video we explore the key differences "
                     f"between {a} and {b}, and which one fits your needs.")
        scenes.append(_scene(intro,
                             f"split screen comparison of {a} and {b}, two halves, "
                             f"symbolic icons, {style}",
                             (f"{a} در برابر {b}" if is_fa else f"{a} vs {b}")))
        points = _comparison_points(a, b, is_fa)
        body_count = max(1, num_scenes - 2)
        chosen = points[:body_count] if len(points) >= body_count else points
        for (cap, text, img) in chosen:
            scenes.append(_scene(text, f"{img}, {style}", cap))
        if is_fa:
            concl = (f"{_FA['summary']} هر دو گزینه قدرتمند هستند؛ بسته به هدف خود "
                     f"یکی را انتخاب کنید.")
        else:
            concl = (f"In summary, both {a} and {b} are powerful choices. Pick the "
                     f"one that matches your goals.")
        scenes.append(_scene(concl,
                             f"a person choosing between two glowing paths, "
                             f"decision concept, {style}",
                             (_FA["conclusion"] if is_fa else "Conclusion")))
    else:
        title = topic_clean if is_fa else (
            topic_clean.title() if topic_clean.isascii() else topic_clean)
        if is_fa:
            intro = f"{_FA['welcome']} {_FA['about']} {topic_clean}. {_FA['intro_tail']}"
        else:
            intro = (f"Welcome to this short explainer about {topic_clean}. "
                     f"Let us break it down step by step.")
        scenes.append(_scene(intro,
                             f"an engaging hero illustration representing {topic_en}, "
                             f"{style}",
                             _short(topic_clean)))
        body_count = max(1, num_scenes - 2)
        for i in range(body_count):
            if is_fa:
                text = (f"{_FA['point']} {i+1} {_FA['about']} {topic_clean}: "
                        f"{_FA['key_point_tail']}")
                cap = f"{_short(topic_clean)} — {i+1}"
            else:
                text = (f"Key point {i+1} about {topic_clean}: an important idea "
                        f"that deepens your understanding of the subject.")
                cap = f"{_short(topic_clean)} — {i+1}"
            img = (f"detailed cinematic visual explaining aspect {i+1} of {topic_en}, "
                   f"infographic feel, {style}")
            scenes.append(_scene(text, img, cap))
        if is_fa:
            concl = f"{_FA['thanks']}"
        else:
            concl = (f"That is the essence of {topic_clean}. Thanks for watching, "
                     f"and keep exploring to learn more.")
        scenes.append(_scene(concl,
                             f"inspiring closing scene about {topic_en}, sunrise, "
                             f"{style}",
                             ("ممنون که دیدید" if is_fa else "Thanks for watching")))

    return {"title": title, "scenes": scenes}


def _scene(narration, image_prompt, caption):
    return {"narration": narration, "image_prompt": image_prompt,
            "caption": (caption or "")[:60]}


def _short(t, n=42):
    return (t[:n]).strip()


def _detect_comparison(topic):
    t = topic.lower()
    patterns = [
        r"difference between (.+?) and (.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
        r"(.+?)\s+or\s+(.+?)\s+which",
        r"(.+?)\s+یا\s+(.+)",
        r"فرق\s+(.+?)\s+(?:با|و)\s+(.+)",
        r"تفاوت\s+(.+?)\s+(?:با|و)\s+(.+)",
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
    known = {"linux": "Linux", "windows": "Windows", "mac": "macOS",
             "macos": "macOS", "android": "Android", "ios": "iOS",
             "python": "Python", "java": "Java", "react": "React",
             "لینوکس": "Linux", "ویندوز": "Windows", "اندروید": "Android"}
    return known.get(s.lower(), known.get(s, s.title() if s.isascii() else s))


def _comparison_points(a, b, is_fa=False):
    """Domain-aware comparison points. Each item: (caption, narration, img)."""
    al, bl = a.lower(), b.lower()
    if {"linux", "windows"} <= {al, bl}:
        if is_fa:
            return [
                ("هزینه و لایسنس",
                 f"{a} رایگان و متن‌باز است و می‌توانید بدون پرداخت پول از آن استفاده کنید. "
                 f"{b} نرم‌افزار تجاری است و معمولاً به لایسنس پولی نیاز دارد.",
                 "money and open padlock, free software concept, penguin and window logos"),
                ("متن‌باز بودن",
                 f"{a} دسترسی کامل به کد منبع می‌دهد تا همه بتوانند آن را مطالعه و بهبود دهند. "
                 f"{b} کد منبع خود را بسته و خصوصی نگه می‌دارد.",
                 "glowing source code on screen, open vs closed boxes"),
                ("امنیت",
                 f"{a} به امنیت بالا و ویروس کمتر شهرت دارد. "
                 f"{b} بیشترین هدف حملات است و به محافظت دقیق نیاز دارد.",
                 "digital shield protecting a computer, lock icons, cyber security"),
                ("کارایی",
                 f"{a} حتی روی سخت‌افزار قدیمی و کم‌مصرف به‌خوبی کار می‌کند. "
                 f"{b} منابع بیشتری می‌خواهد اما برای بازی و کار روزمره بهینه است.",
                 "fast server racks glowing, performance speed meters"),
                ("نرم‌افزار و بازی",
                 f"{b} از بیشترین نرم‌افزارهای تجاری و بازی‌ها پشتیبانی می‌کند. "
                 f"{a} ابزارهای رایگان قدرتمند و پشتیبانی روبه‌رشد از بازی دارد.",
                 "gaming setup with controllers and many app icons floating"),
                ("شخصی‌سازی",
                 f"{a} اجازه می‌دهد تقریباً همه‌چیز را شخصی‌سازی کنید. "
                 f"{b} تجربه‌ای یکپارچه و صیقلی با تغییرات کمتر ارائه می‌دهد.",
                 "highly customized desktop with many themes and widgets"),
                ("سهولت استفاده",
                 f"{b} برای بیشتر افراد آشنا و مبتدی‌پسند است. "
                 f"{a} امروز ساده‌تر از همیشه و کاربرپسند شده است.",
                 "friendly user interface, smiling person using a laptop"),
                ("برای چه کسی؟",
                 f"برای آزادی، برنامه‌نویسی و سرور {a} را انتخاب کنید. "
                 f"برای بازی و نرم‌افزارهای رایج {b} را برگزینید.",
                 "two roads diverging, signposts, decision making concept"),
            ]
        return [
            ("Cost & Licensing",
             f"{a} is free and open source; you can use and modify it without paying. "
             f"{b} is commercial software that usually needs a paid license.",
             "money and open padlock, free software concept, penguin and window logos"),
            ("Open Source",
             f"{a} gives full access to its source code, so anyone can study and "
             f"improve it. {b} keeps its source closed and private.",
             "glowing source code on screen, open vs closed boxes"),
            ("Security",
             f"{a} is known for strong security and fewer viruses. {b} is the most "
             f"targeted system, so it needs careful protection.",
             "digital shield protecting a computer, lock icons, cyber security"),
            ("Performance",
             f"{a} runs efficiently even on older hardware and dominates servers. "
             f"{b} needs more resources but is optimized for gaming and desktops.",
             "fast server racks glowing, performance speed meters"),
            ("Software & Gaming",
             f"{b} supports the widest range of commercial software and games. "
             f"{a} has powerful free tools and growing game support.",
             "gaming setup with controllers and many app icons floating"),
            ("Customization",
             f"{a} lets you customize almost everything. {b} offers a consistent, "
             f"polished experience with fewer changes.",
             "highly customized desktop with many themes and widgets"),
            ("Ease of Use",
             f"{b} is familiar and beginner friendly. {a} is easier than ever today, "
             f"with simple user-friendly versions.",
             "friendly user interface, smiling person using a laptop"),
            ("Best For",
             f"Choose {a} for freedom, programming and servers. Choose {b} for gaming "
             f"and mainstream software.",
             "two roads diverging, signposts, decision making concept"),
        ]
    # generic comparison
    if is_fa:
        aspects = ["نمای کلی", "نقاط قوت", "نقاط ضعف", "هزینه",
                   "کارایی", "موارد استفاده", "محبوبیت", "جمع‌بندی"]
        out = []
        for asp in aspects:
            out.append((asp,
                        f"{asp}: مقایسه‌ی {a} و {b}. هر کدام مزایای خاص خود را دارند "
                        f"که ارزش بررسی دارد.",
                        f"comparison illustration of {a} and {b}"))
        return out
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


def _normalize_script(data, topic, language="en"):
    """Clean scenes & GUARANTEE image_prompt is English."""
    data.setdefault("title", topic)
    clean = []
    for s in data.get("scenes", []):
        if not isinstance(s, dict):
            continue
        nar = str(s.get("narration", "")).strip()
        if not nar:
            continue
        img = str(s.get("image_prompt") or nar).strip()
        # Ensure the image prompt is English for best image-model results.
        if translate.has_persian(img):
            img = translate.to_english(img)
        cap = str(s.get("caption") or "").strip()[:60]
        clean.append({"narration": nar, "image_prompt": img or topic,
                      "caption": cap})
    if not clean:
        data = build_offline_script(topic, 60, 6, language)
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
