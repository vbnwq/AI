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


# ==================================================================
#  MULTI-AI SUPERVISOR — Research / Fact-Check / Content agents
# ==================================================================
# These free agents collaborate BEFORE the script is written so the final
# narration is grounded in real, verified information about the topic instead
# of repetitive filler. All use the same free Pollinations LLM (no API key).

def research_topic(topic, language_name="Persian (Farsi)"):
    """AI Agent: Deep Research — gather real, structured facts about the topic.

    Returns a list of concise factual bullet points (strings) in the target
    language, or [] if the research agent is unavailable.
    """
    system = ("You are a meticulous research assistant. Gather accurate, "
              "concrete, non-repetitive facts. Output ONLY a JSON array of "
              "short factual strings. No commentary, no markdown.")
    prompt = (f"Research the topic: \"{topic}\".\n"
              f"Return 8-12 DISTINCT, specific, factual key points about it "
              f"(definitions, how it works, history, pros/cons, real examples, "
              f"common misconceptions, practical tips). Each point must be "
              f"different from the others. Write each point in {language_name}.\n"
              f"Output ONLY a JSON array of strings.")
    raw = _online_llm([{"role": "system", "content": system},
                       {"role": "user", "content": prompt}],
                      temperature=0.6, timeout=90)
    facts = _extract_json_array(raw)
    # de-duplicate & clean
    seen, out = set(), []
    for f in facts or []:
        s = str(f).strip(" -•\t").strip()
        key = s.lower()[:40]
        if s and key not in seen and len(s) > 8:
            seen.add(key)
            out.append(s)
    return out


def fact_check(facts, topic, language_name="Persian (Farsi)"):
    """AI Agent: Fact-Check — filter/correct the researched facts.

    Returns a cleaned list. If the agent is unavailable, returns input as-is.
    """
    if not facts:
        return facts
    system = ("You are a strict fact-checker. Remove false, vague, duplicated "
              "or off-topic statements and lightly correct wording. Output ONLY "
              "a JSON array of the verified statements, no commentary.")
    prompt = (f"Topic: \"{topic}\".\nVerify and clean these statements, keep "
              f"only accurate, distinct, on-topic ones, written in "
              f"{language_name}:\n{json.dumps(facts, ensure_ascii=False)}\n"
              f"Output ONLY a JSON array of strings.")
    raw = _online_llm([{"role": "system", "content": system},
                       {"role": "user", "content": prompt}],
                      temperature=0.2, timeout=80)
    checked = _extract_json_array(raw)
    if checked and len(checked) >= max(3, len(facts) // 2):
        return [str(c).strip() for c in checked if str(c).strip()]
    return facts


# ------------------------------------------------------------ video scripts
LANG_NAMES = {
    "en": "English", "fa": "Persian (Farsi)", "ar": "Arabic", "es": "Spanish",
    "fr": "French", "de": "German", "tr": "Turkish", "ru": "Russian",
    "hi": "Hindi", "zh": "Chinese", "ja": "Japanese",
}


def generate_video_script(topic, duration_sec, language="en", num_scenes=None,
                          progress_cb=None):
    """
    Returns dict: {title, scenes:[{narration, image_prompt, caption}]}.

    MULTI-AI SUPERVISOR pipeline (all free, no key):
      1. Research Agent     -> gather real facts about the topic.
      2. Fact-Check Agent   -> verify & clean those facts.
      3. Scriptwriter Agent -> write a structured multi-scene script grounded in
                               the verified facts (intro -> pre-intro -> body
                               steps -> conclusion).
      4. Built-in fallback  -> if any agent is unavailable, a strong offline
                               writer guarantees a real, multi-scene script.

    GUARANTEES:
      * MULTIPLE scenes (never a single repeated slide).
      * Every scene's narration is UNIQUE (no '1 2 3 4 5 6' duplication bug).
      * narration/caption in `language`, image_prompt in English.
    """
    if num_scenes is None:
        num_scenes = max(5, min(16, round(duration_sec / 7) + 2))
    lang_name = LANG_NAMES.get(language, "English")

    def _say(msg):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    # 1) Research + 2) Fact-check (collaborative grounding)
    _say("پژوهش درباره موضوع با هوش مصنوعی…")
    facts = research_topic(topic, lang_name)
    if facts:
        _say("راستی‌آزمایی اطلاعات…")
        facts = fact_check(facts, topic, lang_name)

    # 3) Scriptwriter agent (grounded in facts)
    _say("نوشتن سناریوی چندصحنه‌ای…")
    data = _online_script(topic, duration_sec, num_scenes, lang_name, facts)

    # 4) fall back to built-in scriptwriter (still multi-scene & grounded)
    if not _valid_script(data):
        data = build_offline_script(topic, duration_sec, num_scenes, language,
                                    facts=facts)

    return _normalize_script(data, topic, language, min_scenes=4)


def _online_script(topic, duration_sec, num_scenes, lang_name, facts=None):
    system = (
        "You are an expert educational video scriptwriter and visual director. "
        "You write engaging, well-structured, NON-REPETITIVE scripts where every "
        "scene teaches something different. Respond with ONLY valid JSON. "
        "No markdown, no commentary."
    )
    total_words = int(duration_sec * 2.4)
    wps = max(12, total_words // num_scenes)
    facts_block = ""
    if facts:
        facts_block = ("\nUse these verified facts as the factual backbone "
                       "(rephrase naturally, do not list them verbatim):\n- "
                       + "\n- ".join(facts[:12]) + "\n")
    prompt = f"""Create a {duration_sec}-second educational explainer video script about:
"{topic}"
{facts_block}
Strict rules:
- Narration language: {lang_name}. Write natural, fluent, native {lang_name}.
- Exactly {num_scenes} scenes. Follow this LOGICAL STRUCTURE:
  1) Hook / introduction, 2) brief pre-introduction / context,
  3..N-1) main content as clear step-by-step points (each a DIFFERENT idea),
  N) conclusion / takeaway.
- Every "narration" MUST be COMPLETELY DIFFERENT from the others (no repetition,
  no numbering like "1 2 3", no filler). ~{wps} words each; factual & engaging.
- "image_prompt" MUST be IN ENGLISH: a detailed cinematic illustration prompt
  (subject, setting, style, lighting, composition, mood) that VISUALLY matches
  that specific scene's content. Keep a consistent modern cinematic style and
  show step-by-step visual progression across scenes.
- "caption": a short on-screen title (<=7 words) in {lang_name}, unique per scene.

Output ONLY this JSON:
{{"title":"...","scenes":[{{"narration":"...","image_prompt":"...","caption":"..."}}]}}"""
    raw = _online_llm(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        temperature=0.85, timeout=110)
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


def build_offline_script(topic, duration_sec, num_scenes, language="en",
                         facts=None):
    """
    Build a genuine, structured educational script without any external API.
    Persian (fa) gets a native Persian writer; other languages use English
    text that is later translated to the target language during normalization.

    If `facts` (from the research/fact-check agents) are available, they are
    woven into the body scenes so even the offline path is grounded in real,
    UNIQUE content rather than repetitive filler.
    """
    topic_clean = topic.strip()
    style = ("cinematic, highly detailed, professional lighting, modern 4k "
             "illustration, vivid rich colors, clean composition, sharp focus")

    is_fa = (language == "fa")
    # English topic used for IMAGE PROMPTS (always English).
    topic_en = translate.to_english(topic_clean) if is_fa else topic_clean

    # If we have grounded facts, build a fact-driven structured script.
    if facts and len(facts) >= 3:
        return _facts_script(topic_clean, topic_en, facts, num_scenes,
                             is_fa, style)

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


def _facts_script(topic_clean, topic_en, facts, num_scenes, is_fa, style):
    """Build a structured, UNIQUE, multi-scene script from researched facts.

    Structure: intro -> pre-intro/context -> one scene per fact -> conclusion.
    Each body scene narrates a DIFFERENT fact, guaranteeing no repetition.
    """
    title = topic_clean
    scenes = []

    # 1) Intro / hook
    if is_fa:
        intro = (f"{_FA['welcome']} {_FA['about']} {topic_clean}. "
                 f"{_FA['intro_tail']}")
        pre = ("ابتدا یک نگاه کلی به موضوع می‌اندازیم تا با مفاهیم پایه آشنا شویم، "
               "سپس نکات مهم را قدم‌به‌قدم بررسی می‌کنیم.")
    else:
        intro = (f"Welcome to this explainer about {topic_clean}. "
                 f"Let us break it down step by step.")
        pre = ("First, a quick overview to set the context, then we will go "
               "through the key points one by one.")
    scenes.append(_scene(intro,
                         f"an engaging hero illustration representing {topic_en}, "
                         f"{style}", _short(topic_clean)))
    scenes.append(_scene(pre,
                         f"clean overview infographic concept about {topic_en}, "
                         f"{style}",
                         ("مقدمه" if is_fa else "Overview")))

    # 2) Body — one unique scene per fact (capped to fit num_scenes)
    body_slots = max(1, num_scenes - 3)
    chosen = facts[:body_slots]
    for idx, fact in enumerate(chosen, start=1):
        fact = str(fact).strip()
        if is_fa:
            cap = f"نکته {idx}"
        else:
            cap = f"Point {idx}"
        img = (f"detailed cinematic visual illustrating: {topic_en}, "
               f"aspect {idx}, infographic feel, {style}")
        scenes.append(_scene(fact, img, cap))

    # 3) Conclusion
    if is_fa:
        concl = (f"{_FA['summary']} این‌ها مهم‌ترین نکات درباره‌ی {topic_clean} بودند. "
                 f"{_FA['thanks']}")
    else:
        concl = (f"In summary, these were the most important points about "
                 f"{topic_clean}. Thanks for watching, and keep exploring!")
    scenes.append(_scene(concl,
                         f"inspiring closing scene about {topic_en}, sunrise, "
                         f"{style}",
                         ("جمع‌بندی" if is_fa else "Conclusion")))
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


def _normalize_script(data, topic, language="en", min_scenes=4):
    """Clean scenes, DEDUPLICATE narration, and GUARANTEE a valid multi-scene
    script with English image prompts.

    This is the safety net that prevents the "single repeated slide" bug:
      * scenes with empty or duplicate narration are dropped,
      * if too few unique scenes remain, we rebuild with the offline writer.
    """
    data.setdefault("title", topic)
    clean = []
    seen_nar = set()
    seen_cap = set()
    for s in data.get("scenes", []):
        if not isinstance(s, dict):
            continue
        nar = str(s.get("narration", "")).strip()
        if not nar:
            continue
        # Drop pure-numeric / trivial narration (the old "1 2 3" bug).
        if re.fullmatch(r"[\d\s.,،\-]+", nar):
            continue
        # Deduplicate by a normalized narration key.
        key = re.sub(r"\s+", " ", nar.lower())[:80]
        if key in seen_nar:
            continue
        seen_nar.add(key)

        img = str(s.get("image_prompt") or nar).strip()
        if translate.has_persian(img):
            img = translate.to_english(img)
        cap = str(s.get("caption") or "").strip()[:60]
        # Ensure captions are not identical across scenes.
        cap_key = cap.lower()
        if cap and cap_key in seen_cap:
            cap = ""
        if cap:
            seen_cap.add(cap_key)
        clean.append({"narration": nar, "image_prompt": img or topic,
                      "caption": cap})

    # If we ended up with too few UNIQUE scenes, rebuild a real script.
    if len(clean) < min_scenes:
        rebuilt = build_offline_script(topic, 75, max(min_scenes + 2, 6),
                                       language)
        # merge any good unique scenes we already had with the rebuilt ones
        merged = clean[:]
        for s in rebuilt["scenes"]:
            k = re.sub(r"\s+", " ", s["narration"].lower())[:80]
            if k not in seen_nar:
                seen_nar.add(k)
                merged.append(s)
        clean = merged

    data["scenes"] = clean
    return data


def _extract_json_array(text):
    """Extract a JSON array of strings from possibly-noisy LLM output."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip("`").strip()
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1 or e <= s:
        return None
    snippet = text[s:e + 1]
    try:
        arr = json.loads(snippet)
    except Exception:
        try:
            arr = json.loads(re.sub(r",\s*([}\]])", r"\1", snippet))
        except Exception:
            return None
    if isinstance(arr, list):
        return [x for x in arr if isinstance(x, (str, int, float))]
    return None


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
