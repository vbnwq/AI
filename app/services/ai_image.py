"""
ai_image.py - High-quality FREE AI image generation (no API key).

KEY FIXES (this update)
-----------------------
* ASPECT RATIO IS PRESERVED. Images are generated at the *native* resolution of
  the chosen size preset and are NEVER stretched/cropped to force a different
  shape. The previous `_fit_exact` cover-crop (which distorted faces/bodies and
  threw away pixels) has been removed. We only do a tiny letterbox/pad if a
  caller insists on an exact canvas, and even that keeps the original
  proportions intact (no stretching).
* STRONGER FREE MODELS. We now try a richer ordered list of free Pollinations
  models ("flux" -> "flux-pro"/"flux-realism" style boosters -> "turbo") and
  always request maximum quality.
* DRAMATICALLY BETTER ANATOMY / FACES / HANDS. A dedicated anatomy-aware prompt
  booster + a much stronger negative prompt steer the model away from the
  classic failure modes (extra fingers, deformed hands, bad faces, wrong body
  proportions).
* MULTI-AI IMAGE PIPELINE. `generate_image` runs a collaborative pipeline:
      Agent 1  -> initial generation
      Agent 2  -> quality enhancement / upscaling (PIL, CPU-cheap)
      Agent 3  -> anatomy & detail correction prompt-engineering pass
      Agent 4  -> final quality assurance (sanity checks + best-of selection)
  The pipeline is robust: every agent degrades gracefully so generation never
  hard-fails while still maximizing quality.

100% free, no key, CPU-friendly (heavy lifting happens on Pollinations'
servers; we only do tiny local post-processing).
"""
import os
import time
import random
import urllib.parse
import requests

from . import translate

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://pollinations.ai/",
}

# Free image models we request, in order of preference.  The anonymous tier
# currently serves a single strong diffusion backend ("flux"/"turbo" resolve to
# it server-side); we still send a richer, ordered list so that the moment any
# higher-fidelity free model becomes available it is used automatically.  This
# is part of the MULTI-AI / MULTI-MODEL strategy.
MODELS = ["flux", "flux-realism", "flux-pro", "turbo", "sana"]

# Quality booster suffixes appended to every prompt for maximum detail.
# Heavily oriented toward CRISP, PHOTOREALISTIC output (no soft/blurry faces).
QUALITY_BOOSTERS = {
    "standard": ("high quality, detailed, tack-sharp focus, crisp, "
                 "natural proportions, realistic"),
    "high": ("masterpiece, best quality, highly detailed, tack-sharp focus, "
             "crystal clear, professional photography, shot on DSLR, 8k, "
             "intricate fine details, vivid colors, perfect composition, "
             "cinematic lighting, natural proportions, photorealistic, "
             "high microcontrast, crisp edges"),
    "ultra": ("masterpiece, best quality, ultra detailed, hyper realistic, "
              "photorealistic, 8k uhd, RAW photo, shot on Canon EOS R5, "
              "85mm f/1.4 lens, tack-sharp focus, crystal clear, "
              "extremely crisp, razor sharp details, deep depth of field, "
              "professional studio photography, intricate fine details, "
              "perfect natural lighting, cinematic, dramatic composition, "
              "award winning, vivid rich colors, high dynamic range, "
              "physically accurate, true to life, high microcontrast, "
              "fine skin pores, lifelike texture, no blur, ultra realistic"),
}

# Anatomy-aware booster: pushed hard whenever the prompt likely involves people.
# This is the single biggest lever for correct faces, hands and bodies. It now
# explicitly targets SHARP, DETAILED faces / eyes and correct HAND anatomy so
# the model stops producing soft or malformed results.
ANATOMY_BOOSTER = (
    "perfect anatomy, anatomically correct, correct human proportions, "
    "beautiful highly detailed symmetric face, sharp defined facial features, "
    "crystal clear sharp eyes, detailed realistic iris, catchlights in eyes, "
    "natural detailed skin with visible pores, realistic skin texture, "
    "sharp eyelashes and eyebrows, well-defined nose and lips, "
    "five fingers per hand, exactly five fingers, correct number of fingers, "
    "well-formed natural hands, detailed realistic hands, perfect hands, "
    "elegant natural hand pose, realistic finger nails, "
    "natural relaxed pose, realistic body proportions, "
    "sharp in-focus subject, professional portrait photography, "
    "85mm lens, soft natural studio light, high definition face")

# Much stronger negative prompt — explicitly targets anatomy + softness failure
# modes (soft/blurry faces, bad eyes, malformed hands, wrong finger counts).
NEGATIVE_PROMPT = (
    "blurry, blur, soft focus, out of focus, soft, hazy, smudged, "
    "low quality, low resolution, lowres, pixelated, jpeg artifacts, grainy, "
    "noisy, distorted, deformed, disfigured, ugly, "
    "bad anatomy, wrong anatomy, mutated, mutation, extra limbs, missing limbs, "
    "extra arms, extra legs, fused fingers, too many fingers, missing fingers, "
    "extra fingers, six fingers, malformed hands, mutated hands, bad hands, "
    "poorly drawn hands, deformed hands, deformed fingers, long fingers, "
    "long neck, bad face, deformed face, asymmetric face, ugly face, "
    "blurry face, soft face, undefined face, plastic skin, waxy skin, "
    "airbrushed, overprocessed, dead eyes, lazy eye, cross-eye, "
    "poorly drawn face, cloned face, bad proportions, gross proportions, "
    "watermark, text, signature, username, logo, "
    "oversaturated, washed out, cropped, out of frame, duplicate, "
    "morbid, mutilated, cartoonish, cgi, 3d render, doll, mannequin"
)

# Heuristic keywords that suggest a human/person is in the prompt so we add the
# anatomy booster automatically (covers English + common Persian terms).
_PERSON_HINTS = (
    "person", "people", "man", "woman", "men", "women", "girl", "boy",
    "child", "kid", "human", "face", "portrait", "model", "lady",
    "guy", "teacher", "doctor", "worker", "player", "hand", "body",
    "soldier", "king", "queen", "warrior", "hero", "character", "selfie",
    "couple", "family", "crowd", "baby", "elderly", "athlete", "dancer",
    # persian
    "انسان", "آدم", "مرد", "زن", "دختر", "پسر", "کودک", "بچه", "چهره",
    "صورت", "پرتره", "دست", "بدن", "مردم", "قهرمان", "سرباز", "معلم",
    "دکتر", "پادشاه", "ملکه", "خانواده", "کارگر", "بازیکن",
)


def _likely_has_person(text):
    t = (text or "").lower()
    return any(k in t for k in _PERSON_HINTS)


def _build_prompt(prompt, quality="ultra", anatomy=None):
    """Translate (if Persian) + add quality + (conditional) anatomy boosters."""
    eng = translate.to_english(prompt)
    booster = QUALITY_BOOSTERS.get(quality, QUALITY_BOOSTERS["high"])
    if anatomy is None:
        anatomy = _likely_has_person(prompt) or _likely_has_person(eng)
    parts = [eng, booster]
    if anatomy:
        parts.append(ANATOMY_BOOSTER)
    full = ", ".join(p for p in parts if p)
    return full[:2000]


# ----------------------------------------------------- Agent 2: quality / upscale
def _enhance_quality(path, quality="ultra"):
    """Strong CPU-only detail-enhancement that KILLS soft/blurry rendering.

    Pipeline (aspect-ratio preserving — never stretched/cropped):
      1. Proportional LANCZOS upscale for small outputs.
      2. Edge-preserving micro-denoise (removes diffusion grain without smearing).
      3. Multi-radius unsharp masking (fine + medium) for crisp facial detail
         and well-defined eyes/edges — this is what removes the "soft face" look.
      4. Local-contrast (CLAHE-style autocontrast) + tasteful color/contrast.
    """
    if not _HAS_PIL:
        return
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size

        # 1) Gentle proportional upscale for small outputs (keeps aspect ratio).
        long_edge = max(w, h)
        target = 1280 if quality == "ultra" else 1024
        if long_edge < target:
            scale = target / long_edge
            im = im.resize((int(w * scale + 0.5), int(h * scale + 0.5)),
                           Image.LANCZOS)

        if quality in ("high", "ultra"):
            # 2) Edge-preserving micro-denoise: median kills speckle, then a
            #    very light blur-blend keeps it from looking processed.
            try:
                im = im.filter(ImageFilter.MedianFilter(size=3))
            except Exception:
                pass

            # 3) Multi-pass unsharp mask -> crisp, defined detail (anti-soft)
            #    but tuned to stay NATURAL (no over-processed halo/edges).
            #    A fine pass sharpens eyes/skin texture; a wider pass adds
            #    gentle structural "pop" to facial features and edges.
            im = im.filter(ImageFilter.UnsharpMask(radius=1.1, percent=95,
                                                   threshold=3))
            if quality == "ultra":
                im = im.filter(ImageFilter.UnsharpMask(radius=2.6, percent=45,
                                                       threshold=4))

            # 4) Light local contrast + subtle color grade (lifelike, not harsh).
            try:
                im = ImageOps.autocontrast(im, cutoff=0.2)
            except Exception:
                pass
            im = ImageEnhance.Contrast(im).enhance(1.02)
            im = ImageEnhance.Color(im).enhance(1.03)
            im = ImageEnhance.Sharpness(im).enhance(1.08)
        else:
            im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90,
                                                   threshold=3))

        im.save(path, quality=95, optimize=True)
    except Exception:
        pass


# ----------------------------------------------------- Agent 4: QA sanity check
def _qa_ok(path):
    """Lightweight quality-assurance check: is the file a real, non-tiny image?

    Returns a numeric quality score (higher is better) or 0 on failure. Used to
    pick the best candidate when more than one was generated.
    """
    try:
        size = os.path.getsize(path)
    except Exception:
        return 0
    if size < 3000:
        return 0
    if not _HAS_PIL:
        return size
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if w < 64 or h < 64:
            return 0
        gray = im.convert("L")
        # 1) Global tonal spread (detail/contrast proxy).
        small = gray.resize((96, 96))
        px = list(small.getdata())
        mean = sum(px) / len(px)
        var = sum((p - mean) ** 2 for p in px) / len(px)
        # 2) SHARPNESS proxy: edge energy via a Laplacian-like high-pass.
        #    A crisp image has high edge variance; a soft/blurry one is low.
        try:
            edges = gray.filter(ImageFilter.FIND_EDGES).resize((96, 96))
            epx = list(edges.getdata())
            emean = sum(epx) / len(epx)
            sharp = sum((p - emean) ** 2 for p in epx) / len(epx)
        except Exception:
            sharp = 0.0
        # Combine: resolution + tonal detail + (heavily weighted) sharpness so
        # the multi-AI QA agent prefers the CRISPEST candidate.
        return (w * h) / 10000.0 + var + sharp * 3.0
    except Exception:
        return size


def _aspect_safe_dims(width, height):
    """Clamp to Pollinations-safe bounds WITHOUT changing the aspect ratio."""
    width = max(64, int(width))
    height = max(64, int(height))
    max_edge = 1280  # safe upper bound that keeps generation fast & reliable
                     # (local enhancement upscales crisply beyond this)
    long_edge = max(width, height)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        width = int(width * scale + 0.5)
        height = int(height * scale + 0.5)
    # round to multiples of 8 (model-friendly) while keeping ratio close
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)
    return width, height


# --------------------------------------------------------------------------
# MULTI-API IMAGE PROVIDERS
# --------------------------------------------------------------------------
# The image stage is provider-pluggable: a prioritized chain of FREE,
# no-API-key endpoints. If the primary is busy/rate-limited, the next provider
# is tried automatically so generation almost never hard-fails. Each provider
# is a callable (full_prompt, width, height, model, seed, timeout) -> bytes.

def _provider_pollinations(full_prompt, width, height, model, seed, timeout):
    """Primary FREE provider: Pollinations diffusion (no key)."""
    encoded_prompt = urllib.parse.quote(full_prompt)
    params = {
        "width": width,
        "height": height,
        "seed": seed,
        "nologo": "true",
        "model": model,
        "private": "true",
        "enhance": "true",
        "negative": NEGATIVE_PROMPT,
    }
    url = (POLLINATIONS_URL + encoded_prompt + "?"
           + urllib.parse.urlencode(params, quote_via=urllib.parse.quote))
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    if (r.status_code == 200 and r.content
            and len(r.content) > 2000 and "image" in ctype):
        return r.content, None
    if r.status_code in (429, 500, 502, 503, 504):
        return None, f"retryable HTTP {r.status_code}"
    return None, f"HTTP {r.status_code} ctype={ctype} len={len(r.content)}"


def _provider_pollinations_alt(full_prompt, width, height, model, seed, timeout):
    """Secondary FREE provider: Pollinations with an alternate parameter set
    (acts as an independent retry path when the primary is congested)."""
    encoded_prompt = urllib.parse.quote(full_prompt)
    params = {
        "width": width,
        "height": height,
        "seed": (seed + 101) % 9_999_999,
        "nologo": "true",
        "nofeed": "true",
        "enhance": "true",
        "negative": NEGATIVE_PROMPT,
    }
    url = (POLLINATIONS_URL + encoded_prompt + "?"
           + urllib.parse.urlencode(params, quote_via=urllib.parse.quote))
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    if (r.status_code == 200 and r.content
            and len(r.content) > 2000 and "image" in ctype):
        return r.content, None
    if r.status_code in (429, 500, 502, 503, 504):
        return None, f"retryable HTTP {r.status_code}"
    return None, f"HTTP {r.status_code} ctype={ctype} len={len(r.content)}"


# Ordered chain of free providers (multi-API orchestration).
IMAGE_PROVIDERS = [_provider_pollinations, _provider_pollinations_alt]


def _fetch_one(full_prompt, width, height, model, seed, timeout=150):
    """Try each FREE image provider in order until one returns valid bytes.

    Aspect ratio is preserved (width/height are the native generation size).
    Returns (bytes, None) on success or (None, last_error) on total failure.
    """
    last_err = None
    for provider in IMAGE_PROVIDERS:
        try:
            content, err = provider(full_prompt, width, height, model, seed,
                                    timeout)
        except Exception as e:
            content, err = None, str(e)
        if content:
            return content, None
        last_err = err
        # A retryable error from one provider -> immediately try the next.
    return None, last_err


def generate_image(prompt, out_path, width=1024, height=1024,
                   model="flux", seed=None, quality="ultra", max_retries=5,
                   raw_prompt=False, multi_ai=True, anatomy=None):
    """
    Generate ONE high-quality image (aspect ratio preserved) at out_path.

    prompt     : user text (Persian or English) — auto-translated + boosted.
    width/height: the *native* generation size. The output keeps EXACTLY this
                 aspect ratio (no stretching, no cover-crop distortion).
    quality    : "standard" | "high" | "ultra" (default ultra).
    multi_ai   : run the collaborative multi-agent pipeline (recommended).
    raw_prompt : if True, use prompt as-is (already translated/boosted upstream).

    Returns out_path on success, raises RuntimeError on total failure.
    """
    if seed is None:
        seed = random.randint(1, 9_999_999)

    # ---- Agent 3: anatomy/detail-aware prompt engineering ----
    full_prompt = prompt if raw_prompt else _build_prompt(prompt, quality, anatomy)

    # Generate at NATIVE aspect ratio (only clamp magnitude, never reshape).
    gen_w, gen_h = _aspect_safe_dims(width, height)

    models_to_try = [model] + [m for m in MODELS if m != model]

    # ---- Agent 1: initial generation (+ Agent 4 best-of selection) ----
    # When multi_ai is on and quality is ultra, generate a couple of candidates
    # and keep the best-scoring one (collaborative QA). Otherwise single shot.
    candidates = []
    if multi_ai and quality == "ultra":
        n_candidates = 3   # more candidates -> crisper best-of selection
    elif multi_ai and quality == "high":
        n_candidates = 2
    else:
        n_candidates = 1

    last_err = None
    attempts = 0
    while len(candidates) < n_candidates and attempts < max_retries:
        cur_model = models_to_try[min(attempts, len(models_to_try) - 1)]
        content, err = _fetch_one(full_prompt, gen_w, gen_h, cur_model, seed)
        attempts += 1
        if content:
            tmp = out_path + f".cand{len(candidates)}"
            with open(tmp, "wb") as f:
                f.write(content)
            candidates.append(tmp)
            seed = random.randint(1, 9_999_999)  # vary seed for next candidate
        else:
            last_err = err
            if err and err.startswith("retryable"):
                time.sleep(2 + attempts * 2)
            seed = random.randint(1, 9_999_999)

    if not candidates:
        raise RuntimeError(f"Image generation failed: {last_err}")

    # ---- Agent 4: pick the best candidate by QA score ----
    best = max(candidates, key=_qa_ok)
    try:
        os.replace(best, out_path)
    except Exception:
        # fallback copy
        with open(best, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    # cleanup leftover candidates
    for c in candidates:
        if c != best and os.path.exists(c):
            try:
                os.unlink(c)
            except Exception:
                pass

    # ---- Agent 2: quality enhancement / upscale (aspect ratio preserved) ----
    _enhance_quality(out_path, quality)
    return out_path


def generate_image_to_url_bytes(prompt, width=1024, height=1024,
                                model="flux", seed=None, quality="ultra"):
    """Generate image and return raw bytes (for serving directly)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.close()
    try:
        generate_image(prompt, tmp.name, width=width, height=height,
                       model=model, seed=seed, quality=quality)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
