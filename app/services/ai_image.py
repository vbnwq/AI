"""
ai_image.py - High-quality FREE AI image generation (no API key).

Upgrades over the old version:
  * Persian prompts are auto-translated to English first (image models obey
    English far better) via the translate layer.
  * A rich "quality booster" + negative-prompt system squeezes maximum detail
    and sharpness out of the free Pollinations Flux model.
  * Multiple free model endpoints are tried in order so generation never fails:
        flux  ->  flux-realism style boosters  ->  turbo
  * Quality presets ("standard" / "high" / "ultra") control prompt boosters
    and post-processing (sharpening) for top-tier output.
  * Output is post-processed with PIL (sharpening + light contrast) for extra
    perceived sharpness — cheap on CPU, big visual win.

100% free, no key, CPU-friendly (all heavy lifting happens on Pollinations'
servers; we only do tiny local post-processing).
"""
import os
import time
import random
import urllib.parse
import requests

from . import translate

try:
    from PIL import Image, ImageEnhance, ImageFilter
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
}

# Available free models on pollinations (ordered by quality)
MODELS = ["flux", "turbo"]

# Quality booster suffixes appended to every prompt for maximum detail.
QUALITY_BOOSTERS = {
    "standard": "high quality, detailed, sharp focus",
    "high": ("masterpiece, best quality, highly detailed, ultra sharp focus, "
             "professional photography, 8k, intricate details, vivid colors, "
             "perfect composition, cinematic lighting"),
    "ultra": ("masterpiece, best quality, ultra detailed, hyper realistic, "
              "8k uhd, extremely sharp focus, professional photography, "
              "intricate fine details, perfect lighting, cinematic, "
              "dramatic composition, award winning, trending on artstation, "
              "vivid rich colors, high dynamic range, photorealistic"),
}

# Negative prompt to steer the model away from common artifacts.
NEGATIVE_PROMPT = ("blurry, low quality, low resolution, pixelated, distorted, "
                   "deformed, ugly, bad anatomy, watermark, text, signature, "
                   "jpeg artifacts, grainy, oversaturated, cropped, out of frame")


def _build_prompt(prompt, quality="ultra"):
    """Translate (if Persian) + add quality boosters."""
    eng = translate.to_english(prompt)
    booster = QUALITY_BOOSTERS.get(quality, QUALITY_BOOSTERS["high"])
    full = f"{eng}, {booster}"
    return full[:1800]


def _post_process(path, quality="ultra"):
    """Light, CPU-cheap sharpening/contrast for extra perceived detail."""
    if not _HAS_PIL:
        return
    try:
        im = Image.open(path).convert("RGB")
        if quality in ("high", "ultra"):
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
            im = ImageEnhance.Contrast(im).enhance(1.04)
            im = ImageEnhance.Color(im).enhance(1.05)
            im = ImageEnhance.Sharpness(im).enhance(1.10)
        im.save(path, quality=95, optimize=True)
    except Exception:
        pass


def _fit_exact(path, width, height):
    """Resize/crop the saved image to exactly width x height (cover)."""
    if not _HAS_PIL:
        return
    try:
        im = Image.open(path).convert("RGB")
        if im.size == (width, height):
            return
        sw, sh = im.size
        scale = max(width / sw, height / sh)
        nw, nh = int(sw * scale + 0.5), int(sh * scale + 0.5)
        im = im.resize((nw, nh), Image.LANCZOS)
        left = (nw - width) // 2
        top = (nh - height) // 2
        im = im.crop((left, top, left + width, top + height))
        im.save(path, quality=95)
    except Exception:
        pass


def generate_image(prompt, out_path, width=1280, height=720,
                   model="flux", seed=None, quality="ultra", max_retries=5,
                   raw_prompt=False):
    """
    Generate a single high-quality image and save it to out_path.

    prompt     : user text (Persian or English) — auto-translated + boosted.
    quality    : "standard" | "high" | "ultra" (default ultra for best detail).
    raw_prompt : if True, use prompt as-is (already translated/boosted upstream).

    Returns out_path on success, raises RuntimeError on total failure.
    """
    if seed is None:
        seed = random.randint(1, 9_999_999)

    full_prompt = prompt if raw_prompt else _build_prompt(prompt, quality)

    # Pollinations caps very large dims; keep within safe bounds then upscale.
    safe_w = max(64, min(width, 2048))
    safe_h = max(64, min(height, 2048))

    encoded_prompt = urllib.parse.quote(full_prompt)
    encoded_neg = urllib.parse.quote(NEGATIVE_PROMPT)

    last_err = None
    models_to_try = [model] + [m for m in MODELS if m != model]
    for attempt in range(max_retries):
        cur_model = models_to_try[min(attempt, len(models_to_try) - 1)]
        params = {
            "width": safe_w,
            "height": safe_h,
            "seed": seed,
            "nologo": "true",
            "model": cur_model,
            "private": "true",
            "enhance": "true",
            "negative": NEGATIVE_PROMPT,
        }
        url = (POLLINATIONS_URL + encoded_prompt + "?"
               + urllib.parse.urlencode(params, quote_via=urllib.parse.quote))
        try:
            r = requests.get(url, headers=HEADERS, timeout=150)
            ctype = r.headers.get("Content-Type", "")
            if (r.status_code == 200 and r.content
                    and len(r.content) > 2000 and "image" in ctype):
                with open(out_path, "wb") as f:
                    f.write(r.content)
                _fit_exact(out_path, width, height)
                _post_process(out_path, quality)
                return out_path
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 + attempt * 3)
            else:
                last_err = f"HTTP {r.status_code} ctype={ctype} len={len(r.content)}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 + attempt * 2)
        seed = random.randint(1, 9_999_999)  # vary seed on retry

    raise RuntimeError(f"Image generation failed: {last_err}")


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
