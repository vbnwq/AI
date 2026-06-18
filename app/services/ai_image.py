"""
ai_image.py - Free AI image generation (no API key)
Primary: Pollinations image API (flux/turbo models).
Generates high-quality images at requested resolution.
"""
import os
import time
import random
import urllib.parse
import requests

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


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
        im.save(path, quality=92)
    except Exception:
        pass

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# Available free models on pollinations
MODELS = ["flux", "turbo"]


def generate_image(prompt, out_path, width=1280, height=720,
                   model="flux", seed=None, enhance=True, max_retries=4):
    """
    Generate a single image and save it to out_path.
    Returns out_path on success, raises on failure.
    """
    if seed is None:
        seed = random.randint(1, 9_999_999)

    # Pollinations caps very large dims; keep within safe bounds then it serves requested size
    safe_w = max(64, min(width, 2048))
    safe_h = max(64, min(height, 2048))

    params = {
        "width": safe_w,
        "height": safe_h,
        "seed": seed,
        "nologo": "true",
        "model": model,
        "private": "true",
    }
    if enhance:
        params["enhance"] = "true"

    encoded_prompt = urllib.parse.quote(prompt[:1500])
    url = POLLINATIONS_URL + encoded_prompt + "?" + urllib.parse.urlencode(params)

    last_err = None
    models_to_try = [model] + [m for m in MODELS if m != model]
    for attempt in range(max_retries):
        cur_model = models_to_try[min(attempt, len(models_to_try) - 1)]
        params["model"] = cur_model
        url = POLLINATIONS_URL + encoded_prompt + "?" + urllib.parse.urlencode(params)
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code == 200 and r.content and len(r.content) > 2000 and "image" in ctype:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                # Ensure exact requested resolution (upscale/crop to fill).
                _fit_exact(out_path, width, height)
                return out_path
            elif r.status_code in (429, 502, 503, 504):
                time.sleep(3 + attempt * 3)
                continue
            else:
                last_err = f"HTTP {r.status_code} ctype={ctype} len={len(r.content)}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 + attempt * 2)
        # vary seed on retry
        params["seed"] = random.randint(1, 9_999_999)

    raise RuntimeError(f"Image generation failed: {last_err}")


def generate_image_to_url_bytes(prompt, width=1024, height=1024, model="flux", seed=None):
    """Generate image and return raw bytes (for serving directly)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.close()
    try:
        generate_image(prompt, tmp.name, width=width, height=height, model=model, seed=seed)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
