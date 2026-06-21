"""
video_engine.py - Assemble a full educational video from a script.

Pipeline (100% local FFmpeg + PIL, no GPU needed, CPU-optimized):
  1. For each scene: generate image (ai_image) + narration audio (ai_tts) in
     parallel (network/IO-bound).
  2. Render a HIGH-QUALITY caption overlay PNG with PIL — proper Persian font,
     full RTL shaping/reordering, word-wrap, soft shadow + rounded bar. This
     beats ffmpeg's drawtext for Persian/Arabic readability.
  3. Apply a gentle Ken Burns (pan/zoom) animation to each still and overlay
     the caption PNG -> one scene clip whose length matches its narration.
  4. Concatenate all scene clips (stream-copy, near-instant) and mux.
  5. Export at the requested resolution & quality.

CPU optimization:
  * zoompan runs at the TARGET size (not a giant canvas).
  * Quality presets map to x264 preset/crf so low-end CPUs stay responsive
    while still producing crisp output.
  * Caption is pre-rendered ONCE per scene as a PNG (no per-frame text work).
"""
import os
import math
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import ai_image, ai_tts

RESOLUTIONS = {
    "854x480":   (854, 480),
    "1280x720":  (1280, 720),
    "1920x1080": (1920, 1080),
    "1080x1920": (1080, 1920),
    "720x1280":  (720, 1280),
}

# Quality presets -> x264 settings + image quality preset.
# Tuned so that even a GPU-less low-end CPU stays usable.
QUALITY_PRESETS = {
    "fast":     {"x264": "ultrafast", "crf": "26", "img": "high",  "fps": 24},
    "balanced": {"x264": "veryfast",  "crf": "22", "img": "ultra", "fps": 25},
    "high":     {"x264": "faster",    "crf": "19", "img": "ultra", "fps": 30},
}

_RTL_LANGS = {"fa", "ar", "ur", "he", "ps"}

# Does this Pillow build have raqm (HarfBuzz) complex-text layout? If YES, PIL
# shapes + bidi-reorders Arabic/Persian correctly ON ITS OWN, so we must pass
# RAW text (manual reshape+bidi would double-process and BREAK the glyphs).
# If NO (some minimal/Windows builds), we fall back to manual reshape+bidi.
try:
    from PIL import features as _pil_features
    _HAS_RAQM = bool(_pil_features.check("raqm"))
except Exception:
    _HAS_RAQM = False

# ---------------------------------------------------------------------------
# PROFESSIONAL TEMPLATES
# ---------------------------------------------------------------------------
# Each template defines a color theme + layout. Layouts:
#   "cinematic"  -> full-bleed image with gradient + lower caption bar (Ken Burns)
#   "card"       -> image framed inside a rounded card on a themed panel,
#                   with a title strip on top and caption area below (no crop
#                   distortion of the image — it's fitted with aspect preserved)
#   "sidebar"    -> image on one side, text column on the other (great for RTL)
# Templates are auto-rotated across scenes for a dynamic, non-monotone feel.
TEMPLATES = {
    "midnight": {"layout": "cinematic",
                 "bg": (10, 14, 30), "bg2": (26, 22, 60),
                 "accent": (0, 210, 255), "title_bg": (8, 12, 26, 200),
                 "text": (255, 255, 255)},
    "aurora":   {"layout": "card",
                 "bg": (12, 18, 40), "bg2": (40, 18, 70),
                 "accent": (124, 92, 255), "title_bg": (16, 12, 40, 220),
                 "text": (255, 255, 255)},
    "sunset":   {"layout": "sidebar",
                 "bg": (24, 14, 30), "bg2": (60, 22, 40),
                 "accent": (255, 107, 157), "title_bg": (30, 12, 24, 220),
                 "text": (255, 255, 255)},
    "ocean":    {"layout": "card",
                 "bg": (8, 22, 34), "bg2": (12, 44, 60),
                 "accent": (0, 200, 180), "title_bg": (8, 22, 34, 220),
                 "text": (255, 255, 255)},
}
_TEMPLATE_ORDER = ["midnight", "aurora", "sunset", "ocean"]

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_RTL = True
except Exception:
    _HAS_RTL = False


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg error:\nCMD: " + " ".join(cmd) +
            "\nSTDERR:\n" + proc.stderr[-1500:])
    return proc


def _font_dir():
    base = getattr(__import__("sys"), "_MEIPASS", None)
    if base:
        p = os.path.join(base, "app", "fonts")
        if os.path.isdir(p):
            return p
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fonts")


_FONT_DIR = _font_dir()


def _find_font(language="en"):
    """Pick a TTF that supports the target language."""
    if language in _RTL_LANGS:
        cands = [
            os.path.join(_FONT_DIR, "Vazirmatn-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        cands = [
            os.path.join(_FONT_DIR, "Latin-Bold.ttf"),
            os.path.join(_FONT_DIR, "Vazirmatn-Bold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _shape_rtl(text, language="en"):
    """Prepare Arabic/Persian text for correct PIL rendering.

    * If Pillow has raqm/HarfBuzz (`_HAS_RAQM`), return the RAW text — PIL shapes
      and reorders it correctly by itself. (Manual reshaping here would
      double-process and produce broken, disconnected, reversed glyphs.)
    * Otherwise, fall back to manual arabic_reshaper + bidi so older/minimal
      Pillow builds still render readable RTL text.
    """
    if not text or language not in _RTL_LANGS:
        return text
    if _HAS_RAQM:
        return text
    if _HAS_RTL:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


# --------------------------------------------------- PIL caption overlay
def _wrap_text_pixels(draw, text, font, max_width, is_rtl=False):
    """Word-wrap by measured pixel width (works for any language)."""
    words = text.split()
    if not words:
        return [text]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _measure(draw, trial, font, is_rtl) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _render_caption_png(caption, w, h, language, out_png):
    """
    Render a beautiful caption bar to a transparent PNG (w x h).
    Returns out_png if a caption was drawn, else None.

    Persian/Arabic text is reshaped + bidi-reordered and RIGHT-aligned (RTL).
    """
    caption = (caption or "").strip()
    if not caption:
        return None

    fontpath = _find_font(language)
    if not fontpath:
        return None

    is_rtl = language in _RTL_LANGS
    fontsize = max(30, int(h * 0.050))
    try:
        font = ImageFont.truetype(fontpath, fontsize)
    except Exception:
        return None

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    side_pad = int(w * 0.05)
    max_text_w = w - 2 * side_pad

    # Wrap on the ORIGINAL text first, then shape each line (so joining works).
    raw_lines = _wrap_text_pixels(draw, caption, font, max_text_w, is_rtl)
    shaped_lines = [_shape_rtl(ln, language) for ln in raw_lines]

    # Measure block height.
    line_h = int(fontsize * 1.45)
    block_h = line_h * len(shaped_lines)
    bar_pad = int(h * 0.028)
    bar_h = block_h + 2 * bar_pad
    bar_top = h - bar_h - int(h * 0.04)

    # Gradient/translucent rounded bar.
    bar = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bar)
    radius = int(bar_h * 0.18)
    bdraw.rounded_rectangle(
        [side_pad // 2, 0, w - side_pad // 2, bar_h],
        radius=radius, fill=(8, 12, 26, 175))
    # subtle accent top border line
    bdraw.rounded_rectangle(
        [side_pad // 2, 0, w - side_pad // 2, bar_h],
        radius=radius, outline=(0, 210, 255, 90), width=max(2, int(h * 0.003)))
    img.alpha_composite(bar, (0, bar_top))

    # Draw each line (with soft shadow), aligned per direction.
    kw = _text_kwargs(is_rtl)
    y = bar_top + bar_pad
    for line in shaped_lines:
        tw = _measure(draw, line, font, is_rtl)
        if is_rtl:
            x = w - side_pad - tw          # right aligned
        else:
            x = (w - tw) // 2              # centered
        # shadow
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 200), **kw)
        # main text
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), **kw)
        y += line_h

    img.save(out_png)
    return out_png


def _title_png(title, w, h, language, out_png):
    """Render an opening title card overlay (top of frame, big)."""
    title = (title or "").strip()
    if not title:
        return None
    fontpath = _find_font(language)
    if not fontpath:
        return None
    is_rtl = language in _RTL_LANGS
    fontsize = max(40, int(h * 0.072))
    try:
        font = ImageFont.truetype(fontpath, fontsize)
    except Exception:
        return None
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    side_pad = int(w * 0.06)
    raw_lines = _wrap_text_pixels(draw, title, font, w - 2 * side_pad, is_rtl)
    shaped = [_shape_rtl(ln, language) for ln in raw_lines]
    line_h = int(fontsize * 1.35)
    kw = _text_kwargs(is_rtl)
    y = int(h * 0.10)
    for line in shaped:
        tw = _measure(draw, line, font, is_rtl)
        x = (w - tw) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0, 210), **kw)
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), **kw)
        y += line_h
    img.save(out_png)
    return out_png


# --------------------------------------------------- designed slide composer
def _gradient_bg(w, h, c1, c2):
    """Vertical gradient background image."""
    bg = Image.new("RGB", (w, h), c1)
    top = Image.new("RGB", (w, h), c2)
    mask = Image.new("L", (w, h))
    md = mask.load()
    for y in range(h):
        v = int(255 * (y / max(1, h - 1)))
        for x in range(0, w, 1):
            md[x, y] = v
    bg = Image.composite(top, bg, mask)
    return bg


def _fit_within(im, box_w, box_h):
    """Resize image to fit INSIDE box (aspect ratio preserved, no crop/stretch)."""
    sw, sh = im.size
    scale = min(box_w / sw, box_h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    return im.resize((nw, nh), Image.LANCZOS)


def _rounded(im, radius):
    """Apply rounded corners to an RGB image -> RGBA."""
    im = im.convert("RGBA")
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, im.size[0], im.size[1]], radius=radius, fill=255)
    im.putalpha(mask)
    return im


def _text_kwargs(is_rtl):
    """Extra PIL draw kwargs for correct complex-text shaping when raqm exists."""
    if is_rtl and _HAS_RAQM:
        return {"direction": "rtl"}
    return {}


def _measure(draw, text, font, is_rtl=False):
    try:
        bbox = draw.textbbox((0, 0), text, font=font, **_text_kwargs(is_rtl))
        return bbox[2] - bbox[0]
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def _draw_text_block(draw, lines, font, area, color, align="right",
                     line_h=None, shadow=True, is_rtl=False):
    """Draw wrapped lines inside area=(x0,y0,x1,y1) with correct RTL shaping."""
    x0, y0, x1, y1 = area
    if line_h is None:
        line_h = int(font.size * 1.5)
    kw = _text_kwargs(is_rtl)
    y = y0
    for line in lines:
        tw = _measure(draw, line, font, is_rtl)
        if align == "right":
            x = x1 - tw
        elif align == "center":
            x = x0 + (x1 - x0 - tw) // 2
        else:
            x = x0
        if shadow:
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 200), **kw)
        draw.text((x, y), line, font=font, fill=color, **kw)
        y += line_h
        if y > y1:
            break
    return y


def compose_slide(image_path, w, h, title, caption, language, template,
                  out_png):
    """Compose a fully-designed professional slide PNG (no image distortion).

    Returns out_png. The source AI image is fitted with its aspect ratio
    PRESERVED inside the layout's designated image area (letterboxed, never
    stretched/cropped). Title + caption are placed in dedicated zones with
    proper RTL Persian shaping.
    """
    theme = TEMPLATES.get(template, TEMPLATES["midnight"])
    layout = theme["layout"]
    accent = theme["accent"]
    text_color = theme["text"] + (255,) if len(theme["text"]) == 3 else theme["text"]
    is_rtl = language in _RTL_LANGS
    fontpath = _find_font(language)

    canvas = _gradient_bg(w, h, theme["bg"], theme["bg2"]).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    try:
        src = Image.open(image_path).convert("RGB")
    except Exception:
        src = Image.new("RGB", (w, h), theme["bg"])

    pad = int(min(w, h) * 0.045)
    title_fs = max(28, int(h * 0.055))
    cap_fs = max(24, int(h * 0.042))
    try:
        title_font = ImageFont.truetype(fontpath, title_fs) if fontpath else None
        cap_font = ImageFont.truetype(fontpath, cap_fs) if fontpath else None
    except Exception:
        title_font = cap_font = None

    def place_image_card(area, radius):
        x0, y0, x1, y1 = area
        bw, bh = x1 - x0, y1 - y0
        fitted = _fit_within(src, bw, bh)
        # subtle white frame backing
        fw, fh = fitted.size
        fx = x0 + (bw - fw) // 2
        fy = y0 + (bh - fh) // 2
        # shadow panel
        shadow = Image.new("RGBA", (fw + 24, fh + 24), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle([0, 0, fw + 24, fh + 24], radius=radius + 6,
                             fill=(0, 0, 0, 120))
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))
        canvas.alpha_composite(shadow, (fx - 12, fy - 12))
        rounded = _rounded(fitted, radius)
        canvas.alpha_composite(rounded, (fx, fy))
        # accent border
        bd = ImageDraw.Draw(canvas)
        bd.rounded_rectangle([fx, fy, fx + fw, fy + fh], radius=radius,
                             outline=accent + (160,), width=max(2, int(h * 0.004)))

    # ---- layouts ----
    if layout == "cinematic":
        # full-bleed cover for cinematic look (intentional framing, not a
        # distortion of a preview — Ken Burns handles motion later). We fit by
        # COVER but keep proportions; the crop is a deliberate cinematic frame.
        fitted = _fit_within(src, w, h)
        fw, fh = fitted.size
        if fw < w or fh < h:
            # scale up proportionally to cover, center-crop softly
            scale = max(w / src.size[0], h / src.size[1])
            cov = src.resize((int(src.size[0] * scale), int(src.size[1] * scale)),
                             Image.LANCZOS)
            left = (cov.size[0] - w) // 2
            top = (cov.size[1] - h) // 2
            fitted = cov.crop((left, top, left + w, top + h))
            fw, fh = fitted.size
        canvas.alpha_composite(fitted.convert("RGBA"),
                               ((w - fw) // 2, (h - fh) // 2))
        # bottom gradient scrim for text legibility
        scrim = Image.new("RGBA", (w, int(h * 0.42)), (0, 0, 0, 0))
        sd = ImageDraw.Draw(scrim)
        for i in range(scrim.size[1]):
            a = int(200 * (i / scrim.size[1]))
            sd.line([(0, i), (w, i)], fill=(0, 0, 0, a))
        canvas.alpha_composite(scrim, (0, h - scrim.size[1]))
        title_area = (pad, int(h * 0.05), w - pad, int(h * 0.05) + title_fs * 3)
        cap_area = (pad, h - int(h * 0.20), w - pad, h - pad)

    elif layout == "sidebar":
        # image on the (RTL: left) side, text column on the other side
        img_w = int(w * 0.52)
        if is_rtl:
            img_area = (pad, int(h * 0.16), img_w, h - pad)
            text_x0, text_x1 = img_w + pad, w - pad
        else:
            img_area = (w - img_w, int(h * 0.16), w - pad, h - pad)
            text_x0, text_x1 = pad, w - img_w - pad
        place_image_card(img_area, radius=int(h * 0.03))
        title_area = (pad, int(h * 0.05), w - pad, int(h * 0.05) + title_fs * 2)
        cap_area = (text_x0, int(h * 0.30), text_x1, h - pad)

    else:  # "card"
        # title strip on top, image card in the middle, caption below
        title_area = (pad, int(h * 0.045), w - pad, int(h * 0.045) + title_fs * 2)
        img_top = int(h * 0.20)
        img_bottom = int(h * 0.74)
        place_image_card((pad * 2, img_top, w - pad * 2, img_bottom),
                         radius=int(h * 0.03))
        cap_area = (pad, img_bottom + int(h * 0.02), w - pad, h - pad)

    # ---- accent header bar ----
    ImageDraw.Draw(canvas).rectangle([0, 0, w, max(4, int(h * 0.012))],
                                     fill=accent + (255,))

    # ---- draw title ----
    if title and title_font:
        raw = _wrap_text_pixels(draw, title.strip(), title_font,
                                title_area[2] - title_area[0], is_rtl)
        shaped = [_shape_rtl(ln, language) for ln in raw][:2]
        _draw_text_block(draw, shaped, title_font, title_area, accent + (255,),
                         align=("right" if is_rtl else "left"),
                         line_h=int(title_fs * 1.4), is_rtl=is_rtl)

    # ---- draw caption ----
    if caption and cap_font:
        raw = _wrap_text_pixels(draw, caption.strip(), cap_font,
                                cap_area[2] - cap_area[0], is_rtl)
        shaped = [_shape_rtl(ln, language) for ln in raw][:3]
        align = "right" if is_rtl else "left"
        if layout == "cinematic":
            align = "right" if is_rtl else "center"
        _draw_text_block(draw, shaped, cap_font, cap_area, text_color,
                         align=align, line_h=int(cap_fs * 1.45), is_rtl=is_rtl)

    canvas.convert("RGB").save(out_png, quality=95)
    return out_png


# --------------------------------------------------- scene clip
def _make_scene_clip(image_path, audio_path, out_clip, w, h, caption,
                     language, workdir, idx, qp, title_overlay=None,
                     template="midnight", title=""):
    """Create one animated scene clip matching the narration audio.

    Two rendering modes depending on the template layout:
      * "cinematic"  -> full-bleed Ken Burns + caption bar overlay (legacy look)
      * "card"/"sidebar" -> a fully DESIGNED slide (image fitted without
        distortion, title + caption placed in dedicated zones) with a subtle
        slow zoom so it still feels alive.
    """
    duration = ai_tts.get_audio_duration(audio_path)
    if duration <= 0.1:
        duration = 4.0
    duration = max(2.0, duration + 0.45)  # small tail padding

    fps = qp["fps"]
    total_frames = max(1, int(duration * fps))

    layout = TEMPLATES.get(template, TEMPLATES["midnight"])["layout"]

    # ---- Designed-slide path (card / sidebar): no image distortion ----
    if layout in ("card", "sidebar"):
        slide_png = os.path.join(workdir, f"slide_{idx:03d}.png")
        compose_slide(image_path, w, h, title, caption, language, template,
                      slide_png)
        zoom_end = 1.05
        zinc = (zoom_end - 1.0) / max(1, total_frames)
        zoompan = (f"zoompan=z='min(zoom+{zinc:.6f},{zoom_end})':d={total_frames}:"
                   f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}")
        vf_chain = f"scale={w}:{h}," + zoompan + ",format=yuv420p"
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", slide_png, "-i", audio_path,
               "-vf", vf_chain, "-t", f"{duration:.3f}", "-r", str(fps),
               "-c:v", "libx264", "-preset", qp["x264"], "-crf", qp["crf"],
               "-pix_fmt", "yuv420p",
               "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
               "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
               "-shortest", out_clip]
        _run(cmd)
        return out_clip, duration

    # ---- Cinematic path (full-bleed Ken Burns + caption overlay) ----
    zoom_end = 1.10
    zinc = (zoom_end - 1.0) / max(1, total_frames)

    scale_cover = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                   f"crop={w}:{h}")
    zoompan = (f"zoompan=z='min(zoom+{zinc:.6f},{zoom_end})':d={total_frames}:"
               f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}")
    vf_chain = scale_cover + "," + zoompan + ",format=yuv420p"

    # Render caption overlay PNG (and optional title overlay) with PIL.
    cap_png = os.path.join(workdir, f"cap_{idx:03d}.png")
    has_cap = _render_caption_png(caption, w, h, language, cap_png)

    inputs = ["-loop", "1", "-i", image_path, "-i", audio_path]
    filter_complex = None
    map_args = []

    overlays = []
    if has_cap:
        overlays.append(cap_png)
    if title_overlay and os.path.exists(title_overlay):
        overlays.append(title_overlay)

    if overlays:
        # build: [0:v] kenburns [bg]; [bg][1..]overlay chain
        ov_inputs = []
        for ov in overlays:
            inputs += ["-i", ov]
        # image is input 0, audio input 1, overlays start at input 2
        fc = f"[0:v]{vf_chain}[bg]"
        prev = "bg"
        for k, _ov in enumerate(overlays):
            inp_idx = 2 + k
            tag = f"v{k}"
            fade = ""
            # Title overlay (last) fades out after 3.5s on first scene.
            fc += f";[{prev}][{inp_idx}:v]overlay=0:0{fade}[{tag}]"
            prev = tag
        filter_complex = fc
        map_args = ["-map", f"[{prev}]", "-map", "1:a"]
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex] + map_args + [
            "-t", f"{duration:.3f}", "-r", str(fps),
            "-c:v", "libx264", "-preset", qp["x264"], "-crf", qp["crf"],
            "-pix_fmt", "yuv420p",
            "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-shortest", out_clip]
    else:
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-vf", vf_chain, "-t", f"{duration:.3f}", "-r", str(fps),
            "-c:v", "libx264", "-preset", qp["x264"], "-crf", qp["crf"],
            "-pix_fmt", "yuv420p",
            "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-shortest", out_clip]
    _run(cmd)
    return out_clip, duration


def _concat_clips(clip_paths, out_path):
    """Concatenate scene clips with stream-copy (near-instant)."""
    listfile = out_path + ".concat.txt"
    with open(listfile, "w") as f:
        for c in clip_paths:
            f.write(f"file '{os.path.abspath(c)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
           "-c", "copy", "-movflags", "+faststart", out_path]
    try:
        _run(cmd)
    except Exception:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-ar", "44100", "-movflags", "+faststart", out_path]
        _run(cmd)
    try:
        os.unlink(listfile)
    except Exception:
        pass
    return out_path


def build_video(script, resolution="1280x720", language="en", voice=None,
                workdir=None, out_path=None, progress_cb=None, image_model="flux",
                tts_rate="+0%", gender=None, quality="balanced",
                template="auto"):
    """
    script: dict {title, scenes:[{narration, image_prompt, caption}]}
    quality: "fast" | "balanced" | "high"  (CPU/quality trade-off)
    template: a key in TEMPLATES, or "auto" to rotate professional templates
              across scenes for a dynamic, non-monotone presentation.
    Returns final video path.
    """
    if resolution not in RESOLUTIONS:
        resolution = "1280x720"
    w, h = RESOLUTIONS[resolution]
    qp = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="vidgen_")
    os.makedirs(workdir, exist_ok=True)

    scenes = script["scenes"]
    n = len(scenes)

    def report(stage, i, msg):
        if progress_cb:
            progress_cb(stage, i, n, msg)

    img_paths = [os.path.join(workdir, f"img_{i:03d}.jpg") for i in range(n)]
    aud_paths = [os.path.join(workdir, f"aud_{i:03d}.mp3") for i in range(n)]
    done_counter = {"img": 0, "tts": 0}
    lock = threading.Lock()

    def do_image(i):
        try:
            ai_image.generate_image(scenes[i]["image_prompt"], img_paths[i],
                                    width=w, height=h, model=image_model,
                                    quality=qp["img"], raw_prompt=False)
        except Exception:
            _make_placeholder(img_paths[i], w, h,
                              scenes[i].get("caption", "") or script.get("title", ""),
                              language)
        with lock:
            done_counter["img"] += 1
            report("image", done_counter["img"] - 1,
                   f"ساخت تصاویر {done_counter['img']}/{n}")

    def do_tts(i):
        try:
            ai_tts.synthesize(scenes[i]["narration"], aud_paths[i],
                              language=language, voice=voice, rate=tts_rate,
                              gender=gender)
        except Exception:
            _make_silence(aud_paths[i], 4.0)
        with lock:
            done_counter["tts"] += 1
            report("tts", done_counter["tts"] - 1,
                   f"ساخت صدای گوینده {done_counter['tts']}/{n}")

    # Concurrency tuned for low-end CPU + free services.
    with ThreadPoolExecutor(max_workers=min(3, n)) as ex:
        list(ex.map(do_image, range(n)))
    with ThreadPoolExecutor(max_workers=min(3, n)) as ex:
        list(ex.map(do_tts, range(n)))

    # Title overlay for the first scene (used by the cinematic layout).
    title_png = os.path.join(workdir, "title.png")
    title_overlay = _title_png(script.get("title", ""), w, h, language, title_png)
    video_title = script.get("title", "")

    # Choose per-scene templates.
    def _scene_template(i):
        if template != "auto" and template in TEMPLATES:
            return template
        # AUTO: opening = cinematic hero, then rotate card/sidebar themes,
        # closing = cinematic for a strong finish.
        if i == 0 or i == n - 1:
            return "midnight"  # cinematic layout
        rotation = ["aurora", "ocean", "sunset"]  # card/sidebar layouts
        return rotation[(i - 1) % len(rotation)]

    # Render scene clips (CPU-bound -> sequential to avoid lag on weak CPU).
    clips = []
    for i in range(n):
        report("clip", i, f"رندر صحنه {i+1}/{n}")
        clip_path = os.path.join(workdir, f"clip_{i:03d}.mp4")
        tpl = _scene_template(i)
        # Title text shown on the designed slide (skip on the opening cinematic
        # hero which already gets the big title overlay).
        slide_title = "" if (i == 0) else video_title
        _make_scene_clip(img_paths[i], aud_paths[i], clip_path, w, h,
                         scenes[i].get("caption", ""), language, workdir, i, qp,
                         title_overlay=(title_overlay if i == 0 else None),
                         template=tpl, title=slide_title)
        clips.append(clip_path)

    report("concat", n, "ترکیب صحنه‌ها و آماده‌سازی خروجی")
    if out_path is None:
        out_path = os.path.join(workdir, "final.mp4")
    _concat_clips(clips, out_path)
    report("done", n, "ویدیو آماده شد")
    return out_path


def _make_placeholder(path, w, h, text="", language="en"):
    """Generate a nice gradient placeholder image with PIL (no network)."""
    try:
        img = Image.new("RGB", (w, h), (18, 26, 56))
        draw = ImageDraw.Draw(img)
        # diagonal gradient
        for y in range(0, h, 4):
            shade = int(18 + (y / h) * 30)
            draw.rectangle([0, y, w, y + 4], fill=(shade, shade + 8, shade + 30))
        if text:
            fontpath = _find_font(language)
            if fontpath:
                fs = max(28, int(h * 0.06))
                font = ImageFont.truetype(fontpath, fs)
                is_rtl = language in _RTL_LANGS
                shaped = _shape_rtl(text[:60], language)
                kw = _text_kwargs(is_rtl)
                bbox = draw.textbbox((0, 0), shaped, font=font, **kw)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((w - tw) // 2, (h - th) // 2), shaped,
                          font=font, fill=(235, 240, 255), **kw)
        img.save(path, quality=92)
        return path
    except Exception:
        # ultimate ffmpeg fallback
        vf = f"color=c=0x1a2238:s={w}x{h}"
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i", vf, "-frames:v", "1", path])
        return path


def _make_silence(path, seconds):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
           "anullsrc=channel_layout=stereo:sample_rate=44100",
           "-t", f"{seconds}", "-c:a", "libmp3lame", path]
    _run(cmd)
    return path
