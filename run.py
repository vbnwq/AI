#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py - Master one-click launcher for Free AI Studio.

Just run:  python run.py

It does EVERYTHING automatically:
  1. Checks the Python version.
  2. Auto-installs any MISSING Python dependencies (only what is missing).
  3. Verifies FFmpeg/FFprobe exist; on Windows it can auto-download a static
     build if missing; on Linux/macOS it prints the exact install command.
  4. Ensures the output folders exist.
  5. Starts the FastAPI server (uvicorn) on a free local port in a thread.
  6. Waits until the server is healthy, then opens the default web browser.
  7. Keeps running until you press Ctrl+C / close the window.

100% free — no API keys. CPU-only friendly. Full Persian support.
"""
import os
import sys
import time
import socket
import shutil
import platform
import subprocess
import threading
import webbrowser

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _base_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

BASE = _base_dir()
APP_HOME = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else BASE
sys.path.insert(0, BASE)
os.environ.setdefault("AISTUDIO_HOME", APP_HOME)

# Bundled/persistent ffmpeg dirs (if present) take priority on PATH so the
# locally-installed FFmpeg from a previous run is detected INSTANTLY and we
# never re-download it. We register both the read-only bundle dir (next to the
# code, used inside a frozen EXE) and the writable persistent dir next to the
# app home (where one-time downloads are saved).
for _ff in (os.path.join(BASE, "ffmpeg"), os.path.join(APP_HOME, "ffmpeg")):
    if os.path.isdir(_ff) and _ff not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _ff + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------
# Pretty printing
# --------------------------------------------------------------------------
def info(msg):  print("  [*] " + msg)
def ok(msg):    print("  [+] " + msg)
def warn(msg):  print("  [!] " + msg)
def err(msg):   print("  [x] " + msg)

def banner():
    line = "=" * 60
    print(line)
    print("   Free AI Studio  -  Persian AI Image & Video Generator")
    print("   استودیوی هوش مصنوعی فارسی  (۱۰۰٪ رایگان، بدون API)")
    print(line)


# --------------------------------------------------------------------------
# 1. Python version
# --------------------------------------------------------------------------
def check_python():
    if sys.version_info < (3, 9):
        err("Python 3.9+ is required. Current: %s" % platform.python_version())
        sys.exit(1)
    ok("Python %s detected" % platform.python_version())


# --------------------------------------------------------------------------
# 2. Dependencies (install only what's missing)
# --------------------------------------------------------------------------
# (import_name, pip_spec)
REQUIRED = [
    ("fastapi",          "fastapi>=0.110"),
    ("uvicorn",          "uvicorn[standard]>=0.27"),
    ("requests",         "requests>=2.31"),
    ("PIL",              "pillow>=10.0"),
    ("edge_tts",         "edge-tts>=6.1"),
    ("arabic_reshaper",  "arabic-reshaper>=3.0"),
    ("bidi",             "python-bidi>=0.4"),
]


def _is_installed(import_name):
    try:
        __import__(import_name)
        return True
    except Exception:
        return False


def ensure_dependencies():
    info("Checking Python dependencies...")
    missing = [(imp, spec) for imp, spec in REQUIRED if not _is_installed(imp)]
    if not missing:
        ok("All Python dependencies are already installed.")
        return

    warn("Missing packages: " + ", ".join(spec for _, spec in missing))
    info("Installing missing packages (this happens only once)...")
    # Upgrade pip quietly first (best-effort).
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade",
                        "pip", "--quiet", "--disable-pip-version-check"],
                       check=False)
    except Exception:
        pass

    for imp, spec in missing:
        info("  pip install " + spec)
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec,
             "--quiet", "--disable-pip-version-check"]).returncode
        if rc != 0:
            # retry with --user as a fallback
            subprocess.run(
                [sys.executable, "-m", "pip", "install", spec, "--user",
                 "--quiet", "--disable-pip-version-check"], check=False)

    # Verify
    still = [imp for imp, _ in missing if not _is_installed(imp)]
    if still:
        err("Could not install: " + ", ".join(still))
        err("Please run:  python -m pip install -r requirements.txt")
        sys.exit(1)
    ok("Dependencies installed successfully.")


# --------------------------------------------------------------------------
# 3. FFmpeg  (PERSISTENT, ONE-TIME INSTALL)
# --------------------------------------------------------------------------
# The local, persistent install location for a bundled FFmpeg.  Once FFmpeg is
# downloaded here it is reused FOREVER — subsequent runs detect it instantly and
# skip the (~80MB) download entirely.  A tiny marker file records a successful
# install so we never re-download even if PATH lookups are slow.
FFMPEG_DIR = os.path.join(APP_HOME, "ffmpeg")
FFMPEG_MARKER = os.path.join(FFMPEG_DIR, ".installed")


def _exe(name):
    """Return the platform-specific executable filename."""
    return name + (".exe" if platform.system().lower() == "windows" else "")


def _local_ffmpeg_paths():
    """Paths to a locally-bundled ffmpeg/ffprobe (may or may not exist yet)."""
    return (os.path.join(FFMPEG_DIR, _exe("ffmpeg")),
            os.path.join(FFMPEG_DIR, _exe("ffprobe")))


def _local_ffmpeg_present():
    """True if a previously-installed local FFmpeg exists on disk."""
    fm, fp = _local_ffmpeg_paths()
    return os.path.isfile(fm) and os.path.isfile(fp)


def _register_local_ffmpeg():
    """Prepend the local ffmpeg dir to PATH so shutil.which / subprocess find it."""
    if os.path.isdir(FFMPEG_DIR):
        cur = os.environ.get("PATH", "")
        if FFMPEG_DIR not in cur.split(os.pathsep):
            os.environ["PATH"] = FFMPEG_DIR + os.pathsep + cur


def _ffmpeg_ok():
    """True if BOTH ffmpeg and ffprobe are resolvable (local bundle or system)."""
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def _download_ffmpeg_windows():
    """One-time auto-download of a static FFmpeg build on Windows."""
    try:
        import zipfile, urllib.request, tempfile
        url = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
               "ffmpeg-master-latest-win64-gpl.zip")
        info("Downloading FFmpeg for Windows (~80MB, ONE-TIME only)...")
        tmpzip = os.path.join(tempfile.gettempdir(), "ffmpeg_dl.zip")
        urllib.request.urlretrieve(url, tmpzip)
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        with zipfile.ZipFile(tmpzip) as z:
            for member in z.namelist():
                name = os.path.basename(member)
                if name.lower() in ("ffmpeg.exe", "ffprobe.exe"):
                    with z.open(member) as src, \
                            open(os.path.join(FFMPEG_DIR, name), "wb") as out:
                        shutil.copyfileobj(src, out)
        try: os.unlink(tmpzip)
        except Exception: pass
        return _local_ffmpeg_present()
    except Exception as e:
        warn("Auto-download of FFmpeg failed: " + str(e))
        return False


def _mark_installed():
    """Persist a marker so future runs skip the download instantly."""
    try:
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        with open(FFMPEG_MARKER, "w", encoding="utf-8") as f:
            f.write("ffmpeg installed by Free AI Studio\n")
    except Exception:
        pass


def ensure_ffmpeg():
    """Guarantee FFmpeg is available.  Download AT MOST ONCE, ever.

    Order of checks (fast path first → instant startup on every later run):
      1. A previously-installed LOCAL bundle (ffmpeg/ + .installed marker).
      2. A system-wide ffmpeg already on PATH.
      3. (Windows only) ONE-TIME auto-download into the local bundle dir.
    """
    info("Checking FFmpeg...")

    # 1) Persistent local bundle from a previous run → reuse instantly.
    if _local_ffmpeg_present():
        _register_local_ffmpeg()
        if not os.path.exists(FFMPEG_MARKER):
            _mark_installed()
        ok("FFmpeg found locally (cached install — no download needed).")
        return

    # 2) System install already on PATH.
    if _ffmpeg_ok():
        ok("FFmpeg & FFprobe found on system PATH.")
        return

    # 3) No FFmpeg anywhere → install exactly once.
    sysname = platform.system().lower()
    if sysname == "windows":
        if _download_ffmpeg_windows() and _local_ffmpeg_present():
            _register_local_ffmpeg()
            _mark_installed()
            ok("FFmpeg installed locally (one-time). Future runs start instantly.")
            return
        err("FFmpeg not found. Download it from https://ffmpeg.org/download.html "
            "and add it to PATH, then re-run.")
    elif sysname == "darwin":
        err("FFmpeg not found. Install it with:  brew install ffmpeg")
    else:
        err("FFmpeg not found. Install it with:  sudo apt install -y ffmpeg")
    err("The app needs FFmpeg to build videos. Exiting.")
    sys.exit(1)


# --------------------------------------------------------------------------
# 4. Folders
# --------------------------------------------------------------------------
def ensure_folders():
    for sub in ("output", "output/images", "output/videos", "output/temp"):
        os.makedirs(os.path.join(APP_HOME, sub), exist_ok=True)
    ok("Output folders ready.")


# --------------------------------------------------------------------------
# 5/6. Server + browser
# --------------------------------------------------------------------------
def find_free_port(preferred=8000):
    for port in [preferred, 8001, 8080, 8501, 5000, 0]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            actual = s.getsockname()[1]
            s.close()
            return actual
        except OSError:
            continue
    return preferred


def wait_for_server(url, timeout=50):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.6)
    return False


def main():
    banner()
    check_python()
    ensure_dependencies()
    ensure_ffmpeg()
    ensure_folders()

    port = find_free_port(8000)
    host = "127.0.0.1"
    base_url = "http://%s:%d" % (host, port)

    # Import only AFTER deps are guaranteed.
    try:
        from app.server import app
    except Exception:
        from server import app  # type: ignore
    import uvicorn

    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(target=run_server, daemon=True).start()

    print()
    info("Starting server at %s ..." % base_url)
    if wait_for_server(base_url + "/api/health", timeout=55):
        ok("Server is ready!")
        ok("Opening your browser at %s" % base_url)
        try:
            webbrowser.open(base_url)
        except Exception:
            pass
        print()
        print("  >>> Keep this window open while you use the app.")
        print("  >>> Press Ctrl+C (or close this window) to stop.\n")
    else:
        warn("Server did not respond in time.")
        warn("Try opening %s manually in your browser." % base_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down. خداحافظ! 👋")


if __name__ == "__main__":
    main()
