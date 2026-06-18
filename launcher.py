#!/usr/bin/env python3
"""
launcher.py - Double-click entry point for Free AI Studio.

When packaged into an .exe (via PyInstaller) the user just double-clicks it:
  1. Ensures FFmpeg is available (bundled or system).
  2. Starts the FastAPI server (uvicorn) on a free local port.
  3. Waits until the server is healthy.
  4. Opens the default web browser at the app URL.
  5. Keeps running in a small console window until closed.

100% offline-capable launcher (no API keys). The AI features themselves use
free public services (Pollinations / Edge-TTS) + local FFmpeg.
"""
import os
import sys
import time
import socket
import threading
import webbrowser

# ---- Make bundled paths work both in dev and inside PyInstaller onefile ----
def _base_dir():
    if getattr(sys, "frozen", False):
        # PyInstaller: data is unpacked to sys._MEIPASS
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

BASE = _base_dir()
# When frozen, outputs should live next to the exe, not in temp.
if getattr(sys, "frozen", False):
    APP_HOME = os.path.dirname(sys.executable)
else:
    APP_HOME = BASE

# Ensure imports resolve
sys.path.insert(0, BASE)

# Point the server's output dir to a writable location next to the exe
os.environ.setdefault("AISTUDIO_HOME", APP_HOME)

# Add bundled ffmpeg to PATH if present
_bundled_ffmpeg = os.path.join(BASE, "ffmpeg")
if os.path.isdir(_bundled_ffmpeg):
    os.environ["PATH"] = _bundled_ffmpeg + os.pathsep + os.environ.get("PATH", "")


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


def wait_for_server(url, timeout=40):
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
    print("=" * 56)
    print("   Free AI Studio  -  Image & Video Generator")
    print("   (100% free, no API key needed)")
    print("=" * 56)

    port = find_free_port(8000)
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    # Import the app
    try:
        from app.server import app
    except Exception:
        # If launched from inside app/ dir
        from server import app  # type: ignore

    import uvicorn

    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    print(f"\n[*] Starting server at {base_url} ...")
    if wait_for_server(base_url + "/api/health", timeout=45):
        print("[+] Server is ready!")
        print(f"[+] Opening your browser at {base_url}")
        try:
            webbrowser.open(base_url)
        except Exception:
            pass
        print("\n>>> Keep this window open while you use the app.")
        print(">>> Close this window to stop the app.\n")
    else:
        print("[!] Server did not start in time.")
        print(f"[!] Try opening {base_url} manually.")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down. Bye!")


if __name__ == "__main__":
    main()
