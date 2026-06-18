@echo off
REM ============================================================
REM   Free AI Studio - Quick Start (no build needed)
REM   Double-click to run the app directly in your browser.
REM   First run installs dependencies automatically.
REM ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Get it from https://www.python.org/downloads/
    echo Remember to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [*] First-time setup: creating environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM Download FFmpeg if missing (needed for video)
if not exist "ffmpeg\ffmpeg.exe" (
    echo [*] Downloading FFmpeg (one-time, needed for video)...
    mkdir ffmpeg 2>nul
    powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg\ff.zip'"
    powershell -Command "Expand-Archive -Force 'ffmpeg\ff.zip' 'ffmpeg\extracted'"
    for /r "ffmpeg\extracted" %%f in (ffmpeg.exe) do copy "%%f" "ffmpeg\ffmpeg.exe" >nul
    for /r "ffmpeg\extracted" %%f in (ffprobe.exe) do copy "%%f" "ffmpeg\ffprobe.exe" >nul
    del "ffmpeg\ff.zip" 2>nul & rmdir /s /q "ffmpeg\extracted" 2>nul
)

echo [*] Launching Free AI Studio...
python launcher.py
pause
