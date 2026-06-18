@echo off
REM ============================================================
REM   Free AI Studio - Windows EXE Builder
REM   Double-click this file on Windows to build FreeAIStudio.exe
REM   100% free. No API key needed.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo    Free AI Studio  -  Building Windows EXE
echo ============================================================
echo.

REM --- 1. Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo During install, CHECK "Add Python to PATH".
    pause
    exit /b 1
)
echo [OK] Python found.

REM --- 2. Create / use virtual env ---
if not exist ".venv" (
    echo [*] Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM --- 3. Install dependencies ---
echo [*] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
python -m pip install pyinstaller

REM --- 4. Download FFmpeg if missing ---
if not exist "ffmpeg\ffmpeg.exe" (
    echo [*] Downloading FFmpeg (required for video)...
    mkdir ffmpeg 2>nul
    powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg\ff.zip'"
    powershell -Command "Expand-Archive -Force 'ffmpeg\ff.zip' 'ffmpeg\extracted'"
    for /r "ffmpeg\extracted" %%f in (ffmpeg.exe) do copy "%%f" "ffmpeg\ffmpeg.exe" >nul
    for /r "ffmpeg\extracted" %%f in (ffprobe.exe) do copy "%%f" "ffmpeg\ffprobe.exe" >nul
    del "ffmpeg\ff.zip" 2>nul
    rmdir /s /q "ffmpeg\extracted" 2>nul
    echo [OK] FFmpeg ready.
) else (
    echo [OK] FFmpeg already present.
)

REM --- 5. Build EXE ---
echo [*] Building EXE with PyInstaller...
pyinstaller --noconfirm FreeAIStudio.spec

echo.
echo ============================================================
echo  [DONE] Your app is ready at:
echo         dist\FreeAIStudio.exe
echo  Double-click it to launch Free AI Studio in your browser!
echo ============================================================
echo.
pause
