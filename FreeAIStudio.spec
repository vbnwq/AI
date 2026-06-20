# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Free AI Studio.
Builds a single double-clickable executable (.exe on Windows, binary on Linux/Mac).

Build:
    pip install pyinstaller
    pyinstaller FreeAIStudio.spec
Result:
    dist/FreeAIStudio(.exe)
"""
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Bundle the web UI and any package data
datas = [
    ("app/static", "app/static"),
    ("app/fonts", "app/fonts"),
]
# edge-tts ships data files
datas += collect_data_files("edge_tts")

hiddenimports = []
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("edge_tts")
hiddenimports += [
    "anyio", "h11", "websockets", "httptools", "uvloop",
    "arabic_reshaper", "bidi", "bidi.algorithm",
    "app", "app.server", "app.services",
    "app.services.translate",
    "app.services.ai_text", "app.services.ai_image",
    "app.services.ai_tts", "app.services.video_engine",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
    "PIL.ImageFilter", "PIL.ImageEnhance",
]

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy.testing"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="FreeAIStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # keep a small console so the user can close to stop
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="app/static/icon.ico" if os.path.exists("app/static/icon.ico") else None,
)
