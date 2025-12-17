from __future__ import annotations

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    # package dir is <repo>/ollama_copilot_fixer
    return Path(__file__).resolve().parent.parent


def which(executable: str) -> str | None:
    return shutil.which(executable)


def find_llama_gguf_split(custom_path: str | None) -> str | None:
    candidates: list[Path] = []

    if custom_path:
        p = Path(custom_path)
        if p.exists():
            if p.is_file() and p.name.lower().endswith(".exe"):
                candidates.append(p)
            else:
                candidates.append(p / "llama-gguf-split.exe")
                candidates.append(p / "bin" / "llama-gguf-split.exe")

    candidates += [
        Path(r"C:\llama.cpp\bin\llama-gguf-split.exe"),
        Path(r"C:\llama.cpp\llama-gguf-split.exe"),
        Path(os.path.expandvars(r"%USERPROFILE%\llama.cpp\bin\llama-gguf-split.exe")),
        Path(os.path.expandvars(r"%USERPROFILE%\llama.cpp\llama-gguf-split.exe")),
    ]

    for c in candidates:
        if c and c.exists():
            return str(c.resolve())

    on_path = which("llama-gguf-split")
    return on_path
