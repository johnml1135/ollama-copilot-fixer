from __future__ import annotations

import os
import re
import subprocess
import hashlib
from pathlib import Path

_SHARD_PATTERNS = [
    re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE),
    re.compile(r"-part-\d+\.gguf$", re.IGNORECASE),
    re.compile(r"\.part\d+\.gguf$", re.IGNORECASE),
]


def is_sharded_model(file_path: str) -> bool:
    name = Path(file_path).name
    if any(p.search(name) for p in _SHARD_PATTERNS):
        return True

    directory = Path(file_path).parent
    base = name
    base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"-part-\d+\.gguf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\.part\d+\.gguf$", "", base, flags=re.IGNORECASE)

    related = [
        p
        for p in directory.glob(f"{base}*.gguf")
        if re.search(r"-\d{5}-of-\d{5}|part-?\d+", p.name, flags=re.IGNORECASE)
    ]
    return len(related) > 1


def shard_files(first_shard_path: str) -> list[Path]:
    first = Path(first_shard_path)
    directory = first.parent
    name = first.name

    base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"-part-\d+\.gguf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\.part\d+\.gguf$", "", base, flags=re.IGNORECASE)

    shards = [
        p
        for p in directory.glob(f"{base}*.gguf")
        if re.search(r"-\d{5}-of-\d{5}|part-?\d+", p.name, flags=re.IGNORECASE)
    ]
    return sorted(shards, key=lambda p: p.name)


def shards_fingerprint(shards: list[Path]) -> str:
    """Stable-ish fingerprint for a shard set, based on names + sizes + mtimes.

    Avoid hashing full contents (too expensive for large models).
    """
    h = hashlib.sha256()
    for p in shards:
        try:
            st = p.stat()
            h.update(p.name.encode("utf-8", errors="ignore"))
            h.update(str(st.st_size).encode("ascii"))
            h.update(str(int(st.st_mtime)).encode("ascii"))
            h.update(b"\n")
        except Exception:
            h.update(p.name.encode("utf-8", errors="ignore"))
            h.update(b"\n")
    return h.hexdigest()[:16]


def merge_sharded_model(first_shard_path: str, output_path: str, llama_split_path: str) -> str:
    shards = shard_files(first_shard_path)
    if len(shards) < 2:
        raise RuntimeError("Shard merge requested but related shards were not found.")

    os.makedirs(str(Path(output_path).parent), exist_ok=True)

    # llama-gguf-split --merge <first_shard> <output>
    proc = subprocess.run(
        [llama_split_path, "--merge", first_shard_path, output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Merge failed (exit {proc.returncode}).\n{proc.stdout}")

    if not Path(output_path).exists():
        raise RuntimeError("Merge completed but output file not found.")

    return str(Path(output_path).resolve())


def detect_architecture(file_path: str) -> str:
    # Best-effort detection
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(16384)
        content = chunk.decode("ascii", errors="ignore").lower()
    except Exception:
        content = ""

    if re.search(r"llama.*3\.[0-9]|llama3|llama-3", content):
        return "llama3"
    if re.search(r"mistral|mixtral", content):
        return "mistral"
    if re.search(r"phi-3|phi3|phi-4|phi4", content):
        return "phi3"
    if re.search(r"gemma.*2|gemma-2", content):
        return "gemma2"
    if re.search(r"qwen.*2|qwen-2", content):
        return "qwen"

    filename = Path(file_path).name.lower()
    if "nemotron" in filename:
        return "nemotron"
    if re.search(r"llama.*3", filename):
        return "llama3"
    if re.search(r"mistral|mixtral", filename):
        return "mistral"
    if re.search(r"phi", filename):
        return "phi3"
    if re.search(r"gemma", filename):
        return "gemma2"
    if re.search(r"qwen", filename):
        return "qwen"

    return "llama3"
