from __future__ import annotations

import os
import subprocess
from pathlib import Path


_HF_DOWNLOAD_HELP: str | None = None


def _hf_supports_local_dir_use_symlinks() -> bool:
    global _HF_DOWNLOAD_HELP
    if _HF_DOWNLOAD_HELP is None:
        try:
            proc = subprocess.run(
                ["hf", "download", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            _HF_DOWNLOAD_HELP = proc.stdout or ""
        except Exception:
            _HF_DOWNLOAD_HELP = ""

    return "--local-dir-use-symlinks" in (_HF_DOWNLOAD_HELP or "")


def hf_download(repo_id: str, dest_dir: str, quantization_type: str | None) -> str:
    if not shutil_which("hf"):
        raise RuntimeError(
            "HuggingFace CLI ('hf') not found. Install with: python -m pip install -U huggingface_hub"
        )

    download_dir = Path(dest_dir) / f"hf_download_{os.getpid()}_{os.urandom(3).hex()}"
    download_dir.mkdir(parents=True, exist_ok=True)

    def _run(include_pattern: str) -> str:
        # Some hf CLI versions support --local-dir-use-symlinks, others don't.
        # When unsupported, the CLI typically falls back to copying if symlinks aren't available.
        args = [
            "hf",
            "download",
            repo_id,
            "--local-dir",
            str(download_dir),
            "--include",
            include_pattern,
        ]

        if _hf_supports_local_dir_use_symlinks():
            args.insert(6, "--local-dir-use-symlinks")
            args.insert(7, "False")

        env = os.environ.copy()
        # Avoid noisy warnings in environments where symlinks are blocked.
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"HuggingFace download failed (exit {proc.returncode}).\n{proc.stdout}")
        return proc.stdout

    if quantization_type:
        _run(f"*{quantization_type}*.gguf")
        ggufs = list(download_dir.rglob("*.gguf"))
        # If our filter was too strict, retry once without it.
        if not ggufs:
            _run("*.gguf")
    else:
        _run("*.gguf")

    ggufs = list(download_dir.rglob("*.gguf"))
    if not ggufs:
        raise RuntimeError(
            "No GGUF files found after download. If this is a gated repo, run 'hf auth login' first."
        )

    # Prefer real model weights over helper GGUFs when possible.
    primary = [p for p in ggufs if not _is_helper_gguf(p.name)]
    if not primary:
        primary = ggufs

    # Prefer first shard if present.
    first_shards = [p for p in primary if _is_first_shard(p.name)]
    if first_shards:
        return str(sorted(first_shards, key=lambda p: p.name)[0].resolve())

    # Otherwise, pick the largest.
    largest = sorted(primary, key=lambda p: p.stat().st_size, reverse=True)[0]
    return str(largest.resolve())


def _is_helper_gguf(name: str) -> bool:
    lowered = name.lower()
    bad = ["imatrix", "mmproj", "clip", "vision", "text-encoder", "vae"]
    return any(b in lowered for b in bad)


def _is_first_shard(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".gguf") and ("-00001-of-" in lowered)


def shutil_which(exe: str) -> str | None:
    # local helper so we don't require external deps
    from shutil import which

    return which(exe)
