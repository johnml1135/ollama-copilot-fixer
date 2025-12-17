from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import AppConfig


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
    raise RuntimeError(
        "hf_download(repo_id, dest_dir, quantization_type) is deprecated. "
        "Call hf_download_cached(repo_id, config, quantization_type) instead."
    )


def hf_download_cached(repo_id: str, config: AppConfig, quantization_type: str | None) -> str:
    """Download or reuse a GGUF from Hugging Face into the app-managed cache.

    Prefer the python library (huggingface_hub) to avoid local-dir copies.
    Falls back to the 'hf' CLI if the library is not importable.
    """
    # 1) Prefer huggingface_hub if available.
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore

        api = HfApi()
        all_files = api.list_repo_files(repo_id=repo_id)
        gguf_files = [f for f in all_files if f.lower().endswith(".gguf")]
        gguf_files = [f for f in gguf_files if not _is_helper_gguf(Path(f).name)]
        if not gguf_files:
            raise RuntimeError(
                "No GGUF files found in the repository. If this is a gated repo, run 'hf auth login' first."
            )

        def _matches_quant(path: str) -> bool:
            if not quantization_type:
                return True
            return quantization_type.lower() in Path(path).name.lower()

        candidates = [f for f in gguf_files if _matches_quant(f)]
        if quantization_type and not candidates:
            raise RuntimeError(
                f"No GGUF files matched quantization '{quantization_type}'. Try omitting --quantization-type."
            )
        if not candidates:
            candidates = gguf_files

        # Prefer first shard if present, otherwise prefer the largest file by metadata.
        # Note: list_repo_files does not include sizes, so we prefer shard-first and otherwise
        # download the first candidate and let the caller proceed.
        first_shards = [f for f in candidates if _is_first_shard(Path(f).name)]
        selected = sorted(first_shards or candidates)[0]

        selected_name = Path(selected).name

        # If this is a sharded set, download only that shard group.
        if _is_first_shard(selected_name):
            base = selected_name
            base = base.rsplit("-00001-of-", 1)[0]
            shard_paths = [
                f
                for f in candidates
                if Path(f).name.lower().startswith(base.lower() + "-") and Path(f).name.lower().endswith(".gguf")
            ]
            if not shard_paths:
                shard_paths = [selected]

            local_paths: list[Path] = []
            for sp in sorted(shard_paths):
                local_file = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=sp,
                        cache_dir=str(config.hf_cache_dir),
                    )
                )
                local_paths.append(local_file)

            # Return the first shard local path.
            first_local = [p for p in local_paths if _is_first_shard(p.name)]
            return str(sorted(first_local or local_paths, key=lambda p: p.name)[0].resolve())

        # Non-sharded: download only the selected file.
        local_file = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=selected,
                cache_dir=str(config.hf_cache_dir),
            )
        )
        return str(local_file.resolve())
    except ImportError:
        pass

    # 2) Fall back to hf CLI (will copy into a local dir, but we keep it cacheable).
    if not shutil_which("hf"):
        raise RuntimeError(
            "HuggingFace CLI ('hf') not found. Install with: python -m pip install -U huggingface_hub"
        )

    # Use a stable per-repo download directory so repeated runs do not redownload.
    safe_repo = repo_id.replace("/", "__")
    download_dir = config.downloads_dir / safe_repo
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
        # Keep Hugging Face cache under the app cache so users can clear it via this tool.
        env.setdefault("HF_HOME", str(config.hf_cache_dir))
        # Avoid noisy warnings in environments where symlinks are blocked.
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        try:
            out, _ = proc.communicate()
        except KeyboardInterrupt:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            raise

        if proc.returncode != 0:
            raise RuntimeError(f"HuggingFace download failed (exit {proc.returncode}).\n{out}")
        return out

    # If we already have a matching GGUF in the cache, reuse it.
    existing = list(download_dir.rglob("*.gguf"))
    if quantization_type:
        matching = [p for p in existing if quantization_type.lower() in p.name.lower()]
        if not matching:
            _run(f"*{quantization_type}*.gguf")
            existing = list(download_dir.rglob("*.gguf"))
            # If our filter was too strict, retry once without it.
            if not existing:
                _run("*.gguf")
    elif not existing:
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
