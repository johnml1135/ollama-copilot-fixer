from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig


@dataclass(frozen=True)
class CacheInfo:
    cache_root: Path
    hf_cache_dir: Path
    downloads_dir: Path
    merged_dir: Path
    work_dir: Path
    total_bytes: int
    hf_bytes: int
    downloads_bytes: int
    merged_bytes: int
    work_bytes: int


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            # best-effort
            pass
    return total


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{num_bytes} B"


def get_cache_info(config: AppConfig) -> CacheInfo:
    hf_bytes = _dir_size(config.hf_cache_dir)
    downloads_bytes = _dir_size(config.downloads_dir)
    merged_bytes = _dir_size(config.merged_dir)
    work_bytes = _dir_size(config.work_dir)
    total = hf_bytes + downloads_bytes + merged_bytes + work_bytes

    return CacheInfo(
        cache_root=config.cache_root,
        hf_cache_dir=config.hf_cache_dir,
        downloads_dir=config.downloads_dir,
        merged_dir=config.merged_dir,
        work_dir=config.work_dir,
        total_bytes=total,
        hf_bytes=hf_bytes,
        downloads_bytes=downloads_bytes,
        merged_bytes=merged_bytes,
        work_bytes=work_bytes,
    )


def ensure_cache_dirs(config: AppConfig) -> None:
    config.cache_root.mkdir(parents=True, exist_ok=True)
    config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    config.downloads_dir.mkdir(parents=True, exist_ok=True)
    config.merged_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)


def clear_cache(
    *,
    config: AppConfig,
    clear_hf: bool,
    clear_downloads: bool,
    clear_merged: bool,
    clear_work: bool,
) -> None:
    targets: list[Path] = []
    if clear_hf:
        targets.append(config.hf_cache_dir)
    if clear_downloads:
        targets.append(config.downloads_dir)
    if clear_merged:
        targets.append(config.merged_dir)
    if clear_work:
        targets.append(config.work_dir)

    for t in targets:
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)
