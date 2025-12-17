from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _default_cache_root() -> Path:
    # Prefer non-roaming cache location.
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "ollama-copilot-fixer" / "cache"
    # Linux/macOS
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "ollama-copilot-fixer"
    return Path.home() / ".cache" / "ollama-copilot-fixer"


def _default_config_path() -> Path:
    # Config is small; OK to be in roaming profile.
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "ollama-copilot-fixer" / "config.json"
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "ollama-copilot-fixer" / "config.json"
    return Path.home() / ".config" / "ollama-copilot-fixer" / "config.json"


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    cache_root: Path
    keep_downloads: bool
    keep_merged: bool

    @property
    def hf_cache_dir(self) -> Path:
        return self.cache_root / "hf"

    @property
    def downloads_dir(self) -> Path:
        # Used by the hf CLI fallback (it downloads/copies here).
        return self.cache_root / "downloads"

    @property
    def merged_dir(self) -> Path:
        return self.cache_root / "merged"

    @property
    def work_dir(self) -> Path:
        return self.cache_root / "work"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except Exception:
        # If the config is malformed, ignore it and fall back to defaults.
        return {}
    return {}


def load_config(*, config_path: str | None, cache_root_override: str | None) -> AppConfig:
    env_config = os.environ.get("OLLAMA_COPILOT_FIXER_CONFIG")

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else (Path(env_config).expanduser().resolve() if env_config else _default_config_path())
    )

    data = _read_json(resolved_config_path)

    cache_root = (
        Path(cache_root_override).expanduser().resolve()
        if cache_root_override
        else Path(str(data.get("cache_root") or "")).expanduser().resolve()
        if data.get("cache_root")
        else _default_cache_root()
    )

    keep_downloads = bool(data.get("keep_downloads", True))
    keep_merged = bool(data.get("keep_merged", False))

    return AppConfig(
        config_path=resolved_config_path,
        cache_root=cache_root,
        keep_downloads=keep_downloads,
        keep_merged=keep_merged,
    )
