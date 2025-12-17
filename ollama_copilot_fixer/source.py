from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ParsedSource:
    is_hf: bool
    repo_id: str | None
    quant_suffix: str | None
    local_path: str | None


_HF_HOST_RE = re.compile(r"^(?:https?://)?(?P<host>hf\.co|huggingface\.co)/", re.IGNORECASE)
_OWNER_REPO_RE = re.compile(r"^(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)$")


def _split_repo_segment(repo_segment: str) -> tuple[str, str | None]:
    # repo:Q4_0
    if ":" in repo_segment:
        name, suffix = repo_segment.split(":", 1)
        name = name.strip()
        suffix = suffix.strip() or None
        return name, suffix
    return repo_segment, None


def parse_model_source(model_source: str) -> ParsedSource:
    s = model_source.strip()

    # Accept full ollama command text
    m = re.match(r"^\s*ollama\s+(run|pull)\s+(?P<rest>.+)$", s, re.IGNORECASE)
    if m:
        s = m.group("rest").strip()

    # Only consider first token
    token = s.split()[0]

    # hf.co / huggingface.co forms
    if _HF_HOST_RE.match(token):
        uri = token if re.match(r"^https?://", token, re.IGNORECASE) else f"https://{token}"
        parsed = urlparse(uri)
        segments = [seg for seg in parsed.path.split("/") if seg]
        if len(segments) >= 2:
            owner = segments[0]
            repo_segment = segments[1]
            repo_name, suffix = _split_repo_segment(repo_segment)
            return ParsedSource(True, f"{owner}/{repo_name}", suffix, None)

    # owner/repo or owner/repo:Q4_0
    m2 = _OWNER_REPO_RE.match(token)
    if m2:
        owner = m2.group("owner")
        repo_segment = m2.group("repo")
        repo_name, suffix = _split_repo_segment(repo_segment)
        return ParsedSource(True, f"{owner}/{repo_name}", suffix, None)

    # else: treat as local path (resolved later)
    return ParsedSource(False, None, None, token)
