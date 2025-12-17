from __future__ import annotations

import subprocess


def _run(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(args)}\n{proc.stdout}")
    return proc.stdout


def create_model(model_name: str, modelfile_path: str) -> str:
    return _run(["ollama", "create", model_name, "-f", modelfile_path])


def list_models() -> str:
    return _run(["ollama", "list"])


def run_model(model_name: str, prompt: str) -> str:
    return _run(["ollama", "run", model_name, prompt])
