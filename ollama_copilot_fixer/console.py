from __future__ import annotations

import sys


def _print(prefix: str, message: str, stream) -> None:
    stream.write(f"{prefix} {message}\n")


def info(message: str) -> None:
    _print("ℹ", message, sys.stdout)


def success(message: str) -> None:
    _print("✓", message, sys.stdout)


def warn(message: str) -> None:
    _print("⚠", message, sys.stdout)


def error(message: str) -> None:
    _print("✗", message, sys.stderr)
