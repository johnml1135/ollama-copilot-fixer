from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from . import console
from .gguf import detect_architecture, is_sharded_model, merge_sharded_model
from .huggingface import hf_download
from .modelfile import generate_modelfile, supported_architectures
from .ollama import create_model, list_models, run_model
from .paths import find_llama_gguf_split
from .source import parse_model_source


def _sanitize_model_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9-_]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _auto_model_name_from_path(p: Path) -> str:
    base = p.stem.lower()
    base = re.sub(r"[^a-z0-9-_]", "-", base)
    base = re.sub(r"-+", "-", base)
    base = re.sub(r"-\d{5}-of-\d{5}$", "", base)
    base = re.sub(r"-\d{4}-of-\d{4}$", "", base)
    return base.strip("-") or "ollama-model"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama-copilot-fixer",
        description="Download/merge GGUF and create an Ollama model with Tool-capable template for GitHub Copilot.",
    )

    p.add_argument(
        "--model-source",
        required=True,
        help=(
            "Local GGUF path OR Hugging Face repo id/URL. Examples: "
            "unsloth/Llama-3.2-3B-Instruct-GGUF, hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_0, "
            "or 'ollama run hf.co/owner/repo:Q4_0'"
        ),
    )
    p.add_argument("--model-name", help="Name to register in Ollama (default: derived from GGUF filename).")
    p.add_argument(
        "--architecture",
        default="auto",
        choices=["auto", *supported_architectures()],
        help="Force architecture, or auto-detect.",
    )
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--quantization-type", help="Quant filter for Hugging Face downloads, e.g. Q4_0, Q4_K_M, IQ2_XXS")
    p.add_argument(
        "--llama-cpp-path",
        help="Path to llama.cpp folder or llama-gguf-split.exe (required to merge sharded GGUFs).",
    )
    p.add_argument("--keep-downloads", action="store_true", help="Keep temporary download/merge directory.")
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip running a quick 'ollama run' smoke test.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not shutil.which("ollama"):
        console.error("Ollama ('ollama') not found on PATH. Install from https://ollama.ai and ensure it's running.")
        return 1

    parsed = parse_model_source(args.model_source)

    temp_dir = Path(tempfile.mkdtemp(prefix="ollama_copilot_fixer_"))
    console.info(f"Working directory: {temp_dir}")

    try:
        quant = args.quantization_type
        if not quant and parsed.quant_suffix:
            quant = parsed.quant_suffix

        working_gguf: Path

        if parsed.is_hf and parsed.repo_id:
            console.info(f"Hugging Face repo detected: {parsed.repo_id}")
            if quant:
                console.info(f"Quantization filter: {quant}")
            console.info("Downloading GGUF(s) from Hugging Face...")
            gguf_path = hf_download(parsed.repo_id, str(temp_dir), quant)
            working_gguf = Path(gguf_path)
            console.success(f"Downloaded/selected: {working_gguf.name}")
        else:
            if not parsed.local_path:
                console.error("Model source not recognized.")
                return 1
            candidate = Path(parsed.local_path)
            if not candidate.exists():
                console.error(f"Local path not found: {candidate}")
                return 1
            working_gguf = candidate.resolve()
            console.success(f"Using local file: {working_gguf}")

        final_gguf = working_gguf

        console.info("Checking for sharded GGUF...")
        if is_sharded_model(str(working_gguf)):
            console.warn("Sharded model detected; merge required.")
            llama_split = find_llama_gguf_split(args.llama_cpp_path)
            if not llama_split:
                console.error(
                    "llama-gguf-split not found. Install llama.cpp and/or add llama-gguf-split to PATH, "
                    "or pass --llama-cpp-path."
                )
                return 1
            console.info(f"Using llama-gguf-split: {llama_split}")
            merged = temp_dir / "merged_model.gguf"
            final_path = merge_sharded_model(str(working_gguf), str(merged), llama_split)
            final_gguf = Path(final_path)
            console.success(f"Merged into: {final_gguf.name}")
        else:
            console.success("Single-file model (no merge needed).")

        size_gb = final_gguf.stat().st_size / (1024**3)
        console.success(f"Working with: {final_gguf.name} ({size_gb:.2f} GB)")

        model_name = args.model_name
        if not model_name:
            model_name = _auto_model_name_from_path(final_gguf)
            console.info(f"Auto model name: {model_name}")
        model_name = _sanitize_model_name(model_name)

        arch = args.architecture
        if arch == "auto":
            console.info("Detecting architecture...")
            arch = detect_architecture(str(final_gguf))
        console.success(f"Architecture: {arch}")

        console.info("Generating Modelfile...")
        modelfile_path = temp_dir / "Modelfile"
        modelfile_text = generate_modelfile(
            absolute_model_path=str(final_gguf.resolve()),
            architecture=arch,
            context_length=args.context_length,
            temperature=args.temperature,
        )
        modelfile_path.write_text(modelfile_text, encoding="utf-8")
        console.success(f"Wrote Modelfile: {modelfile_path}")

        console.info("Creating model in Ollama...")
        create_out = create_model(model_name, str(modelfile_path))
        if create_out.strip():
            console.info(create_out.strip())
        console.success(f"Created model: {model_name}")

        console.info("Verifying with 'ollama list'...")
        lst = list_models()
        if model_name in lst:
            console.success("Model is registered in Ollama.")
        else:
            console.warn("Model name not found in 'ollama list' output (this can be transient).")

        if not args.skip_test:
            console.info("Running a quick smoke test (ollama run)...")
            try:
                out = run_model(model_name, "Hello, can you help me with code?")
                if out.strip():
                    console.success("Model responded successfully.")
            except Exception as e:
                console.warn(f"Smoke test failed: {e}")

        console.success("Setup complete. Your model should show Tool capability in VS Code Copilot.")

        if args.keep_downloads:
            console.info(f"Kept working directory: {temp_dir}")
        return 0

    except Exception as e:
        console.error(str(e))
        return 1

    finally:
        if not args.keep_downloads:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
