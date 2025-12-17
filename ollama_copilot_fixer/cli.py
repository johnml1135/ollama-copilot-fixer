from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from . import console
from .cache import clear_cache, ensure_cache_dirs, format_bytes, get_cache_info
from .config import load_config
from .gguf import detect_architecture, is_sharded_model, merge_sharded_model, shard_files, shards_fingerprint
from .huggingface import hf_download_cached
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


def build_setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama-copilot-fixer",
        description="Download/merge GGUF and create an Ollama model with Tool-capable template for GitHub Copilot.",
    )

    p.add_argument("--config", help="Path to config.json (default: OS config dir or $OLLAMA_COPILOT_FIXER_CONFIG)")
    p.add_argument("--cache-root", help="Override cache root directory (downloads/merged/work live here)")

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
    p.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Context window (num_ctx). If omitted, do not set num_ctx in the Modelfile "
            "and let Ollama/model defaults apply (Ollama may cap this, e.g. to 256k)."
        ),
    )
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


def build_cache_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama-copilot-fixer cache",
        description="Inspect and clear the ollama-copilot-fixer cache.",
    )
    p.add_argument("--config", help="Path to config.json (default: OS config dir or $OLLAMA_COPILOT_FIXER_CONFIG)")
    p.add_argument("--cache-root", help="Override cache root directory")

    sub = p.add_subparsers(dest="cache_cmd", required=True)
    sub.add_parser("info", help="Show cache locations and sizes")
    clear = sub.add_parser("clear", help="Clear cache directories")
    clear.add_argument("--all", action="store_true", help="Clear everything (default)")
    clear.add_argument("--hf", action="store_true", help="Clear Hugging Face cache (managed by this tool)")
    clear.add_argument("--downloads", action="store_true", help="Clear CLI download copies")
    clear.add_argument("--merged", action="store_true", help="Clear merged GGUF outputs")
    clear.add_argument("--work", action="store_true", help="Clear work directories")
    return p


def _run_cache(argv: list[str]) -> int:
    args = build_cache_parser().parse_args(argv)
    config = load_config(config_path=args.config, cache_root_override=args.cache_root)
    ensure_cache_dirs(config)

    if args.cache_cmd == "info":
        info = get_cache_info(config)
        console.info(f"Config: {info.cache_root}")
        console.info(f"HF cache: {info.hf_cache_dir} ({format_bytes(info.hf_bytes)})")
        console.info(f"Downloads: {info.downloads_dir} ({format_bytes(info.downloads_bytes)})")
        console.info(f"Merged: {info.merged_dir} ({format_bytes(info.merged_bytes)})")
        console.info(f"Work: {info.work_dir} ({format_bytes(info.work_bytes)})")
        console.success(f"Total: {format_bytes(info.total_bytes)}")
        return 0

    if args.cache_cmd == "clear":
        # If no specific flags given, default to --all.
        any_specific = bool(args.hf or args.downloads or args.merged or args.work)
        clear_all = bool(args.all) or not any_specific
        clear_cache(
            config=config,
            clear_hf=clear_all or bool(args.hf),
            clear_downloads=clear_all or bool(args.downloads),
            clear_merged=clear_all or bool(args.merged),
            clear_work=clear_all or bool(args.work),
        )
        console.success("Cache cleared.")
        return 0

    console.error("Unknown cache command.")
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0].lower() == "cache":
        return _run_cache(argv[1:])

    args = build_setup_parser().parse_args(argv)

    config = load_config(config_path=args.config, cache_root_override=args.cache_root)
    ensure_cache_dirs(config)

    if not shutil.which("ollama"):
        console.error("Ollama ('ollama') not found on PATH. Install from https://ollama.ai and ensure it's running.")
        return 1

    parsed = parse_model_source(args.model_source)

    temp_dir = Path(tempfile.mkdtemp(prefix="ollama_copilot_fixer_", dir=str(config.work_dir)))
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
            gguf_path = hf_download_cached(parsed.repo_id, config, quant)
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

            shards = shard_files(str(working_gguf))
            fp = shards_fingerprint(shards)
            merged = config.merged_dir / f"merged_{fp}.gguf"
            merged_created = False
            if merged.exists() and merged.stat().st_size > 0:
                console.success(f"Reusing cached merge: {merged.name}")
                final_gguf = merged
            else:
                final_path = merge_sharded_model(str(working_gguf), str(merged), llama_split)
                final_gguf = Path(final_path)
                merged_created = True
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

        # Nemotron GGUFs frequently leak turn markers / non-standard tool markup.
        # We apply a small set of safe stop tokens and stronger system guidance.
        source_hint = (parsed.repo_id or "") + " " + final_gguf.name
        is_nemotron = "nemotron" in source_hint.lower()
        extra_stop: list[str] = []
        system_message = None
        if is_nemotron:
            console.info("Nemotron detected; applying compatibility tweaks.")
            extra_stop = ["<|start_of_turn|>", "<|end_of_turn|>"]
            system_message = (
                "You are a helpful AI assistant with tool calling capabilities. "
                "Use tools when needed. Do not emit tool-call markup as plain text. "
                "When calling a tool, use the tool calling mechanism only."
            )

        modelfile_text = generate_modelfile(
            absolute_model_path=str(final_gguf.resolve()),
            architecture=arch,
            context_length=args.context_length,
            temperature=args.temperature,
            extra_stop=extra_stop or None,
            system_message=system_message,
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

        # Cleanup cached merged artifacts if configured.
        try:
            if is_sharded_model(str(working_gguf)) and (not config.keep_merged):
                if 'merged' in locals() and isinstance(merged, Path) and merged.exists():
                    if 'merged_created' in locals() and merged_created:
                        merged.unlink(missing_ok=True)
        except Exception:
            pass

        if args.keep_downloads:
            console.info(f"Kept working directory: {temp_dir}")
        return 0

    except KeyboardInterrupt:
        console.warn("Cancelled (Ctrl-C).")
        return 130
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
