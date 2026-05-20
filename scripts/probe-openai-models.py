#!/usr/bin/env python3
"""Probe a llama-server OpenAI-compatible endpoint with Copilot-like requests.

The goal is to separate "the model loaded" from "the model can survive the
agent/tool workload".  The script uses only the Python standard library so it
can run in a fresh Windows checkout without installing the OpenAI package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "llama-server.log.err"
DEFAULT_OUT_DIR = ROOT / "logs"


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 120,
) -> tuple[int, dict[str, Any] | None, str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed, raw


def log_offset(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_log_delta(path: pathlib.Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(min(offset, path.stat().st_size))
        return handle.read().decode("utf-8", errors="replace")


def summarize_log_delta(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    prompt_tokens = [int(x) for x in re.findall(r"task\.n_tokens\s*=\s*(\d+)", text)]
    progress = []
    for match in re.finditer(
        r"prompt processing, n_tokens\s*=\s*(\d+).*?t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+)\s*tokens per second",
        text,
    ):
        progress.append(
            {
                "tokens": int(match.group(1)),
                "elapsed_sec": float(match.group(2)),
                "tokens_per_sec": float(match.group(3)),
            }
        )

    prompt_eval = []
    for match in re.finditer(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        text,
    ):
        prompt_eval.append(
            {
                "elapsed_ms": float(match.group(1)),
                "tokens": int(match.group(2)),
                "tokens_per_sec": float(match.group(3)),
            }
        )

    eval_timing = []
    for match in re.finditer(
        r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        text,
        re.MULTILINE,
    ):
        eval_timing.append(
            {
                "elapsed_ms": float(match.group(1)),
                "tokens": int(match.group(2)),
                "tokens_per_sec": float(match.group(3)),
            }
        )

    stops = []
    for match in re.finditer(r"stop processing: n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)", text):
        stops.append({"tokens": int(match.group(1)), "truncated": bool(int(match.group(2)))})

    fatal_lines = [
        line
        for line in lines
        if re.search(r"\b(CUDA error|out of memory|OOM|GGML_ASSERT|fatal error|failed to allocate)\b", line, re.I)
    ]

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_processing": progress,
        "prompt_eval": prompt_eval,
        "eval": eval_timing,
        "stop_processing": stops,
        "cancel_count": len(re.findall(r"cancel task", text)),
        "slot_unavailable_count": len(re.findall(r"no slot is available", text)),
        "fatal_count": len(fatal_lines),
        "fatal_lines": fatal_lines[-8:],
        "log_excerpt": lines[-24:],
    }


def tail_lines(text: str, count: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-count:]


def content_preview(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True)
    value = value.replace("\r", " ").replace("\n", " ")
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def make_basic_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 64,
    }


def make_two_tools_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Use structured tool calls when tools are requested. Return no prose.",
            },
            {
                "role": "user",
                "content": (
                    "Make exactly two function calls. First call get_model_status with "
                    "component='llama-server'. Then call record_metric with name='tool_probe' "
                    "and value=2. Do not write normal text."
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_model_status",
                    "description": "Return status for one local model-serving component.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "component": {
                                "type": "string",
                                "enum": ["llama-server", "gateway", "copilot"],
                            }
                        },
                        "required": ["component"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_metric",
                    "description": "Record one diagnostic metric from a model probe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"},
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 192,
    }


def make_two_tools_step_payload(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 192,
) -> dict[str, Any]:
    tools = make_two_tools_payload(model)["tools"]
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def make_schema_bloat_payload(model: str, schema_kb: int) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    target = max(1, schema_kb) * 1024
    index = 0
    while len(json.dumps(tools, ensure_ascii=True)) < target and index < 80:
        properties: dict[str, Any] = {}
        for prop_index in range(10):
            properties[f"field_{prop_index}"] = {
                "type": "string",
                "description": (
                    "Diagnostic field used to emulate verbose Copilot tool schemas. "
                    "It should be copied only when the user explicitly asks for it. " * 2
                ).strip(),
            }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"diagnostic_tool_{index}",
                    "description": (
                        "A synthetic diagnostic tool with intentionally long schema text. "
                        "This emulates the grammar and prompt burden from a tool-rich agent "
                        "session without exposing real workspace data. " * 3
                    ).strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": ["field_0"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        index += 1

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Use a tool call for diagnostics."},
            {
                "role": "user",
                "content": "Call diagnostic_tool_0 with field_0 set to ping. Return no prose.",
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 192,
    }


def make_prefill_payload(model: str, approx_tokens: int) -> dict[str, Any]:
    line = "The local model probe is filling prompt context with deterministic text. "
    chars = max(256, approx_tokens * 4)
    repeated = (line * ((chars // len(line)) + 1))[:chars]
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read this diagnostic filler, then reply with exactly PREFILL_OK.\n\n"
                    f"{repeated}\n\nReply with exactly PREFILL_OK."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
    }


def payload_for_test(name: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "basic":
        return prepare_payload(make_basic_payload(model), args)
    if name == "two-tools":
        return prepare_payload(make_two_tools_payload(model), args)
    if name == "schema-bloat":
        return prepare_payload(make_schema_bloat_payload(model, args.schema_kb), args)
    if name == "prefill":
        return prepare_payload(make_prefill_payload(model, args.prefill_tokens), args)
    raise ValueError(f"unknown test: {name}")


def prepare_payload(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = apply_cache_bust(payload, args.probe_id)
    if args.append_no_think:
        payload = append_no_think_hint(payload)
    return payload


def apply_cache_bust(payload: dict[str, Any], probe_id: str | None) -> dict[str, Any]:
    if not probe_id:
        return payload

    payload = json.loads(json.dumps(payload, ensure_ascii=True))
    messages = payload.setdefault("messages", [])
    messages.insert(0, {"role": "system", "content": f"diagnostic probe id: {probe_id}"})

    for index, tool in enumerate(payload.get("tools") or []):
        function = tool.get("function") or {}
        function["description"] = f"Probe {probe_id} tool {index}. {function.get('description', '')}"
    return payload


def append_no_think_hint(payload: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload, ensure_ascii=True))
    for message in reversed(payload.get("messages") or []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            if "/no_think" not in message["content"]:
                message["content"] = message["content"].rstrip() + "\n/no_think"
            break
    return payload


def discover_model(base_url: str, api_key: str, timeout_sec: float) -> str:
    status, parsed, raw = request_json("GET", f"{base_url}/models", api_key, timeout_sec=timeout_sec)
    if status >= 400 or not parsed:
        raise RuntimeError(f"could not discover /models: HTTP {status}: {content_preview(raw)}")
    data = parsed.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError(f"/models returned no model ids: {content_preview(parsed)}")
    return data[0]["id"]


def run_one_test(name: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "two-tools":
        return run_two_tools_sequence(model, args)

    payload = payload_for_test(name, model, args)
    payload_bytes = len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
    before = log_offset(args.log_file)
    start = time.perf_counter()
    try:
        status, parsed, raw = request_json(
            "POST",
            f"{args.base_url}/chat/completions",
            args.api_key,
            payload,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "http_status": 0,
            "ok": False,
            "elapsed_sec": round(elapsed, 3),
            "request_kb": round(payload_bytes / 1024, 2),
            "error_preview": repr(exc),
            "log": summarize_log_delta(read_log_delta(args.log_file, before)),
        }
    elapsed = time.perf_counter() - start
    log_summary = summarize_log_delta(read_log_delta(args.log_file, before))

    result: dict[str, Any] = {
        "name": name,
        "http_status": status,
        "ok": 200 <= status < 300,
        "elapsed_sec": round(elapsed, 3),
        "request_kb": round(payload_bytes / 1024, 2),
        "log": log_summary,
    }

    if parsed:
        fields = extract_completion_fields(parsed, raw)
        result.update({key: value for key, value in fields.items() if key != "message"})
        result["ok"] = expected_behavior_passed(name, result)
    else:
        result.update({"error_preview": content_preview(raw, 500)})

    return result


def expected_behavior_passed(name: str, result: dict[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    if name == "basic":
        return result.get("finish_reason") != "length" and result.get("content_preview", "").strip() == "OK"
    if name == "schema-bloat":
        return result.get("finish_reason") == "tool_calls" and result.get("tool_call_count", 0) >= 1
    if name == "prefill":
        return result.get("finish_reason") != "length" and "PREFILL_OK" in result.get("content_preview", "")
    return True


def extract_completion_fields(parsed: dict[str, Any], raw: str) -> dict[str, Any]:
    choice = (parsed.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return {
        "finish_reason": choice.get("finish_reason"),
        "usage": parsed.get("usage"),
        "tool_call_count": len(tool_calls),
        "tool_call_names": [
            ((call.get("function") or {}).get("name") or call.get("name") or "") for call in tool_calls
        ],
        "content_preview": content_preview(message.get("content")),
        "raw_tool_xml": "<tool_call" in raw or "<tool_calls" in raw,
        "message": message,
    }


def run_two_tools_sequence(model: str, args: argparse.Namespace) -> dict[str, Any]:
    before = log_offset(args.log_file)
    start = time.perf_counter()
    steps: list[dict[str, Any]] = []
    total_request_bytes = 0

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Use structured tool calls when tools are requested. Return no prose.",
        },
        {
            "role": "user",
            "content": "Call get_model_status with component='llama-server'. Do not write normal text.",
        },
    ]

    for step_index in range(2):
        payload = prepare_payload(make_two_tools_step_payload(model, messages), args)
        total_request_bytes += len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
        try:
            status, parsed, raw = request_json(
                "POST",
                f"{args.base_url}/chat/completions",
                args.api_key,
                payload,
                timeout_sec=args.timeout_sec,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return {
                "name": "two-tools",
                "http_status": 0,
                "ok": False,
                "elapsed_sec": round(elapsed, 3),
                "request_kb": round(total_request_bytes / 1024, 2),
                "error_preview": repr(exc),
                "steps": steps,
                "log": summarize_log_delta(read_log_delta(args.log_file, before)),
            }

        fields = extract_completion_fields(parsed, raw) if parsed else {"error_preview": content_preview(raw, 500)}
        step = {"http_status": status, "ok": 200 <= status < 300, **fields}
        steps.append({key: value for key, value in step.items() if key != "message"})

        if not parsed or status >= 400:
            break

        message = fields.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            tool_name = (call.get("function") or {}).get("name") or "unknown"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", f"probe_step_{step_index}"),
                    "content": json.dumps({"ok": True, "tool": tool_name, "source": "probe"}),
                }
            )

        if step_index == 0:
            messages.append(
                {
                    "role": "user",
                    "content": "Now call record_metric with name='tool_probe' and value=2. Do not write normal text.",
                }
            )

    elapsed = time.perf_counter() - start
    all_tool_names: list[str] = []
    for step in steps:
        all_tool_names.extend(step.get("tool_call_names") or [])

    return {
        "name": "two-tools",
        "http_status": steps[-1]["http_status"] if steps else 0,
        "ok": len(all_tool_names) >= 2 and all(step.get("ok") for step in steps),
        "elapsed_sec": round(elapsed, 3),
        "request_kb": round(total_request_bytes / 1024, 2),
        "finish_reason": "+".join(str(step.get("finish_reason")) for step in steps),
        "tool_call_count": len(all_tool_names),
        "tool_call_names": all_tool_names,
        "steps": steps,
        "log": summarize_log_delta(read_log_delta(args.log_file, before)),
    }


def parse_tests(value: str) -> list[str]:
    known = ["basic", "two-tools", "schema-bloat", "prefill"]
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if requested == ["all"]:
        return known
    bad = [item for item in requested if item not in known]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown tests: {', '.join(bad)}; known: {', '.join(known)}")
    return requested


def print_summary(results: list[dict[str, Any]]) -> None:
    print("")
    print("Probe summary")
    print("-------------")
    header = f"{'test':<14} {'http':>5} {'sec':>8} {'reqKB':>8} {'prompt':>8} {'pp/tps':>8} {'tools':>7} finish"
    print(header)
    print("-" * len(header))
    for item in results:
        log = item.get("log") or {}
        prompt_tokens = (log.get("prompt_tokens") or [""])[-1]
        pp = log.get("prompt_processing") or log.get("prompt_eval") or []
        pp_tps = pp[-1].get("tokens_per_sec") if pp else ""
        print(
            f"{item['name']:<14} {item['http_status']:>5} {item['elapsed_sec']:>8.2f} "
            f"{item['request_kb']:>8.2f} {prompt_tokens!s:>8} {pp_tps!s:>8} "
            f"{item.get('tool_call_count', '')!s:>7} {item.get('finish_reason', '')}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", help="Model id/alias. Defaults to the first /v1/models id.")
    parser.add_argument("--tests", type=parse_tests, default=parse_tests("all"))
    parser.add_argument("--schema-kb", type=int, default=32, help="Approximate synthetic tool schema size.")
    parser.add_argument("--prefill-tokens", type=int, default=4096, help="Approximate text tokens for prefill test.")
    parser.add_argument("--timeout-sec", type=float, default=240)
    parser.add_argument("--log-file", type=pathlib.Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--no-cache-bust",
        action="store_true",
        help="Do not inject a unique probe id into prompts/tool schemas.",
    )
    parser.add_argument(
        "--append-no-think",
        action="store_true",
        help="Append Qwen's /no_think hint to the last user message in each request.",
    )
    args = parser.parse_args(argv)

    args.base_url = normalize_base_url(args.base_url)
    args.log_file = args.log_file.resolve()
    args.probe_id = None if args.no_cache_bust else dt.datetime.now().strftime("%Y%m%d%H%M%S%f")

    try:
        model = args.model or discover_model(args.base_url, args.api_key, args.timeout_sec)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {model}")
    print(f"Log file : {args.log_file}")
    print(f"Tests    : {', '.join(args.tests)}")

    results = []
    for name in args.tests:
        print(f"\nRunning {name}...", flush=True)
        try:
            result = run_one_test(name, model, args)
        except Exception as exc:
            result = {
                "name": name,
                "http_status": 0,
                "ok": False,
                "elapsed_sec": None,
                "request_kb": None,
                "error_preview": repr(exc),
                "log": summarize_log_delta(read_log_delta(args.log_file, log_offset(args.log_file))),
            }
        results.append(result)
        status = "ok" if result.get("ok") else "failed"
        print(f"  {status}: HTTP {result.get('http_status')} in {result.get('elapsed_sec')} sec")

    print_summary(results)

    output = args.output
    if output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = DEFAULT_OUT_DIR / f"openai-probe-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now().isoformat(),
        "endpoint": args.base_url,
        "model": model,
        "log_file": str(args.log_file),
        "tests": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nWrote {output}")

    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))