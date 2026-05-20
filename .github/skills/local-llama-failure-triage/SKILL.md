---
name: local-llama-failure-triage
description: 'Use when: local Qwen, llama-server, ik_llama.cpp, or Copilot local agent died, stalled, timed out, was suspected to OOM, or needs log-based failure diagnosis.'
argument-hint: '[symptom, timestamp, or failing request]'
---

# Local Llama Failure Triage

Determine whether a local-model failure was a real crash, an OS/GPU resource issue, or an oversized-request slowdown/cancellation.

## Procedure

1. Preserve evidence before restarting anything.
   - Copy or rename `logs\llama-server.log`, `logs\llama-server.log.err`, `logs\llama-server.info.json`, and any relevant `logs\openai-probe-*.json` files.
   - Do this first: `scripts\start-server.ps1` truncates the live stdout/stderr log files on restart.
2. Classify the failure surface.
   - If `llama-server.exe` is still running and `/health` or `/v1/models` responds, treat it as a client, gateway, prompt-size, or parser problem until proven otherwise.
   - If the process is gone, determine whether it was a clean stop, an application crash, or an OS/GPU fault.
   - Use `logs\llama-server.info.json` to confirm the exact runtime config before changing settings.
3. Check the highest-signal repo diagnostics.
   - Run `scripts\status-server.ps1` for process, endpoint, throughput, and GPU status.
   - Run `scripts\inspect-copilot-context.ps1` to compare Copilot request size with llama prompt processing.
   - Inspect `logs\llama-server.log` for `prompt eval time`, `eval time`, long prompt checkpoints, slot transitions, cancellations, or repeated load/unload.
   - Treat `logs\llama-server.log.err` as low-trust: it can contain echoed model output or tool payloads, not just server diagnostics.
4. Prefer server telemetry over regex parsing when the server is alive.
   - Query `/metrics` for prompt throughput, generation throughput, requests in flight, and observed context high watermarks.
   - Query `/slots` for whether a slot is actually busy, idle, or stuck in prompt processing.
5. Check system-level crash evidence.
   - Inspect Windows Event Viewer or `Get-WinEvent` for `llama-server.exe`, Windows Error Reporting, `Resource-Exhaustion-Detector`, and `nvlddmkm` events.
   - No matching OS or GPU fault plus a live server usually means overload, timeout, or cancellation, not a hard OOM.
6. Diagnose by pattern.
   - Live process plus giant Copilot `inputTokens` and long `prompt eval time` with no OS crash events means slow prefill or context overload.
   - Missing process plus WER or Application Error events for `llama-server.exe` means a real process crash.
   - GPU reset or OOM-detector events means a system resource problem.
   - Failures concentrated on tool-heavy turns usually mean tool-schema bloat, parser/template issues, or prompt growth before they mean hardware failure.
7. Change the smallest load-bearing setting first.
   - Reduce `github.copilot.llm-gateway.defaultMaxTokens` before changing models; on this repo, start around `16384` for tool-heavy local work.
   - Keep `parallelToolCalling` off until the model and template path are proven stable.
   - Reduce `--ctx-size` or increase VRAM headroom with `--fit` or `--fit-margin` when the model is near the edge.
   - Use one preserved failing artifact to validate the next change instead of retrying blind.

## Output Contract

Return:

- Whether the failure is a real crash, OS/GPU resource fault, slow prefill/context overload, or client/gateway failure.
- The strongest evidence supporting that classification.
- The smallest next change to try.
- One validation step that can disconfirm the diagnosis.

## References

- [Evidence and rationale](./references/evidence.md)