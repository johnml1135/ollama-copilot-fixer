# Evidence

- Repo behavior: `scripts/start-server.ps1` truncates `logs/llama-server.log` and `logs/llama-server.log.err` on restart, so preserving logs before any restart is mandatory for forensics.
- Repo behavior: `scripts/start-server.ps1` already launches the server with `--metrics --verbose`, so `/metrics` is available telemetry on this setup, not a theoretical upstream option.
- Repo workflow: `scripts/inspect-copilot-context.ps1` is the fastest way to compare Copilot request growth against local llama prompt processing.
- Repo finding: on this machine, very large prompt-prefill events and deep context checkpoints were observed without matching Windows crash, OOM-detector, or GPU-reset events, so treat “the model died” as overload or cancellation until OS evidence disproves it.
- Upstream llama.cpp server docs: `/metrics` exposes prompt and generation throughput plus request backlog; `/slots` exposes per-slot processing state; responses can include `timings` with `prompt_per_second` and `predicted_per_second`; `/v1/messages/count_tokens` can estimate request size before inference. Source: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Upstream llama.cpp performance tips: verify CUDA offload from load logs and keep thread counts near physical cores; oversaturating CPU threads can reduce token generation speed. Source: <https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md>
- ik_llama.cpp parameter docs: `--fit`, `--fit-margin`, `--dry-run`, `--ctx-size`, `--cache-ram`, KV quantization, and `--peg` are the main knobs for memory headroom, cache pressure, and Qwen tool-call edge cases. Source: <https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md>

Full source list for the shared context-budget and local-agent guidance: [shared references](../../REFERENCES.md)