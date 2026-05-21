---
name: local-tool-reliability
description: 'Use when local Copilot/Qwen tool calls fail, malformed tool JSON appears, tools are skipped, or LLM Gateway agent mode is flaky.'
argument-hint: '[tool-call symptom or error message]'
---

# Local Tool Reliability

Debug and stabilize local-model tool calling before changing application code.

## Procedure

1. Separate server health from agent-tool health.
   - Confirm `llama-server` is running and `/v1/models` responds.
   - Send a tiny direct chat-completions request to confirm basic generation.
   - If the server is healthy but Copilot fails, suspect gateway, prompt size, or tool-set bloat next.
2. Check local-model gateway settings.
   - `github.copilot.llm-gateway.agentTemperature: 0.0` for tool mode.
   - Start with `parallelToolCalling: false` for Qwen.
   - Keep `enableToolCalling: true` unless isolating a basic chat failure.
   - Use a long enough `requestTimeout` for large local prompts.
3. Check Qwen llama-server launch settings.
   - Use Jinja chat templates (`--jinja`).
   - Preserve Qwen thinking when needed for multi-turn tool use (`preserve_thinking: true`).
   - Use the `qwen3_coder` tool parser where available.
   - Keep KV cache at `q8_0` when reliability matters.
4. Trim the tool surface.
   - Disable MCP servers and extension tools the task does not need.
   - Prefer one well-named tool over overlapping tools.
   - Make argument names and types unambiguous.
   - Use absolute paths.
   - Prefer plain-text or search-replace edit formats over giant JSON payloads.
5. Reduce the prompt before blaming the parser.
   - Start a fresh chat and retry with a minimal reproduction.

## Output Contract

Return:

- Whether the failure is server, gateway, parser, tool-set, or context-size.
- The smallest reliability setting change to try.
- A validation step.
- Whether the change trades speed for correctness.

## References

- [Evidence and rationale](./references/evidence.md)
