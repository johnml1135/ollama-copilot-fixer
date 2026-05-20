---
name: local-context-budget
description: 'Use when: local Copilot, LLM Gateway, Qwen, reduced context, prompt bloat, token budget, context window, 24GB GPU, or local chats time out from huge prompts.'
argument-hint: '[symptom or target token budget]'
---

# Local Context Budget

Keep local sessions inside the effective context budget of a single 24 GB GPU.

## Procedure

1. Pick a budget tied to the model's effective context, not its claimed maximum.
   - On a 24 GB GPU running Qwen3.x, use **16K for tool-heavy local agent work**, treat **32K as the high-quality upper comfort zone**, **64K as a soft research ceiling**, and **>100K as degraded**.
   - Keep `github.copilot.llm-gateway.defaultMaxTokens` below llama-server `--ctx-size` so generation, tool continuation, and reserve space still fit.
2. Inspect prompt growth before changing settings.
   - Run `scripts\inspect-copilot-context.ps1` when available.
   - Otherwise inspect Copilot Chat debug logs for tool-schema bloat, repeated tool results, full-file pastes, and accumulated chat history.
3. Reduce context pressure in this order.
   - Start a fresh chat and carry forward only a compact checkpoint.
   - Ask for paths and signatures before reading full files.
   - Prefer range-limited reads over full-file reads.
   - Clear stale tool results and repeated outputs.
   - Trim unused tools and MCP surfaces.
   - Delegate exploration only when a short summary can come back.
4. Reserve output space.
   - Keep `defaultMaxOutputTokens` around `4096` unless a task truly needs more.
   - Ask for concise findings, not raw transcript dumps.
5. Place the most important context at the start and end of the prompt.
   - Re-state the immediate task right before the model's turn.

## Output Contract

When reporting context-budget findings, return:

- Current advertised context vs. recommended effective ceiling.
- Largest likely context sources observed.
- The smallest change that reduces prompt size.
- Any speed or correctness tradeoff that change introduces.

## References

- [Evidence and rationale](./references/evidence.md)
