---
name: local-subagent-delegation
description: 'Use when local Qwen/Copilot should delegate isolated research, review, or summarization to subagents without bloating parent context.'
argument-hint: '[subtask to delegate]'
---

# Local Subagent Delegation

Use subagents for context isolation, not for every task.

## Procedure

1. Delegate only when isolation helps.
   - Good targets: codebase exploration, independent review, log inspection, alternative-plan comparison, focused documentation lookup.
   - Poor targets: tiny edits, single-file changes, anything where the subagent would need the full parent transcript.
   - Default to no delegation for short tasks.
2. Give the subagent a narrow contract.
   - Exact question.
   - Only the files, commands, or symptoms it needs.
   - Output format and length cap.
   - Whether editing is allowed (default: read-only).
3. Require a compact return format.
   - Paths, facts, risks, recommendation.
   - No raw logs, no full files, no long diffs.
   - Stop as soon as evidence is sufficient.
4. Synthesize in the parent context.
   - Treat the subagent result as a compressed evidence packet.
   - Re-open original files only when the summary is ambiguous or the change is high-risk.
   - Avoid nested subagents unless the user explicitly wants divide-and-conquer.
5. Hand off long sessions instead of growing them.
   - If the parent context is already large, prefer a session handoff over dispatching a subagent that has to re-read everything.

## Output Contract

For each subagent result, preserve:

- Subtask name.
- Files or sources inspected.
- Findings in priority order.
- Recommendation.
- Confidence or uncertainty.

## References

- [Evidence and rationale](./references/evidence.md)
