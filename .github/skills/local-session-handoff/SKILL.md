---
name: local-session-handoff
description: 'Use when a local Copilot/Qwen chat is too long and needs compaction, checkpointing, restart, or a handoff summary.'
argument-hint: '[current work to checkpoint]'
---

# Local Session Handoff

Compress a long local-agent session into a short checkpoint for a fresh chat.

## Procedure

1. Identify what must survive the handoff.
   - User goal and **newest** request (state it verbatim).
   - Decisions already made.
   - Files changed or intentionally left alone.
   - Commands run and validation results (pass/fail, not raw output).
   - Known blockers and the next concrete action.
2. Drop low-value context aggressively.
   - No raw terminal logs unless an exact line matters.
   - No full file contents — paths and ranges only.
   - No completed or superseded plans.
   - No repeated background explanation.
3. Make the checkpoint actionable.
   - Use absolute or workspace-relative paths and exact settings names.
   - Include the next command or file to inspect.
   - Call out user changes that must not be overwritten.
4. Front-load and back-load the critical facts.
   - State the goal and the next step both at the **top** and the **bottom** of the checkpoint.
5. Persist the checkpoint outside chat when possible.
   - Save to a NOTES.md or `/memories/session/` file so a future chat can re-read it without paying for the whole transcript.

## Output Contract

Return a handoff using these headings, in this order:

- Current Goal — one sentence; restate verbatim if the user just gave one.
- Completed — bullet list, one line each.
- Changed Files — paths with one-line summary.
- Important Decisions — what was chosen and why.
- Validation — commands run and pass or fail.
- Next Step — single concrete action.
- Risks — anything fragile, partially done, or user-modified.

## References

- [Evidence and rationale](./references/evidence.md)
