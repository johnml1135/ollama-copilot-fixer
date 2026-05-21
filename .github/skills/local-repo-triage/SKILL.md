---
name: local-repo-triage
description: 'Use when a local-agent coding task needs minimal repo context, targeted file discovery, compact search, or a small implementation brief.'
argument-hint: '[task or bug to triage]'
---

# Local Repo Triage

Find the smallest file set that can support an edit without flooding the prompt.

## Procedure

1. Convert the request into concrete search targets.
   - Filenames, error text, settings keys, model names, function names, command names.
   - Prefer exact text searches and file globs over broad semantic scans when the target is concrete.
2. Build a small candidate set, not an exhaustive one.
   - List only the files likely to matter.
   - Read README or architecture docs only when they explain the workflow being changed.
   - Read source files in ranges around relevant symbols, not whole files.
3. Stop reading as soon as the implementation surface is clear.
   - Name the files to edit.
   - Name the existing pattern to follow.
   - Name the validation command or manual check.
4. Keep the working context compact.
   - Summarize findings instead of carrying raw output forward.
   - Exclude unrelated files, generated artifacts, cache directories, model weights, and dependency folders.
   - Delegate exploration only when the parent context would otherwise grow too large.

## Output Contract

Return a compact triage brief:

- Relevant files.
- Existing pattern to follow.
- Proposed edits.
- Validation plan.
- Open risk, if any.

## References

- [Evidence and rationale](./references/evidence.md)
