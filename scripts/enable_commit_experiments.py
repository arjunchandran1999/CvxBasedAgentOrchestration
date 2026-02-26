#!/usr/bin/env python3
"""
Modify .gitignore so experiment outputs can be committed.

Specifically updates the block:
  # Generated runs / telemetry (project defaults)
  runs/
  bench_runs/
  *.log
  *.jsonl

to STOP ignoring:
  - runs/
  - bench_runs/
  - *.jsonl

while keeping:
  - *.log

Usage (from repo root):
  python3 scripts/enable_commit_experiments.py
"""

from __future__ import annotations

from pathlib import Path


BLOCK_HEADER = "# Generated runs / telemetry (project defaults)"
REMOVE_PATTERNS = {"runs/", "bench_runs/", "*.jsonl"}


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / ".gitignore"
    if not path.exists():
        raise SystemExit(f"Expected .gitignore at: {path}")

    original = path.read_text(encoding="utf-8").splitlines(True)  # keep line endings

    try:
        start = next(i for i, line in enumerate(original) if line.rstrip("\n") == BLOCK_HEADER)
    except StopIteration:
        raise SystemExit(f"Could not find block header in .gitignore: {BLOCK_HEADER!r}")

    # Block continues until blank line OR next comment header (but ignore inline comments within block).
    end = start + 1
    while end < len(original):
        s = original[end].strip()
        if s == "":
            break
        if s.startswith("#"):
            break
        end += 1

    block = [ln.rstrip("\n") for ln in original[start + 1 : end]]
    kept = [ln for ln in block if ln.strip() and ln.strip() not in REMOVE_PATTERNS]

    # Ensure *.log stays ignored (if it was there), but don't duplicate.
    if "*.log" not in [k.strip() for k in kept]:
        kept.append("*.log")

    replacement = []
    replacement.append(BLOCK_HEADER + "\n")
    replacement.append("# NOTE: runs/, bench_runs/, and *.jsonl are NOT ignored so experiment artifacts can be committed.\n")
    replacement.extend([k.rstrip("\n") + "\n" for k in kept])
    # Preserve the blank line delimiter if it existed
    if end < len(original) and original[end].strip() == "":
        replacement.append(original[end])
        end += 1
    else:
        replacement.append("\n")

    updated = original[:start] + replacement + original[end:]

    if updated == original:
        print("No changes needed; .gitignore already allows committing experiment outputs.")
        return 0

    backup = path.with_suffix(".gitignore.bak")
    backup.write_text("".join(original), encoding="utf-8")
    path.write_text("".join(updated), encoding="utf-8")

    removed = ", ".join(sorted(REMOVE_PATTERNS))
    print(f"Updated {path}")
    print(f"- Removed ignore patterns from '{BLOCK_HEADER}': {removed}")
    print(f"- Backup written to: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

