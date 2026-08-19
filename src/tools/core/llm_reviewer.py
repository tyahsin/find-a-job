"""
core/llm_reviewer.py
Version: v1 (2026-08-19)

Shared parser for interactive reviewer confirmation gates, used by
experience_rewriter.py, claim_ranker.py, skill_grouper.py, and
summary_writer.py.

Input contract (final, per user confirmation 2026-08-19):
- "" (Enter): accept the current proposal.
- "r": regenerate, no steering instruction.
- "r <text>": regenerate, <text> is an optional steering instruction.
- "o <text>": manual override with <text>.
- "o" alone (no text): invalid -- caller displays an error and re-prompts.
- "skip": request omission -- only reviewers that support omission
  (currently summary_writer.py) should act on this; others should treat
  it as invalid if they don't support skipping.
- anything else non-empty: invalid -- caller displays an error and
  re-prompts. (No implicit fallback to manual override for unrecognized
  text -- every input must match one of the commands above.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReviewAction = Literal["accept", "regenerate", "override", "invalid", "skip"]


@dataclass(frozen=True)
class ReviewDecision:
    action: ReviewAction
    value: str | None = None


def parse_review_input(raw: str) -> ReviewDecision:
    """Parses one raw reviewer response into a ReviewDecision. Performs no
    I/O -- callers own input()/print() and decide how to react to each
    action (including whether "skip" is supported in that context)."""
    normalized = raw.strip()
    if not normalized:
        return ReviewDecision("accept")

    command, separator, remainder = normalized.partition(" ")
    command_lower = command.lower()
    value = remainder.strip() if separator else None

    if command_lower == "r":
        return ReviewDecision("regenerate", value or None)

    if command_lower == "o":
        if value:
            return ReviewDecision("override", value)
        return ReviewDecision("invalid", "The 'o' command requires replacement text: o <text>")

    if command_lower == "skip" and value is None:
        return ReviewDecision("skip")

    return ReviewDecision(
        "invalid",
        "Unrecognized input. Use Enter to accept, 'r' or 'r <text>' to regenerate, "
        "'o <text>' to override manually, or 'skip' where supported.",
    )
