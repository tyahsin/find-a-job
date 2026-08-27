"""
core/llm_reviewer.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ReviewAction = Literal["accept", "regenerate", "override", "invalid", "skip"]

from core.llm_client import get_llm_client
from onboard.prompts import build_regenerate_user_prompt


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


def _regenerate(system_prompt: str, draft: str, steering_note: str | None) -> str:
    """Shared by confirm_field and confirm_decision: asks the LLM to redo its
    previous output, optionally steered by a note from the reviewer. No
    parsing/validation here -- that's the caller's job (confirm_decision
    needs to check the result still matches the decision grammar; confirm_field
    accepts whatever text comes back)."""
    user_prompt = build_regenerate_user_prompt(previous_output=draft, steering_note=steering_note)
    return get_llm_client().generate_text(system_prompt=system_prompt, user_prompt=user_prompt).text


def confirm_field(draft: str, system_prompt: str, hint: str | None = None) -> str:
    while True:
        print(draft if not hint else hint.replace("value", draft))
        raw = input(
            f"(Enter to accept / 'r' or 'r <text>' to regenerate / 'o <text>' to override/ 'skip' to skip): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            if system_prompt:
                print(f"  Regenerating...\n")
                draft = _regenerate(system_prompt, draft, decision.value)
            else:
                return ValueError("Missing system prompt for regeneration.")
        elif decision.action == "override":
            print(f"  -> Using manually entered text\n")
            return decision.value
        elif decision.action == "accept":
            print(f"  -> Using this LLM-(re)generated text\n")
            return draft
        elif decision.action == "skip":
            print(f"  -> Moving on to the next entry...\n")
            return None


def confirm_fields_multiple(drafts: list[tuple], field_order: list, system_prompts: list):
    """
    Confirm or edit the LLM suggested entries one by one. User can either keep,
    regenerate, override, or skip the entry of more than 1 value to evaluate.
    """
    finalized = []
    for idx, items in enumerate(drafts, start=1):
        groups = []
        for item, field_name, system_prompt in zip(items, field_order, system_prompts):
            edited = confirm_field(item, system_prompt, f"Accept item {idx}'s {field_name} [value]? ")
            if not edited:  # if the first item content is None, the whole entry will be dropped
                break
            groups.append(edited)
        if groups:
            finalized.append(tuple(groups))
    return finalized


# ---------------------------------------------------------------------------
# KEEP/MERGE/DROP group-decision confirmation (used by gencv/claim_ranker.py)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^(\d+):(.+)$")
_LABEL_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+){0,2}$")  # 1-3 words, letters only, no symbols


@dataclass(frozen=True)
class GroupDecision:
    keep: list[int]
    merge_groups: list[list[int]]
    labels: dict[int, str]  # index -> short capability/reason label


def parse_decision_text(text: str, valid_indices: set[int]) -> GroupDecision | None:
    """Parses a KEEP/MERGE/DROP decision text where each record number carries
    a short label as 'n:label' (tokens within a line separated by ';', NOT
    ',' -- a label/reason may itself contain a comma). One KEEP line (0+
    entries, or NONE), zero or more MERGE lines (2+ entries each; a 1-entry
    MERGE line is treated as a KEEP instead of failing), zero or more DROP
    lines (0+ entries, or NONE). Every valid index must appear in exactly one
    of KEEP/MERGE/DROP, and every label must be 1-3 words with no symbols.
    Returns None if the text violates any of this."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None

    keep: list[int] = []
    merge_groups: list[list[int]] = []
    labels: dict[int, str] = {}
    seen: set[int] = set()

    def _parse_tokens(body):
        nums = []
        for tok in (t.strip() for t in body.split(";")):
            if not tok:
                continue
            m = _TOKEN_RE.match(tok)
            if not m:
                return None
            idx, label = int(m.group(1)), m.group(2).strip()
            if idx not in valid_indices or idx in seen or not _LABEL_RE.match(label):
                return None
            nums.append(idx)
            seen.add(idx)
            labels[idx] = label
        return nums

    for line in lines:
        upper = line.upper()
        if upper.startswith("KEEP"):
            body = line[4:].strip()
            if body.upper() == "NONE":
                continue
            nums = _parse_tokens(body)
            if nums is None:
                return None
            keep.extend(nums)
        elif upper.startswith("MERGE"):
            body = line[5:].strip()
            if body.upper() == "NONE":
                continue
            nums = _parse_tokens(body)
            if nums is None:
                return None
            if len(nums) == 1:
                # LLM proposed a "merge" with only one member -- there's
                # nothing to merge it with, so treat it as a normal KEEP
                # instead of failing the whole decision over a labeling slip.
                keep.extend(nums)
            else:
                merge_groups.append(nums)
        elif upper.startswith("DROP"):
            body = line[4:].strip()
            if body.upper() == "NONE":
                continue
            nums = _parse_tokens(body)
            if nums is None:
                return None
        else:
            return None

    if seen != valid_indices:
        return None  # every record must be accounted for in KEEP, MERGE, or DROP

    return GroupDecision(keep=keep, merge_groups=merge_groups, labels=labels)


def confirm_decision(draft_text: str, system_prompt: str, valid_indices: set[int], render_fn=None) -> GroupDecision | None:
    """Confirms an entire KEEP/MERGE/DROP decision for a group of candidate
    records at once -- not one record at a time. render_fn, given the
    currently parsed GroupDecision, returns a human-readable preview to print
    each iteration (falls back to the raw decision text if parsing fails).
    No manual override -- 'r'/'r <text>' regenerates via the LLM, 'skip'
    drops every record in the group."""
    draft = draft_text
    while True:
        parsed_for_display = parse_decision_text(draft, valid_indices)
        if render_fn and parsed_for_display is not None:
            print(render_fn(parsed_for_display))
        else:
            print(draft)

        raw = input(
            "(Enter to accept / 'r' or 'r <text>' to regenerate / 'skip' to skip all): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            print("  Regenerating...\n")
            new_draft = _regenerate(system_prompt, draft, decision.value)
            if parse_decision_text(new_draft, valid_indices) is None:
                print(f"  [Notice] Regenerated output didn't match the decision grammar ({new_draft!r}) -- try again.\n")
                continue
            draft = new_draft
            continue

        if decision.action == "override":
            print("  [Error] Manual override isn't available here -- use 'r' to regenerate instead.\n")
            continue

        if decision.action == "accept":
            print("  -> Using this decision\n")
            return parse_decision_text(draft, valid_indices)

        if decision.action == "skip":
            print("  -> Dropping all candidates in this group...\n")
            return None

        print(f"  [Error] {decision.value}")