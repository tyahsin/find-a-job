"""
core/llm_reviewer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReviewAction = Literal["accept", "regenerate", "override", "invalid", "skip"]

from core.llm_client import get_llm_client
from core.prompts import build_regenerate_user_prompt

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


def _format_fields_block(fields: dict, field_order: list[str]) -> str:
    lines = []
    for key in field_order:
        lines.append(f"  {key}: {fields.get(key)}")
    return "\n".join(lines)


def _parse_overrides(raw_value: str, field_order: list[str]) -> tuple[dict, list[str]]:
    """Parses 'key=value;key2=value2' into a dict, validated against
    field_order. Returns (overrides, unknown_keys) -- unknown_keys is
    non-empty if the user typed a key that isn't in field_order, which
    confirm_fields_once() treats as an error and re-prompts for, rather
    than silently applying a partial override."""
    overrides: dict = {}
    unknown_keys: list[str] = []
    for chunk in raw_value.split(";"): # changed to ";" for issues with job_location where there's "," in between
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in field_order:
            unknown_keys.append(key)
            continue
        overrides[key] = value
    return overrides, unknown_keys

def confirm_fields_once(
    fields: dict,
    field_order: list[str] | None = None,
    header: str = "Drafted fields:",
    prompt_text: str | None = None,
) -> dict | None:
    """One-shot confirmation gate. Shows every field in `fields` (in
    `field_order` order), then loops on core.llm_reviewer.
    parse_review_input() until the user accepts, overrides, or aborts.

    (v2) Commands, shared contract:
      - Enter                    : accept all fields as currently drafted.
      - o key=value;key2=value2  : override one or more fields, then
                                    re-display the updated draft and
                                    prompt again (does not immediately
                                    return -- supports several rounds of
                                    override before accepting).
      - skip                     : abort. Returns None -- caller should
                                    write nothing.
      - anything else (including
        'r' / 'r <text>')        : invalid, re-prompts with an error.

    Returns the final fields dict (a copy of `fields` with any overrides
    applied), or None if the user aborted via 'skip'.
    """
    current_fields = dict(fields)
    if field_order is None:
        field_order = list(fields.keys())

    while True:
        print(f"\n{header}")
        print(_format_fields_block(current_fields, field_order))

        raw = input(
            (prompt_text or "\nAccept these fields?")
            + " (Enter to accept / 'o key=value;...' to override / 'skip' to abort): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "accept":
            return current_fields

        if decision.action == "skip":
            print("  -> Aborted, nothing will be written.")
            return None

        if decision.action == "override":
            overrides, unknown_keys = _parse_overrides(decision.value, field_order)
            if unknown_keys:
                print(
                    f"  [Error] Unrecognized field name(s): {', '.join(unknown_keys)}. "
                    f"Valid fields: {', '.join(field_order)}"
                )
                continue
            if not overrides:
                print("  [Error] Could not parse any 'key=value' pairs from your input.")
                continue
            current_fields.update(overrides)
            print(f"  -> Updated: {', '.join(overrides.keys())}")
            continue

        if decision.action == "regenerate":
            print(
                "  [Error] This confirmation gate has no regenerate option -- there is "
                "no LLM re-call to trigger here. Use Enter to accept, 'o key=value,...' "
                "to override, or 'skip' to abort."
            )
            continue

        print(f"  [Error] {decision.value}")


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
                user_prompt = build_regenerate_user_prompt(
                    previous_output=draft,
                    steering_note=decision.value,
                )
                client = get_llm_client()
                draft = client.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ).text
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
            if not edited: # if the first item content is None, the whole entry will be dropped
                break
            groups.append(edited)
        if groups:
            finalized.append(tuple(groups))
    return finalized 