"""
core/llm_reviewer.py
"""

from __future__ import annotations

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