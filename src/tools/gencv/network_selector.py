"""
gencv/network_selector.py
"""

from __future__ import annotations

import re

from core.db import get_session, NetworkORM
from core.llm_client import get_llm_client

# ---------------------------------------------------------------------------
# Network judgment (target_position match)
# ---------------------------------------------------------------------------
# Deliberately a SEPARATE judgment call from identity selection above --
# which social links to show has nothing to do with which contact identity
# (location/visa) is used. Coupling them into one LLM call would conflate
# two independent decisions and make both harder to reason about /
# debug individually.

NETWORK_SYSTEM_PROMPT = """You are a job-function matching assistant. You will be given a job
description and a list of target job positions/functions.

Judge which one of these target positions best matches the role described
in the job description.

Output format: output ONLY the index number of the best-matching target
position, no other text, no explanation. If none of the listed positions
reasonably match the job description, output NONE instead."""

NETWORK_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Candidate target positions ({count} total)
{positions_block}

Output the index number of the best-matching target position:"""


def _fetch_all_networks(session) -> list[NetworkORM]:
    """Fetch all network rows. No governance-layer filtering -- network has
    no governance concept, every row is a candidate."""
    return session.query(NetworkORM).all()


def _collect_distinct_positions(networks: list[NetworkORM]) -> list[str]:
    """Flattens every network row's target_position array into a single
    deduplicated list, preserving first-seen order. Networks with an empty
    target_position (universal links) contribute nothing here -- they are
    handled separately in _confirm_networks(), not via LLM position
    matching."""
    seen: dict[str, None] = {}
    for network in networks:
        for position in network.target_position:
            seen.setdefault(position, None)
    return list(seen.keys())


def _llm_suggest_position(positions: list[str], job_description: str) -> str | None:
    """Single LLM call. Returns the matched target_position string, or None
    if the LLM returned NONE or the response couldn't be parsed."""
    lines = [f"[{i}] {position}" for i, position in enumerate(positions, 1)]
    positions_block = "\n".join(lines)

    user_prompt = NETWORK_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(positions),
        positions_block=positions_block,
    )

    response = get_llm_client().generate_text(
        system_prompt=NETWORK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=20,
    )

    raw = response.text.strip()
    if not raw or "NONE" in raw.upper():
        return None

    match = re.search(r"\d+", raw)
    if not match:
        return None

    idx = int(match.group())
    if 1 <= idx <= len(positions):
        return positions[idx - 1]
    return None


def _confirm_networks(
    matched_position: str | None,
    all_networks: list[NetworkORM],
) -> list[NetworkORM]:
    """Mandatory human confirmation gate for network selection. Shows which
    networks would be included based on the matched position (if any), lets
    the user accept with Enter, or type comma-separated indices to override
    which networks are included.
    """
    print("\n=== Network (social link) selection ===")

    universal_networks = [n for n in all_networks if not n.target_position]

    if matched_position is not None:
        matched_networks = [n for n in all_networks if matched_position in n.target_position]
        print(f"Matched target position: {matched_position}")
    else:
        matched_networks = []
        print(
            "[Notice] Could not match this job to any known target position -- "
            "only universal (any-position) networks are suggested by default."
        )

    suggested_networks = universal_networks + [
        n for n in matched_networks if n not in universal_networks
    ]

    print("\nAll available networks:")
    for i, network in enumerate(all_networks, 1):
        marker = " <- suggested" if network in suggested_networks else ""
        position_label = ", ".join(network.target_position) if network.target_position else "ANY (universal)"
        print(f"  [{i}] {network.network} ({network.username}) | positions: {position_label}{marker}")

    choice = input(
        "\nAccept the suggested networks? (Enter to accept / type comma-separated indices to override, "
        "or 'skip' for no networks): "
    ).strip()

    if not choice:
        print(f"  -> Using {len(suggested_networks)} suggested network(s).")
        return suggested_networks

    if choice.lower() == "skip":
        print("  -> Using no networks.")
        return []

    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        chosen = [all_networks[i - 1] for i in indices]
        print(f"  -> Using {len(chosen)} manually selected network(s).")
        return chosen
    except (ValueError, IndexError):
        raise ValueError(f"Invalid selection: {choice}")


def select_networks_from_jd(job_description: str) -> list[NetworkORM]:
    """Main entry point: LLM judgment + mandatory human confirmation.
    Returns the final list of NetworkORM rows to include in the resume.

    Independent from select_identity_from_jd() -- see module docstring for
    why these are two separate judgment calls rather than one combined call.

    Flow:
    1. Fetch all network candidates and their distinct target_position values.
    2. Single LLM call to judge which target_position best matches the job
    description (or return NONE).
    3. ALWAYS stop for human confirmation -- print the matched networks
    (universal networks are always included) + full menu, let the user
    accept with Enter, override with comma-separated indices, or select
    none.
    4. Returns an empty list (not an error) when no networks exist or none
    are selected -- unlike identity, having zero networks is a valid
    resume state (social_networks is an optional RenderCV field).
    """
    with get_session() as session:
        networks = _fetch_all_networks(session)

    if not networks:
        print("[Notice] The network table is empty -- resume will have no social_networks section.")
        return []

    positions = _collect_distinct_positions(networks)
    matched_position = _llm_suggest_position(positions, job_description) if positions else None

    return _confirm_networks(matched_position, networks)

