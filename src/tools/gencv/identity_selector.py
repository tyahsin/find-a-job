"""
identity_selector.py
Version: v3 (2026-08-16)

Changelog:
- v3 (2026-08-16): _confirm_networks() now treats a network with an EMPTY
target_position as UNIVERSAL -- it applies to any job title (e.g. a
personal portfolio site or a general GitHub profile not tied to one
specific role) and is automatically included in the suggested set for
every job description, regardless of what target_position the LLM
matched. Previously, `matched_position in network.target_position`
evaluated to False for any network with an empty target_position list,
which meant such networks were NEVER auto-suggested -- the exact
opposite of "applies to any job title". This paired with
onboard/cli.py v1.10's _resolve_target_positions(), which now allows
leaving target_position blank at add-network time (previously at least
one job title was required). The confirmation menu now also labels
universal networks explicitly as "ANY (universal)" instead of an empty
string, so the distinction is visible at review time.
- v2 (2026-08-15): Added extract_headline_from_jd() -- extracts the job
title verbatim from the job description text, if explicitly present.
Deliberately EXTRACTION, not generation: mirrors the "no overclaiming"
principle from onboard/prompts.py's _BASE_SYSTEM_RULES (never add a fact
not already present in the source material), applied here to the
headline field specifically. No mandatory confirmation gate for this one
(unlike the two selectors below) -- see function docstring for rationale.
- v1 (2026-08-15): Initial select_identity_from_jd() and
select_networks_from_jd(), each with a mandatory human confirmation gate.

Three independent judgment/extraction functions, used by pipeline.py before
assembling the final resume YAML:

1. select_identity_from_jd(job_description) -> IdentityORM
LLM judgment call (location/visa_status match) + MANDATORY human
confirmation gate -- selection is never fully silent/automated,
mirroring the safety principle already used in
gencv/experience_rewriter.py (LLM proposes, human confirms every time).

2. select_networks_from_jd(job_description) -> list[NetworkORM]
Independently judges which target_position this job falls under, then
returns all network rows whose target_position array contains that
match, UNIONED with every network whose target_position is empty
(universal links). Separate judgment axis from identity -- which contact
identity to use has nothing to do with which social links are relevant
to the job function.

3. extract_headline_from_jd(job_description) -> str | None
Extracts (not generates) the job title verbatim from the JD text. No
confirmation gate -- lower stakes than the two selectors above, closer
to mechanical text extraction than a risk-bearing choice.

All prompts are English-only per project-wide language policy (see
onboard/prompts.py's LANGUAGE POLICY note) -- system prompts, user prompts,
and all CLI-facing confirmation strings.
"""

from __future__ import annotations

import re

from core.db import get_session, IdentityORM, NetworkORM
from core.llm_client import get_llm_client

# ---------------------------------------------------------------------------
# Identity judgment (location / visa_status)
# ---------------------------------------------------------------------------

IDENTITY_SYSTEM_PROMPT = """You are an identity-matching assistant. You will be given a job description
and a list of the candidate's identity options (each with an index,
location, and visa status).

Judge which identity is the best fit for applying to this job. Base your
judgment on:
1. Whether the job's required work location matches or is compatible with
the identity's location (e.g. same city, same country, or the job
explicitly allows remote work).
2. Whether the job explicitly requires local work authorization, or
explicitly does/does not offer visa sponsorship, matched against the
identity's visa_status.

Output format: output ONLY the index number of the best-fit identity, no
other text, no explanation. If the job description gives no usable clue
about location or visa requirements, output UNCLEAR instead."""

IDENTITY_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Identity options ({count} total)
{identities_block}

Output the index number of the best-fit identity:"""


def _fetch_all_identities(session) -> list[IdentityORM]:
    """Fetch all identity rows. No governance-layer filtering needed --
    identity has no governance concept, every row is a candidate."""
    return session.query(IdentityORM).all()


def _build_identity_prompt(
    identities: list[IdentityORM], job_description: str
) -> tuple[str, dict[int, IdentityORM]]:
    """Builds a single batch prompt listing every identity option by index.
    Returns (prompt, {index: IdentityORM})."""
    index_to_identity: dict[int, IdentityORM] = {}
    lines = []
    for i, identity in enumerate(identities, 1):
        index_to_identity[i] = identity
        lines.append(
            f"[{i}] Name: {identity.name}\n"
            f"    Location: {identity.location or 'not specified'}\n"
            f"    Visa status: {identity.visa_status or 'not specified'}"
        )
    identities_block = "\n\n".join(lines)
    user_prompt = IDENTITY_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(identities),
        identities_block=identities_block,
    )
    return user_prompt, index_to_identity


def _llm_suggest_identity(
    identities: list[IdentityORM], job_description: str
) -> IdentityORM | None:
    """Single LLM call. Returns the suggested IdentityORM, or None if the
    LLM returned UNCLEAR or the response couldn't be parsed -- caller should
    fall back to is_default=True rather than picking arbitrarily."""
    user_prompt, index_to_identity = _build_identity_prompt(identities, job_description)

    response = get_llm_client().generate_text(
        system_prompt=IDENTITY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=20,
    )

    raw = response.text.strip()
    if not raw or "UNCLEAR" in raw.upper():
        return None

    match = re.search(r"\d+", raw)
    if not match:
        return None

    idx = int(match.group())
    return index_to_identity.get(idx)


def _confirm_identity(
    suggested: IdentityORM | None,
    all_identities: list[IdentityORM],
    default_identity: IdentityORM | None,
) -> IdentityORM:
    """Mandatory human confirmation gate -- cannot be skipped. Shows the LLM
    suggestion (if any), lets the user press Enter to accept, or type an
    index to override."""
    print("\n=== Identity selection ===")

    if suggested is not None:
        print(f"Suggested identity for this job: {suggested.name}")
        print(f"  Location: {suggested.location or 'not specified'}")
        print(f"  Visa status: {suggested.visa_status or 'not specified'}")
    else:
        fallback_label = default_identity.name if default_identity else "(no default identity set)"
        print(f"[Notice] Not enough information in the job description to judge -- falling back to default identity: {fallback_label}")
        suggested = default_identity

    print("\nAvailable identities:")
    for i, identity in enumerate(all_identities, 1):
        marker = " <- current suggestion" if identity is suggested else ""
        print(f"  [{i}] {identity.name} | {identity.location or 'not specified'} | {identity.visa_status or 'not specified'}{marker}")

    choice = input("\nAccept the suggested identity? (Enter to accept / type an index to override): ").strip()

    if not choice:
        if suggested is None:
            raise ValueError(
                "No suggested identity available and no is_default set -- "
                "specify an identity manually or run add-identity to set a default first."
            )
        print(f"  -> Using: {suggested.name}")
        return suggested

    try:
        idx = int(choice)
        chosen = all_identities[idx - 1]
        print(f"  -> Switched to: {chosen.name}")
        return chosen
    except (ValueError, IndexError):
        raise ValueError(f"Invalid selection: {choice}")


def select_identity_from_jd(job_description: str) -> IdentityORM:
    """Main entry point: LLM judgment + mandatory human confirmation.
    Returns the final IdentityORM to use.

    Flow:
    1. Fetch all identity candidates (no governance filtering).
    2. Single LLM call to judge the best-fit identity from the job
    description (or return UNCLEAR).
    3. ALWAYS stop for human confirmation, regardless of LLM outcome --
    print the suggestion + full menu, let the user accept with Enter or
    override by typing an index.
    4. Falls back to is_default=True when the LLM can't decide, still
    subject to confirmation.
    """
    with get_session() as session:
        identities = _fetch_all_identities(session)

    if not identities:
        raise ValueError(
            "The identity table is empty -- run add-identity to create at least one identity first."
        )

    default_identity = next((i for i in identities if i.is_default), None)

    if len(identities) == 1:
        # Only one identity exists -- still go through the confirmation
        # gate for a consistent UX, but skip the LLM call entirely since
        # there's no choice to make.
        suggested = identities[0]
    else:
        suggested = _llm_suggest_identity(identities, job_description)

    return _confirm_identity(suggested, identities, default_identity)


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

    (v3) A network with an empty target_position is treated as UNIVERSAL --
    it applies to any job title (e.g. a personal portfolio site or a
    general GitHub profile that isn't tied to one specific role) and is
    ALWAYS included in the suggestion, regardless of what target_position
    the LLM matched for this JD. Previously, `matched_position in
    network.target_position` evaluated to False for an empty
    target_position, so such networks were never auto-suggested -- the
    opposite of the intended "applies to any job" behavior.
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
        "or 'none' for no networks): "
    ).strip()

    if not choice:
        print(f"  -> Using {len(suggested_networks)} suggested network(s).")
        return suggested_networks

    if choice.lower() == "none":
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


# ---------------------------------------------------------------------------
# Headline extraction (job title, verbatim from the JD)
# ---------------------------------------------------------------------------
# Deliberately EXTRACTION, not generation -- the LLM may only pull out a job
# title that is explicitly written in the job description text, never
# invent, rephrase into a "nicer" synonym, or infer a title from the
# candidate's own experience. This mirrors the "no overclaiming" governance
# principle already applied throughout onboard/prompts.py's
# _BASE_SYSTEM_RULES (rule 1: never add a fact not already present in the
# source material) -- here applied specifically to the headline field.

# No mandatory human confirmation gate here, unlike select_identity_from_jd()
# / select_networks_from_jd() -- those two carry real risk if picked wrong
# (wrong contact info / wrong visa disclosure sent out), whereas extracting
# a title string that's already written verbatim in the JD is closer to
# mechanical text extraction with much lower stakes. If this turns out to
# need a confirmation step in practice, it can be added the same way as the
# other two selectors.

HEADLINE_SYSTEM_PROMPT = """You will be given a job description. Your task is to extract the job
title/position name, if and only if it is explicitly written in the text.

Strict rules (must follow):
1. Only extract a title that literally appears in the job description --
do not paraphrase it into a different wording, do not "clean it up" into
a more standard-sounding title, and do not translate it.
2. Do not infer or guess a title from the responsibilities/requirements
described if no explicit title is stated anywhere in the text.
3. If multiple title-like phrases appear (e.g. both a posting title and an
internal team name), pick the one that most clearly reads as the actual
job title (typically the one near the top of the posting).

Output format: output ONLY the extracted title text, exactly as written in
the source, no other text, no explanation, no quotation marks. If no
explicit job title is present anywhere in the text, output NONE instead."""

HEADLINE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

Extract the job title (output NONE if none is explicitly stated):"""


def extract_headline_from_jd(job_description: str) -> str | None:
    """Extracts the job title verbatim from the job description, if one is
    explicitly present. Returns None if no explicit title is found -- the
    caller (pipeline.py) should treat None as "omit the headline field"
    rather than falling back to a guess or a hardcoded default."""
    user_prompt = HEADLINE_USER_PROMPT_TEMPLATE.format(job_description=job_description)

    response = get_llm_client().generate_text(
        system_prompt=HEADLINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=50,
    )

    raw = response.text.strip()
    if not raw or raw.upper() == "NONE":
        return None
    return raw
