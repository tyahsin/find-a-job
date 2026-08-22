"""
onboard/cli.py
Version: v1.10 (2026-08-19)

Changelog:
- v1.10 (2026-08-19): Version/changelog sync fix -- no logic changes.
  The previous v1.9 header/changelog described _resolve_target_positions()
  allowing a blank first entry (returning an empty list, interpreted by
  gencv/identity_selector.py as "applies to ANY job title") and
  add_network()'s corresponding "Applies to: ANY job title" message
  branch as a SEPARATE v1.10 patch still pending manual application on
  top of this file. On review, both pieces of behavior were already
  present and working in this file's actual code (see
  _resolve_target_positions()'s docstring/loop and add_network()'s
  if/else message branch below) -- the patch had already been applied,
  but the version header and changelog were never bumped to reflect it,
  leaving the project's spec doc (v6 md) out of sync with the real file.
  This version bump is purely a metadata correction so the file's own
  header, the changelog, and the project spec doc all agree that this
  functionality is DONE, not pending. No function bodies, control flow,
  or strings were changed in this pass.
- v1.9 (2026-08-10): add-project's work/education branches now try a flat
  join-based selector FIRST (_select_existing_position_flat /
  _select_existing_degree_flat), mirroring the pattern already used by
  add-claim's project selector. If any position+company (work track) or
  degree+school (education track) rows already exist, the user sees one
  numbered table with Institution / Role columns and can pick directly --
  skipping the separate "select/create company" then "select/create
  position" steps entirely. A "[Create new]" row (option 0) is always
  offered so the user can still create a brand-new company+position (or
  school+degree) even when existing ones are shown. If NO rows exist at
  all yet, the flat selector returns None immediately (no menu shown) and
  add_project() falls back to the original _resolve_company ->
  _resolve_position (or _resolve_school -> _resolve_degree) stepwise
  flow, unchanged from prior versions. add-claim's own project selector
  and governance flow are untouched by this change.
- v1.8 (2026-08-10): Governance support_level moved to an explicit call
  (core.db.fill_pending_support_levels()) made once, right after
  add_claim()'s `with get_session()` block finishes committing. Only
  add-claim calls it -- governance conceptually only exists for claims.
- v1.7 (2026-08-10): [SUPERSEDED by v1.8] support_level via global
  after_commit hook.
- v1.6 (2026-08-10): Added Step 6 (governance) to add-claim.
- v1.5 (2026-08-10): Added add-company / add-school / add-position /
  add-degree standalone commands.
- v1.4 (2026-08-10): Removed "Start Date" column from add-claim's project
  selector -- Institution / Role / Project only.
- v1.3 (2026-08-10): add-claim's project selector flattened via two
  explicit JOIN queries.
- v1.2 (2026-08-10): work/education track selection via fixed menu.
- v1.1 (2026-08-10): Full English-only pass.
- v1.0 (2026-08-04, inferred): Original Chinese-language CLI interface.

Typer CLI entrypoint for the onboard module.

Implements:
- `add-company` / `add-school`: standalone single-entity commands for the
  upper-level entities.
- `add-position` / `add-degree`: standalone single-entity commands for the
  mid-level entities.
- `add-project`: walks through Step 0-4 (company/school -> position/degree
  -> project). As of v1.9, tries a flat Institution/Role selector first
  for the position/degree step if any exist; falls back to the original
  select-or-create steps otherwise. Never touches governance.
- `add-claim`: walks through Step 0-6 (select project -> problem -> action ->
  claim/description -> metrics -> evidence -> tags -> governance), single
  transactional write, followed by an explicit
  core.db.fill_pending_support_levels() call.

NOTE (Aug 2026): the education-track mid-level entity is called `degree`
in the actual Postgres schema (table `degree`, column `degree_id`,
`degree_name`), NOT `education`/`education_id`.

Design notes (per Aug 2026 review):
- --project-id is OPTIONAL on add-claim. If omitted, the user is walked
  through a flat, single-step project selector (see
  _select_project_interactively) that joins across both the work track
  (company/position/project) and education track (school/degree/project).
- description prompt wording no longer overlaps with action's wording.
- evidence_type is a fixed numbered menu; evidence_level is auto-derived.
- work/education track selection in add-project is a fixed numbered menu.
- add-project's position/degree step (v1.9) prefers a flat join-based
  selector over the original company-then-position drill-down, when
  existing rows are available -- consistent with add-claim's project
  selector philosophy of "join and flatten wherever the FK chain allows
  it, rather than making the user click through every parent level."
- add-company/add-school/add-position/add-degree let entities be staged
  incrementally, reusing the exact same resolver helpers as add-project.
- Tags are NOT typed by the user; LLM proposes from in-memory content,
  user reviews/edits.
- Governance: support_level is NEVER computed or asked in this file --
  the governance row is written with support_level=None, and
  core.db.fill_pending_support_levels() is called explicitly, exactly
  once, right after add-claim's own transaction commits.
- (v1.9/v1.10) target_position on a network row may be left blank --
  _resolve_target_positions() allows the very first prompt to be blank,
  returning an empty list, which gencv/identity_selector.py v3 treats as
  a UNIVERSAL link (applies to any job title). add_network()'s closing
  message explicitly distinguishes this case ("Applies to: ANY job
  title") from a link scoped to specific titles, so the user gets
  immediate confirmation of which behavior they just configured.

LANGUAGE POLICY: the entire interface is English.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from core.db import (
    ActionORM,
    MetricORM,
    ClaimORM,
    ClaimTagORM,
    CompanyORM,
    DegreeORM,
    EvidenceORM,
    GovernanceORM,
    IdentityORM,
    NetworkORM,
    PositionORM,
    ProblemORM,
    ProjectORM,
    SchoolORM,
    TagORM,
    create_row,
    fill_pending_support_levels,
    fuzzy_find_candidates,
    get_session,
    list_rows,
    select_or_create,
)
from core.source_reader import FileReadError, UnsupportedFileTypeError, read_source
from core.llm_client import get_llm_client, LLMResponse
from onboard.prompts import *

app = typer.Typer(help="Add experience entries to database.")
console = Console()

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

TRACK_OPTIONS: list[str] = [
    "work",
    "education",
]

def _prompt_date(label: str, allow_blank: bool = False) -> Optional[date]:
    while True:
        raw = typer.prompt(label, default="" if allow_blank else ...)
        if allow_blank and raw.strip() == "":
            return None
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]Invalid date format, please use YYYY-MM-DD[/red]")

def _confirm_or_edit(draft: str, label: str) -> str:
    console.print(f"\n[bold]{label} (LLM draft)[/bold]")
    console.print(draft)
    edited = typer.prompt(
        "Press Enter to accept, or type new text to override",
        default="",
        show_default=False,
    )
    return edited.strip() if edited.strip() else draft

def _select_from_fuzzy_candidates(candidates, label_field: str):
    """
    Shows fuzzy match candidates and lets the user pick one, or choose to
    create a new row instead. Returns the chosen ORM object, or None if
    the user wants to create new.
    """
    table = Table(title=f"Found {len(candidates)} similar existing record(s)")
    table.add_column("#")
    table.add_column("Name")
    table.add_column("Similarity")

    for idx, (row, score) in enumerate(candidates, start=1):
        table.add_row(str(idx), getattr(row, label_field), f"{score:.0f}%")
    table.add_row("0", "[Create new]", "")
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if choice == 0:
        return None
    return candidates[choice - 1][0]

def _select_from_list(rows, columns: list[tuple[str, str]], title: str, allow_new_label: Optional[str] = None):
    """
    Generic numbered-list selector. `columns` is a list of (header, attr_name)
    tuples. If `allow_new_label` is provided, option 0 is offered and this
    function returns None to signal "create new" instead of an existing row.

    Returns the chosen row (or None if allow_new_label was chosen).
    """
    table = Table(title=title)
    table.add_column("#")
    for header, _ in columns:
        table.add_column(header)
    for idx, row in enumerate(rows, start=1):
        values = [str(getattr(row, attr) or "") for _, attr in columns]
        table.add_row(str(idx), *values)
    if allow_new_label is not None:
        table.add_row("0", f"[{allow_new_label}]", *["" for _ in columns[1:]])
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if allow_new_label is not None and choice == 0:
        return None
    return rows[choice - 1]

def _select_from_menu(options: list[str], title: str, allow_blank: bool = False) -> Optional[str]:
    """
    Fixed-choice numbered menu -- used for fields where free-typing would
    lead to inconsistent values across records (e.g. role_scope, value_type,
    metric_confidence, work/education track). If `allow_blank` is True,
    option 0 lets the user skip the field entirely (returns None).
    """
    table = Table(title=title)
    table.add_column("#")
    table.add_column("Option")
    if allow_blank:
        table.add_row("0", "[Skip]")
    for idx, option in enumerate(options, start=1):
        table.add_row(str(idx), option)
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if allow_blank and choice == 0:
        return None
    return options[choice - 1]

def _select_existing_company(session) -> CompanyORM:
    """
    Plain numbered list of ALL existing companies (no fuzzy search --
    used when a caller needs to attach a new child row, e.g. a position,
    to a company that must already exist). Exits with an error if there
    are no companies yet.
    """
    companies = list_rows(session, CompanyORM)
    if not companies:
        console.print("[red]No company records found -- please run add-company first[/red]")
        raise typer.Exit(code=1)
    return _select_from_list(
        companies, [("Company name", "company_name")], title="Select a company"
    )

def _select_existing_school(session) -> SchoolORM:
    """
    Plain numbered list of ALL existing schools (no fuzzy search -- used
    when a caller needs to attach a new child row, e.g. a degree, to a
    school that must already exist). Exits with an error if there are no
    schools yet.
    """
    schools = list_rows(session, SchoolORM)
    if not schools:
        console.print("[red]No school records found -- please run add-school first[/red]")
        raise typer.Exit(code=1)
    return _select_from_list(
        schools, [("School name", "school_name")], title="Select a school"
    )

# ---------------------------------------------------------------------------
# add-identity: personal identity records
# ---------------------------------------------------------------------------

def _validate_phone_format(phone: str) -> bool:
    """Lightweight sanity check for RenderCV's phone format requirement --
    international format with a leading '+' (e.g. +14155551234). This is
    NOT a full libphonenumber validation (RenderCV does that authoritatively
    at render time via pydantic_extra_types.phone_numbers.PhoneNumber) --
    just catches the most common mistake (a bare local-format number like
    "970813605") before it reaches the database."""
    if not phone.startswith("+"):
        return False
    digits_only = phone[1:].replace("-", "").replace(" ", "")
    return digits_only.isdigit() and len(digits_only) >= 7

def _prompt_phone() -> str | None:
    """Prompts for phone in RenderCV's required international format,
    re-prompting on an obviously invalid entry rather than silently storing
    a value that will fail validation later at render time."""
    while True:
        phone = typer.prompt(
            "Phone in international format with country code, e.g. +14155551234 "
            "(optional, leave blank to skip)",
            default="",
            show_default=False,
        )
        if not phone.strip():
            return None
        if _validate_phone_format(phone.strip()):
            return phone.strip()
        console.print(
            "[red]Invalid format -- must start with '+' followed by the country code "
            "and number, e.g. +14155551234. Try again, or leave blank to skip.[/red]"
        )

def _handle_default_flag(session, make_default: bool) -> None:
    """If the user chooses to set this row as the new default, first flip
    any existing default identity (if any) back to FALSE, to avoid hitting
    the identity_only_one_default partial unique index."""
    if not make_default:
        return
    existing_default = (
        session.query(IdentityORM)
        .filter(IdentityORM.is_default.is_(True))
        .one_or_none()
    )
    if existing_default is not None:
        existing_default.is_default = False
        session.flush()

@app.command("add-identity")
def add_identity() -> None:
    """Interactively add a personal identity record (name/email/phone/location/visa status)."""
    with get_session() as session:
        name = typer.prompt("Name")
        email = typer.prompt("Email")
        phone = _prompt_phone()
        location = typer.prompt(
            "Current location (e.g. Tokyo, Japan; optional, leave blank to skip)", default="", show_default=False
        )
        visa_status = typer.prompt(
            "Visa status (e.g. 'Requires sponsorship'; optional, leave blank to skip)",
            default="",
            show_default=False,
        )

        existing_count = session.query(IdentityORM).count()
        if existing_count == 0:
            make_default = True
            console.print("[dim](This is the first identity record -- automatically set as default)[/dim]")
        else:
            make_default = typer.confirm("Set this as the default identity (is_default)?", default=False)

        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)

        _handle_default_flag(session, make_default)

        identity = create_row(
            session,
            IdentityORM,
            {
                "name": name,
                "email": email,
                "phone": phone,
                "location": location or None,
                "visa_status": visa_status or None,
                "is_default": make_default,
            },
        )
        identity_id = identity.identity_id

    console.print(f"[green]Done! Identity ID: {identity_id}[/green]")
    if make_default:
        console.print("[green]Set as default identity.[/green]")

# ---------------------------------------------------------------------------
# add-network: social/portfolio links
# ---------------------------------------------------------------------------
# RENDERCV_SUPPORTED_NETWORKS mirrors RenderCV's own SocialNetworkName
# Literal type exactly (see docs.rendercv.com/api_reference/schema/models/
# cv/social_network/) -- any value outside this list will fail RenderCV's
# validation at render time, so the CLI now restricts input to this fixed
# menu instead of accepting free text (which previously allowed values
# like "Behance" that RenderCV does not support).

# KNOWN GAP: username format is still not validated per-platform here.
# Several networks have platform-specific username rules enforced by
# RenderCV at render time (e.g. Mastodon needs "@user@domain",
# StackOverflow needs "user_id/username", IMDB needs "nmXXXXXXX", ORCID
# needs "XXXX-XXXX-XXXX-XXX", WhatsApp needs an international phone
# number). Not duplicated here to avoid drifting out of sync with
# RenderCV's own validation logic -- a render-time failure on username
# format will still surface clearly, just not at add-network time.

RENDERCV_SUPPORTED_NETWORKS: list[str] = [
    "LinkedIn",
    "GitHub",
    "GitLab",
    "IMDB",
    "Instagram",
    "ORCID",
    "Mastodon",
    "StackOverflow",
    "ResearchGate",
    "YouTube",
    "Google Scholar",
    "Telegram",
    "WhatsApp",
    "Leetcode",
    "X",
    "Bluesky",
    "Reddit",
]

def _prompt_network_platform() -> str:
    """Fixed menu of RenderCV-supported platforms -- prevents entries that
    would pass add-network but fail validation later at render time."""
    console.print("\nSelect a platform (must match RenderCV's supported list):")
    for i, platform in enumerate(RENDERCV_SUPPORTED_NETWORKS, 1):
        console.print(f"  [{i}] {platform}")

    choice = typer.prompt("Select a number", type=int)
    return RENDERCV_SUPPORTED_NETWORKS[choice - 1]

def _resolve_target_positions() -> list[str]:
    """Repeatedly prompts for job titles this link applies to, Enter on a
    blank line ends the loop. Leaving the very first prompt blank is
    allowed and returns an empty list -- gencv interprets an empty
    target_position as "applies to any job title" (a universal link,
    e.g. a personal portfolio site or a general GitHub profile), rather
    than requiring the user to enumerate every possible title."""
    positions: list[str] = []
    console.print(
        "\nEnter the job title(s) this link applies to (e.g. Software Engineer), "
        "one at a time. Press Enter on a blank line to finish. "
        "Leave blank immediately (no titles at all) if this link should apply "
        "to ANY job title:"
    )
    while True:
        position = typer.prompt(
            "Job title (leave blank to finish)",
            default="",
            show_default=False,
        )
        if not position.strip():
            break
        positions.append(position.strip())
    return positions

@app.command("add-network")
def add_network() -> None:
    """Interactively add a social/portfolio link (platform/username/target positions).

    Platform must be one of RenderCV's supported networks (fixed menu, not
    free text) -- see RENDERCV_SUPPORTED_NETWORKS above. Username format is
    NOT validated here per-platform; RenderCV will surface a clear error at
    render time if the format doesn't match that platform's requirements
    (e.g. Mastodon, StackOverflow, IMDB, ORCID, WhatsApp all have specific
    username patterns).

    target_position may be an empty list -- gencv's identity_selector.py
    treats an empty target_position as applying to ANY job title, so this
    is a valid and intentional choice, not an incomplete entry.
    """
    with get_session() as session:
        network_name = _prompt_network_platform()
        username = typer.prompt(f"Username for {network_name} (e.g. yahsintsai)")
        target_position = _resolve_target_positions()

        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)

        network = create_row(
            session,
            NetworkORM,
            {
                "network": network_name,
                "username": username,
                "target_position": target_position,
            },
        )
        network_id = network.network_id

    console.print(f"[green]Done! Network ID: {network_id}[/green]")
    if target_position:
        console.print(f"[green]Applies to: {', '.join(target_position)}[/green]")
    else:
        console.print("[green]Applies to: ANY job title[/green]")

# ---------------------------------------------------------------------------
# add-project (v1.9): flat position/degree selectors
# ---------------------------------------------------------------------------
# Joins position+company (work track) or degree+school (education track)
# directly, skipping the separate "pick/create company" then "pick/create
# position" steps whenever existing rows are available. Returns None in
# TWO cases, both of which signal the caller to fall back to the original
# _resolve_company -> _resolve_position (or _resolve_school ->
# _resolve_degree) stepwise flow:
# 1. No rows exist at all yet (menu isn't even shown).
# 2. The user explicitly picks "[Create new]" (option 0), even though
#    existing rows were available -- the escape hatch for genuinely
#    new companies/positions.

def _select_existing_position_flat(session) -> Optional[PositionORM]:
    rows = (
        session.query(PositionORM, CompanyORM)
        .join(CompanyORM, PositionORM.company_id == CompanyORM.company_id)
        .all()
    )
    if not rows:
        return None

    table = Table(title="Select an existing position, or create a new one")
    table.add_column("#")
    table.add_column("Institution")
    table.add_column("Role")
    table.add_row("0", "[Create new]", "")
    for idx, (position, company) in enumerate(rows, start=1):
        table.add_row(str(idx), company.company_name or "", position.title or "")
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if choice == 0:
        return None
    return rows[choice - 1][0]

def _select_existing_degree_flat(session) -> Optional[DegreeORM]:
    rows = (
        session.query(DegreeORM, SchoolORM)
        .join(SchoolORM, DegreeORM.school_id == SchoolORM.school_id)
        .all()
    )
    if not rows:
        return None

    table = Table(title="Select an existing education record, or create a new one")
    table.add_column("#")
    table.add_column("Institution")
    table.add_column("Role")
    table.add_row("0", "[Create new]", "")
    for idx, (degree, school) in enumerate(rows, start=1):
        table.add_row(str(idx), school.school_name or "", degree.degree_name or "")
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if choice == 0:
        return None
    return rows[choice - 1][0]

# ---------------------------------------------------------------------------
# Step 1: upper-level entity (company / school)
# ---------------------------------------------------------------------------

def _resolve_company(session) -> CompanyORM:
    name = typer.prompt("Company name")
    candidates = fuzzy_find_candidates(session, CompanyORM, "company_name", name)
    if candidates:
        chosen = _select_from_fuzzy_candidates(candidates, "company_name")
        if chosen is not None:
            return chosen

    location = typer.prompt("Location", default="", show_default=False)
    return create_row(
        session,
        CompanyORM,
        {"company_name": name, "location": location or None},
    )

def _resolve_school(session) -> SchoolORM:
    name = typer.prompt("School name")
    candidates = fuzzy_find_candidates(session, SchoolORM, "school_name", name)
    if candidates:
        chosen = _select_from_fuzzy_candidates(candidates, "school_name")
        if chosen is not None:
            return chosen

    location = typer.prompt("Location", default="", show_default=False)
    return create_row(
        session,
        SchoolORM,
        {"school_name": name, "location": location or None},
    )

# ---------------------------------------------------------------------------
# Standalone entity commands -- lets upper/mid-level entities be staged
# one at a time instead of only inside the monolithic add-project flow.
# None of these call fill_pending_support_levels() -- governance never
# applies to them.
# ---------------------------------------------------------------------------

@app.command("add-company")
def add_company() -> None:
    """Interactively add a single company record (standalone, no position/project required)."""
    with get_session() as session:
        company = _resolve_company(session)
        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)
        company_id = company.company_id

    console.print(f"[green]Done! Company ID: {company_id}[/green]")

@app.command("add-school")
def add_school() -> None:
    """Interactively add a single school record (standalone, no degree/project required)."""
    with get_session() as session:
        school = _resolve_school(session)
        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)
        school_id = school.school_id

    console.print(f"[green]Done! School ID: {school_id}[/green]")

@app.command("add-position")
def add_position() -> None:
    """Interactively add a single position record under an existing company (standalone, no project required)."""
    with get_session() as session:
        company = _select_existing_company(session)
        position = _resolve_position(session, company)
        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)
        position_id = position.position_id

    console.print(f"[green]Done! Position ID: {position_id}[/green]")


@app.command("add-degree")
def add_degree() -> None:
    """Interactively add a single degree record under an existing school (standalone, no project required)."""
    with get_session() as session:
        school = _select_existing_school(session)
        degree = _resolve_degree(session, school)
        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)
        degree_id = degree.degree_id

    console.print(f"[green]Done! Degree ID: {degree_id}[/green]")

# ---------------------------------------------------------------------------
# Step 2: mid-level entity (position / degree)
# ---------------------------------------------------------------------------

def _draft_responsibility() -> tuple[str, Optional[str]]:
    """Returns (responsibility_text, responsibility_source)."""
    client = get_llm_client()
    has_source = typer.confirm(
        "Do you have a source document to link the responsibility description to (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                source_text = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")

        response = client.generate_text(
            system_prompt=RESPONSIBILITY_SYSTEM_PROMPT,
            user_prompt=build_responsibility_prompt_from_source(source_text),
        )
        final_text = _confirm_or_edit(response.text, "Responsibility summary")
        return final_text, path

    description = typer.prompt("Briefly describe the scope of responsibilities for this position")
    response = client.generate_text(
        system_prompt=RESPONSIBILITY_SYSTEM_PROMPT,
        user_prompt=build_responsibility_prompt_from_description(description),
    )
    final_text = _confirm_or_edit(response.text, "Responsibility summary")
    return final_text, None

def _resolve_position(session, company: CompanyORM) -> PositionORM:
    existing = list_rows(session, PositionORM, company_id=company.company_id)
    if existing:
        chosen = _select_from_list(
            existing,
            [("Title", "title"), ("Department", "department"), ("Start date", "start_date")],
            title=f"Existing positions at {company.company_name}",
            allow_new_label="Add new position",
        )
        if chosen is not None:
            return chosen

    title = typer.prompt("Job title")
    department = typer.prompt("Department", default="", show_default=False)
    start_dt = _prompt_date("Start date (YYYY-MM-DD)")
    end_dt = _prompt_date("End date (YYYY-MM-DD, leave blank for present)", allow_blank=True)
    responsibility, responsibility_source = _draft_responsibility()

    return create_row(
        session,
        PositionORM,
        {
            "company_id": company.company_id,
            "title": title,
            "department": department or None,
            "responsibility": responsibility,
            "responsibility_source": responsibility_source,
            "start_date": start_dt,
            "end_date": end_dt,
        },
    )

def _draft_degree_description() -> tuple[str, Optional[str]]:
    """Returns (description_text, description_source). Mirrors
    _draft_responsibility() exactly -- source-file-or-typed-description
    dual entry point, LLM draft + _confirm_or_edit() gate."""
    client = get_llm_client()
    has_source = typer.confirm(
        "Do you have a source document to base the degree description on (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                source_text = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")

        response = client.generate_text(
            system_prompt=DEGREE_DESCRIPTION_SYSTEM_PROMPT,
            user_prompt=build_degree_description_prompt_from_source(source_text),
        )
        final_text = _confirm_or_edit(response.text, "Degree description")
        return final_text, path

    description = typer.prompt("Briefly describe the focus/scope of this degree program")
    response = client.generate_text(
        system_prompt=DEGREE_DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=build_degree_description_prompt_from_description(description),
    )
    final_text = _confirm_or_edit(response.text, "Degree description")
    return final_text, None

def _resolve_degree_achievements():
    """Repeatedly prompts for degree_honors or degree_wards. Enter on a
    blank line ends the loop. Leaving the very first prompt blank is
    allowed and returns an empty list"""
    honors: list = []
    awards: list = []
    console.print(
        "\nEnter the hornors/awards this degree received one at a time. "
        "Press Enter on a blank line to finish. "
        "Leave blank immediately if none: "
    )
    while True:
        achievement = typer.prompt(
            "Achievement(leave blank to finish)",
            default="",
            show_default=False,
        )
        if not achievement.strip():
            break
        is_honor = typer.prompt(
            "Is this an honor or an award (type Y if it's an honor and N if award)",
            default = "Y",
            show_default=True
        )
        if is_honor:
            honors.append(achievement.strip())
        else:
            awards.append(achievement.strip())
    return honors, awards

def _resolve_degree(session, school: "SchoolORM") -> "DegreeORM":
    """v3: honors/awards prompts and fields removed -- see module docstring."""
    existing = list_rows(session, DegreeORM, school_id=school.school_id)
    if existing:
        chosen = _select_from_list(
            existing,
            [("Degree level", "degree_level"), ("Subject", "degree_name"), ("Start date", "start_date")],
            title=f"Existing education records at {school.school_name}",
            allow_new_label="Add new education record",
        )
        if chosen is not None:
            return chosen

    degree_level = typer.prompt("Degree level (e.g. MA, MS, PhD, Short Program)")
    degree_type = typer.prompt("Degree type (e.g. Major, Minor, Concentration)", default="", show_default=False)
    degree_name = typer.prompt("Subject / area of study (e.g. Computer Science)")
    start_dt = _prompt_date("Start date (YYYY-MM-DD)")
    end_dt = _prompt_date("End date (YYYY-MM-DD, leave blank)", allow_blank=True)

    degree_description, degree_description_source = _draft_degree_description()

    degree_honors, degree_awards = _resolve_degree_achievements()

    return create_row(
        session,
        DegreeORM,
        {
            "school_id": school.school_id,
            "degree_level": degree_level,
            "degree_type": degree_type or None,
            "degree_name": degree_name,
            "degree_description": degree_description,
            "degree_description_source": degree_description_source,
            "degree_honors": degree_honors,
            "degree_awards": degree_awards,
            "start_date": start_dt,
            "end_date": end_dt,
        },
    )

def _select_existing_degree_flat(session):
    """No functional change from earlier versions -- add-project unaffected.
    See module docstring for the display-semantics note on degree_name."""
    rows = (
        session.query(DegreeORM, SchoolORM)
        .join(SchoolORM, DegreeORM.school_id == SchoolORM.school_id)
        .all()
    )
    if not rows:
        return None

    table = Table(title="Select an existing education record, or create a new one")
    table.add_column("#")
    table.add_column("Institution")
    table.add_column("Role")
    table.add_row("0", "[Create new]", "")
    for idx, (degree, school) in enumerate(rows, start=1):
        table.add_row(str(idx), school.school_name or "", degree.degree_name or "")
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if choice == 0:
        return None
    return rows[choice - 1][0]

# ---------------------------------------------------------------------------
# Step 3: project
# ---------------------------------------------------------------------------

def _resolve_project(
    session,
    position_id: Optional[UUID],
    degree_id: Optional[UUID],
) -> ProjectORM:
    project_name = typer.prompt("Project name")
    start_dt = _prompt_date("Project start date (YYYY-MM-DD)")
    end_dt = _prompt_date("Project end date (YYYY-MM-DD, leave blank)", allow_blank=True)

    client = get_llm_client()
    has_source = typer.confirm(
        "Do you have a source document to link the project description to (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                source_text = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")

        response = client.generate_text(
            system_prompt=PROJECT_CONTEXT_SYSTEM_PROMPT,
            user_prompt=build_project_context_prompt_from_source(source_text),
        )

        project_context = _confirm_or_edit(response.text, "Project background narrative")

        return create_row(
                session,
                ProjectORM,
                {
                    "position_id": position_id,
                    "degree_id": degree_id,
                    "project_name": project_name,
                    "start_date": start_dt,
                    "end_date": end_dt,
                    "project_context": project_context,
                    "project_context_source": path,
                },
            )
    
    description = typer.prompt("Briefly describe the scope of responsibilities for this position")

    response = client.generate_text(
        system_prompt=PROJECT_CONTEXT_SYSTEM_PROMPT,
        user_prompt=build_project_context_prompt_from_description(description),
    )

    project_context = _confirm_or_edit(response.text, "Project background narrative")

    return create_row(
        session,
        ProjectORM,
        {
            "position_id": position_id,
            "degree_id": degree_id,
            "project_name": project_name,
            "start_date": start_dt,
            "end_date": end_dt,
            "project_context": project_context,
        },
    )

# ---------------------------------------------------------------------------
# Step 4: final confirmation
# ---------------------------------------------------------------------------

def _confirm_final_write() -> bool:
    return typer.confirm("\nConfirm write to database?", default=True)

# ---------------------------------------------------------------------------
# add-project command (v1.9: flat position/degree selector first)
# ---------------------------------------------------------------------------

@app.command("add-project")
def add_project() -> None:
    """Interactively add a new project record under a work or education experience."""
    track = _select_from_menu(
        TRACK_OPTIONS,
        title="Is this a work experience or education experience?",
        allow_blank=False,
    )

    project_id: Optional[UUID] = None

    with get_session() as session:
        if track == "work":
            position = _select_existing_position_flat(session)
            if position is None:
                company = _resolve_company(session)
                position = _resolve_position(session, company)
            project = _resolve_project(session, position_id=position.position_id, degree_id=None)
        else:
            degree = _select_existing_degree_flat(session)
            if degree is None:
                school = _resolve_school(session)
                degree = _resolve_degree(session, school)
            project = _resolve_project(session, position_id=None, degree_id=degree.degree_id)

        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)

        project_id = project.project_id
        # transaction commits automatically on successful exit of get_session().
        # NOTE: add-project never calls fill_pending_support_levels() --
        # governance rows never exist for anything but claims.

    console.print(f"[green]Done! New project ID: {project_id}[/green]")
    return project_id

# ---------------------------------------------------------------------------
# add-claim: interactive project selector (flat, join-based, single menu --
# Institution / Role / Project columns only)
# ---------------------------------------------------------------------------

def _select_project_interactively(session) -> ProjectORM:
    """
    Shows every existing project in ONE flat numbered menu, regardless of
    whether it sits under the work track (company -> position -> project)
    or the education track (school -> degree -> project). Replaces the
    old multi-step drill-down (pick track, then company/school, then
    position/degree, then project) with a single selection step.

    Since core/db.py's ORM models only define Column(ForeignKey(...)) and
    no relationship(), cross-table lookups can't use attribute-chain
    access (e.g. project.position.company) -- this runs two explicit JOIN
    queries instead: one for the work track (project+position+company) and
    one for the education track (project+degree+school). Results from
    both are merged into a single list before being shown.

    Each row displays exactly three columns:
    - Institution: company_name (work track) or school_name (education track)
    - Role: position.title (work track) or degree.degree_name (education track)
    - Project: project_name
    """
    work_rows = (
        session.query(ProjectORM, PositionORM, CompanyORM)
        .join(PositionORM, ProjectORM.position_id == PositionORM.position_id)
        .join(CompanyORM, PositionORM.company_id == CompanyORM.company_id)
        .all()
    )
    education_rows = (
        session.query(ProjectORM, DegreeORM, SchoolORM)
        .join(DegreeORM, ProjectORM.degree_id == DegreeORM.degree_id)
        .join(SchoolORM, DegreeORM.school_id == SchoolORM.school_id)
        .all()
    )

    combined: list[tuple[ProjectORM, str, str]] = []
    for project, position, company in work_rows:
        combined.append((project, company.company_name, position.title))
    for project, degree, school in education_rows:
        combined.append((project, school.school_name, degree.degree_name))

    if not combined:
        console.print("[red]No project records found -- please run add-project first[/red]")
        raise typer.Exit(code=1)

    combined.sort(key=lambda row: (row[1] or "", row[0].project_name or ""))

    table = Table(title="Select a project")
    table.add_column("#")
    table.add_column("Institution")
    table.add_column("Role")
    table.add_column("Project")
    for idx, (project, institution, role) in enumerate(combined, start=1):
        table.add_row(
            str(idx),
            institution or "",
            role or "",
            project.project_name or "",
        )
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    return combined[choice - 1][0]

# ---------------------------------------------------------------------------
# add-claim: Step 1 -- problem
# ---------------------------------------------------------------------------

def _resolve_problem(session, project: ProjectORM) -> ProblemORM:
    description = typer.prompt("Describe the problem encountered in this project (no need to describe the solution)")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=PROBLEM_SYSTEM_PROMPT,
        user_prompt=build_problem_prompt(description),
    )
    problem_description = _confirm_or_edit(response.text, "Problem statement")

    stakeholder_impact = typer.prompt(
        "Which stakeholders were affected by this problem? (optional)", default="", show_default=False
    )

    return create_row(
        session,
        ProblemORM,
        {
            "project_id": project.project_id,
            "description": problem_description,
            "stakeholder_impact": stakeholder_impact or None,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 2 -- action
# ---------------------------------------------------------------------------

def _resolve_action(session, problem: ProblemORM) -> ActionORM:
    description = typer.prompt("What did you do about this problem? (describe only the action itself)")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=ACTION_SYSTEM_PROMPT,
        user_prompt=build_action_prompt(description),
    )
    action_description = _confirm_or_edit(response.text, "Action statement")

    role_scope = _select_from_menu(
        ROLE_SCOPE_OPTIONS,
        title="What was your role in this action?",
        allow_blank=False,
    )

    return create_row(
        session,
        ActionORM,
        {
            "problem_id": problem.problem_id,
            "description": action_description,
            "role_scope": role_scope,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 3 -- claim (description)
# ---------------------------------------------------------------------------

def _resolve_claim(session, action: ActionORM) -> ClaimORM:
    raw_input_text = typer.prompt("What concrete result did this action produce? (describe only the result, don't repeat the action)")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=RESULT_CLAIM_SYSTEM_PROMPT,
        user_prompt=build_result_claim_prompt(raw_input_text),
    )
    description = _confirm_or_edit(response.text, "Result statement")

    return create_row(
        session,
        ClaimORM,
        {
            "action_id": action.action_id,
            "description": description,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 3.5 -- metrics
# ---------------------------------------------------------------------------

def _review_metric_drafts(drafts: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    console.print("\n[bold]Metric drafts (extracted by LLM from the result statement)[/bold]")
    if drafts:
        table = Table()
        table.add_column("#")
        table.add_column("Desc")
        table.add_column("Value")
        for idx, (m_desc, m_value) in enumerate(drafts, start=1):
            table.add_row(str(idx), m_desc, m_value)
        console.print(table)
    else:
        console.print("[dim](LLM did not detect any quantifiable metrics)[/dim]")

    finalized: list[tuple[str, str, str, str]] = []
    for m_desc, m_value in drafts:
        keep = typer.confirm(f'Keep the metric "{m_desc}: {m_value}"?', default=True)
        if not keep:
            continue
        edited_desc = typer.prompt("Description", default=m_desc)
        edited_value = typer.prompt("Value", default=m_value)
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f'Confidence level for "{edited_desc}"?',
            allow_blank=False,
        )
        finalized.append((edited_desc, edited_value, confidence))

    while typer.confirm("Manually add another metric? (add anything the LLM missed here)", default=False):
        m_desc = typer.prompt("Description (what this number represents)")
        m_value = typer.prompt("Value")
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f'Confidence level for "{m_desc}"?',
            allow_blank=False,
        )
        finalized.append((m_desc, m_value, confidence))

    return finalized

def _resolve_metrics(session, claim: ClaimORM) -> list[MetricORM]:
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=METRIC_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_metric_extraction_prompt(claim.claim_description),
    )
    drafts = parse_metric_response(response.text)
    finalized = _review_metric_drafts(drafts)

    metric_orms: list[MetricORM] = []
    for m_desc, m_value, confidence in finalized:
        metric_orms.append(
            create_row(
                session,
                MetricORM,
                {
                    "claim_id": claim.claim_id,
                    "metric_description": m_desc,
                    "metric_value": m_value,
                    "metric_confidence": confidence,
                },
            )
        )
    return metric_orms

# ---------------------------------------------------------------------------
# add-claim: Step 4 -- evidence
# ---------------------------------------------------------------------------

def _select_evidence_type() -> tuple[str, str]:
    choices = list(EVIDENCE_TYPE_TO_LEVEL.keys())
    table = Table(title="Select evidence type (level will be assigned automatically)")
    table.add_column("#")
    table.add_column("Evidence type")
    table.add_column("Corresponding level")
    for idx, choice in enumerate(choices, start=1):
        table.add_row(str(idx), choice, EVIDENCE_TYPE_TO_LEVEL[choice])
    console.print(table)

    idx = typer.prompt("Select a number", type=int)
    evidence_type = choices[idx - 1]
    evidence_level = EVIDENCE_TYPE_TO_LEVEL[evidence_type]
    return evidence_type, evidence_level

def _resolve_one_evidence(session, claim: ClaimORM) -> EvidenceORM:
    evidence_type, evidence_level = _select_evidence_type()
    evidence_location = typer.prompt(
        "Evidence location/link? (e.g. file path, URL, optional)", default="", show_default=False
    )
    console.print(f"[dim]Corresponding evidence level: {evidence_level} (auto-derived from evidence type)[/dim]")

    return create_row(
        session,
        EvidenceORM,
        {
            "claim_id": claim.claim_id,
            "evidence_type": evidence_type,
            "evidence_location": evidence_location or None,
            "evidence_level": evidence_level,
        },
    )

def _resolve_evidences(session, claim: ClaimORM) -> list[EvidenceORM]:
    evidences: list[EvidenceORM] = [_resolve_one_evidence(session, claim)]
    while typer.confirm("Any other evidence to add for this claim?", default=False):
        evidences.append(_resolve_one_evidence(session, claim))
    return evidences

# ---------------------------------------------------------------------------
# add-claim: Step 5 -- tags
# ---------------------------------------------------------------------------

def _confirm_or_edit_tags(proposed: list[tuple[str, str]]) -> list[tuple[str, str]]:
    console.print("\n[bold]Tag drafts (generated by LLM from this record's full content)[/bold]")
    table = Table()
    table.add_column("#")
    table.add_column("Tag")
    table.add_column("Category")
    for idx, (name, category) in enumerate(proposed, start=1):
        table.add_row(str(idx), name, category)
    console.print(table)

    edited = typer.prompt(
        "Press Enter to accept the tags above, or type a new tag list (format: tag1|category1, tag2|category2, ...)",
        default="",
        show_default=False,
    )
    if not edited.strip():
        return proposed

    valid_categories = {"skill", "domain", "job_function"}
    result: list[tuple[str, str]] = []
    for chunk in edited.split(","):
        chunk = chunk.strip()
        if not chunk or "|" not in chunk:
            continue
        name_part, _, category_part = chunk.partition("|")
        name = name_part.strip()
        category = category_part.strip().lower()
        if name and category in valid_categories:
            result.append((name, category))
    return result if result else proposed

def _build_static_context(session, project: ProjectORM) -> str:
    if project.position_id is not None:
        position = session.get(PositionORM, project.position_id)
        company = session.get(CompanyORM, position.company_id) if position else None
        parts = []
        if company is not None:
            parts.append(f"Company: {company.company_name}" + (f" ({company.location})" if company.location else ""))
        if position is not None:
            parts.append(f"Title: {position.title}" + (f", Department: {position.department}" if position.department else ""))
        return "; ".join(parts)

    if project.degree_id is not None:
        degree = session.get(DegreeORM, project.degree_id)
        school = session.get(SchoolORM, degree.school_id) if degree else None
        parts = []
        if school is not None:
            parts.append(f"School: {school.school_name}" + (f" ({school.location})" if school.location else ""))
        if degree is not None:
            parts.append(f"Degree/Program: {degree.degree_name}") # + (f", Subject: {degree.subject}" if degree.subject else "")
        return "; ".join(parts)

    return ""

def _resolve_tags(
    session,
    claim: ClaimORM,
    problem: ProblemORM,
    action: ActionORM,
    evidences: list[EvidenceORM],
    project: ProjectORM,
) -> list[TagORM]:
    static_context = _build_static_context(session, project)
    evidence_types_joined = ", ".join(sorted({e.evidence_type for e in evidences}))

    client = get_llm_client()
    response = client.generate_text(
        system_prompt=TAG_SYSTEM_PROMPT,
        user_prompt=build_tag_prompt(
            problem_description=problem.problem_description,
            action_description=action.action_description,
            description=claim.claim_description,
            evidence_type=evidence_types_joined,
            static_context=static_context,
        ),
    )
    proposed = parse_tag_response(response.text)
    final_tags = _confirm_or_edit_tags(proposed)

    tag_orms: list[TagORM] = []
    for tag_name, tag_category in final_tags:
        tag = select_or_create(
            session,
            TagORM,
            match_filters={"tag_name": tag_name},
            create_values={"tag_name": tag_name, "tag_category": tag_category},
        )
        tag_orms.append(tag)
        create_row(
            session,
            ClaimTagORM,
            {"claim_id": claim.claim_id, "tag_id": tag.tag_id},
        )
    return tag_orms

# ---------------------------------------------------------------------------
# add-claim: Step 6 -- governance
# ---------------------------------------------------------------------------

def _resolve_governance(session, claim: ClaimORM) -> GovernanceORM:
    use_policy = typer.prompt(
        "Usage restriction, if any (e.g. senior roles only, specific industry)? (optional, blank = follow global rules)",
        default="",
        show_default=False,
    )
    notes_for_llm = typer.prompt(
        "Any operating notes for the generation model? (optional, blank = follow global rules)",
        default="",
        show_default=False,
    )
    allowed_for_generation = typer.confirm(
        "Allow this claim into the CV-generation candidate pool?", default=True
    )

    return create_row(
        session,
        GovernanceORM,
        {
            "claim_id": claim.claim_id,
            "support_level": None,  # filled in by fill_pending_support_levels(), called after commit
            "use_policy": use_policy or None,
            "notes_for_llm": notes_for_llm or None,
            "allowed_for_generation": allowed_for_generation,
        },
    )

# ---------------------------------------------------------------------------
# add-claim command
# ---------------------------------------------------------------------------

@app.command("add-claim")
def add_claim(
    project_id: Optional[UUID] = typer.Option(
        None,
        "--project-id",
        help="UUID of the project to add a claim to. If omitted, an interactive selector is shown.",
    ),
) -> None:
    """Interactively add a problem -> action -> claim -> metric(s) -> evidence(s) -> tag(s) -> governance record."""
    claim_id: Optional[UUID] = None
    evidence_ids: list[UUID] = []
    metric_summaries: list[str] = []
    tag_names: list[str] = []

    with get_session() as session:
        if project_id is not None:
            project = session.get(ProjectORM, project_id)
            if project is None:
                console.print(f"[red]No project found for project_id: {project_id}[/red]")
                raise typer.Exit(code=1)
        else:
            project = _select_project_interactively(session)

        console.print(f"[bold]Selected project: {project.project_name}[/bold]\n")

        problem = _resolve_problem(session, project)
        action = _resolve_action(session, problem)
        claim = _resolve_claim(session, action)
        metrics = _resolve_metrics(session, claim)
        evidences = _resolve_evidences(session, claim)
        tags = _resolve_tags(session, claim, problem, action, evidences, project)
        _resolve_governance(session, claim)

        if not _confirm_final_write():
            console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
            raise typer.Exit(code=0)

        claim_id = claim.claim_id
        # evidence_ids = [evidence.evidence_id for evidence in evidences]
        # metric_summaries = [f"{m.metric_label}: {m.metric_value}" for m in metrics]
        # tag_names = [tag.tag_name for tag in tags]

    fill_pending_support_levels()

    console.print(f"[green]Done! New claim ID: {claim_id}[/green]")
    # console.print(f"[green]Evidence IDs ({len(evidence_ids)} total): {', '.join(str(e) for e in evidence_ids)}[/green]")
    # if metric_summaries:
    #     console.print(f"[green]Metrics: {'; '.join(metric_summaries)}[/green]")
    # if tag_names:
    #     console.print(f"[green]Tags: {', '.join(tag_names)}[/green]")
    # console.print("[green]Governance support_level: computed and filled in.[/green]")

# ---------------------------------------------------------------------------
# add-project-claim: from project to claims to all corresponding columns
# ---------------------------------------------------------------------------
def _resolve_llm_response(llm_response: LLMResponse) -> list:
    import re, json

    raw_text = llm_response.text.strip()
    match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw_text, re.DOTALL)
    json_str = match.group(1) if match else raw_text

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLMResponse unable to parse: {e}\nerror message: {llm_response.finish_reason}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected data type: {type(parsed)}")

    return parsed


def _resolve_claims(project: ProjectORM) -> list:
    """
    """
    source: str = project.project_context_source
    has_source: bool = False
    path: str

    if source is not None:
         has_source = typer.confirm(
                f"There's a source is your project. Would you like to derive claims from {source}?",
                default=True,
            )     
         path = source
    else:                 
        has_source = typer.confirm(
                "Do you have a source document to base the degree description on (file path or URL)?",
                default=True,
            )
        path = typer.prompt("Enter a new file path (.txt / .md / .pdf) or URL")
        
    if has_source:
        while True:
            # path = typer.prompt("Enter a new file path (.txt / .md / .pdf) or URL")
            try:
                source_text = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")

    client = get_llm_client()
    response = client.generate_text(
        system_prompt=CLAIMS_SYSTEM_PROMPT,
        user_prompt=build_claims_prompt_from_source(source_text),
        max_tokens=2000 # tyahsin: this has to do with the constraint of number of claims generated
    )
    raw_claims = _resolve_llm_response(response)

    return raw_claims


@app.command("add-claims")
def add_claims(
        project_id: Optional[UUID] = typer.Option(
        None,
        "--project-id",
        help="UUID of the project to add a claim to. If omitted, an interactive selector is shown.",
    ),
):
    """Bulk add claims to database through source files or URLs."""
    claim_id: Optional[UUID] = None
    evidence_ids: list[UUID] = []
    metric_summaries: list[str] = []
    tag_names: list[str] = []

    with get_session() as session:
        if project_id is not None:
            project = session.get(ProjectORM, project_id)
            if project is None:
                console.print(f"[red]No project found for project_id: {project_id}[/red]")
                raise typer.Exit(code=1)
        else:
            project = _select_project_interactively(session)

    raw_claims = _resolve_claims(project) 
    for i, raw_claim in enumerate(raw_claims):
        confirmed = {}
        for key, val in raw_claim.items(): 
            confirmed_val = _confirm_or_edit(val, key)
            confirmed[key] = confirmed_val

        with get_session() as session:
            problem = create_row(
                session,
                ProblemORM,
                {
                    "project_id": project.project_id,
                    "problem_description": confirmed["problem_description"],
                    "stakeholder_impact": confirmed["stakeholder_impact"],
                }
            )
            action = create_row(
                session,
                ActionORM,
                {
                    "problem_id": problem.problem_id,
                    "action_description": confirmed["action_description"],
                    "role_scope": confirmed["role_scope"],
                }
            )
            claim = create_row(
                session,
                ClaimORM,
                {
                    "action_id": action.action_id,
                    "claim_description": confirmed["claim_description"],
                }
            )
            if not _confirm_final_write():
                console.print(f"[yellow] Claim {(i+1)} dumped -- no changes were written.[/yellow]")
                continue

        with get_session() as session:
            metrics = _resolve_metrics(session, claim)
            evidences = _resolve_evidences(session, claim)
            tags = _resolve_tags(session, claim, problem, action, evidences, project)
            _resolve_governance(session, claim)

            if not _confirm_final_write():
                console.print("[yellow]Cancelled -- no changes were written.[/yellow]")
                continue

            # claim_id = claim.claim_id
            # evidence_ids = [evidence.evidence_id for evidence in evidences]
            # metric_summaries = [f"{m.metric_label}: {m.metric_value}" for m in metrics]
            # tag_names = [tag.tag_name for tag in tags]

        fill_pending_support_levels()
        console.print(f"[green] {project.project_name} has {(i+1)} claim(s) written into database [/green]")
        # console.print(f"[green]Done! New claim ID: {claim_id}[/green]")
        # console.print(f"[green]Evidence IDs ({len(evidence_ids)} total): {', '.join(str(e) for e in evidence_ids)}[/green]")
        # if metric_summaries:
        #     console.print(f"[green]Metrics: {'; '.join(metric_summaries)}[/green]")
        # if tag_names:
        #     console.print(f"[green]Tags: {', '.join(tag_names)}[/green]")
        # console.print("[green]Governance support_level: computed and filled in.[/green]")

@app.command("add-project-claims")
def add_project_claims() -> None:
    """Bulk add a project and its claims to database through source files or URLs."""
    project_id = add_project()
    add_claims(project_id)

if __name__ == "__main__":
    app()
