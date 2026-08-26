"""
onboard/cli.py
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from core.db import * # *ORM, *OPTIONS, functions
from core.source_reader import FileReadError, UnsupportedFileTypeError, read_source
from core.llm_client import get_llm_client, LLMResponse
from core.llm_reviewer import confirm_field, confirm_fields_multiple
from onboard.prompts import *

app = typer.Typer(help="Add experience entries to database.")
console = Console()

# ---------------------------------------------------------------------------
# Shared helpers
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

def _confirm_or_edit(draft, label: str, system_prompt: str) -> str:
    console.print(f"\n[bold]{label} (LLM draft)[/bold]")
    return confirm_field(draft.text, system_prompt)

# def _confirm_final_write() -> bool:
#     return typer.confirm("\nConfirm write to database?", default=True)

def _create_table(items: list, columns: list, title: str | None = None, add_0: list | None = None) -> None:

    columns = ["#"] + columns

    table = Table(title=title)
    for column in columns: # header
        table.add_column(column)

    if items: # if there's element inside items list
        if isinstance(items[0], (tuple, list)):
            if add_0:
                table.add_row(0, *add_0)
            for idx, item in enumerate(items, start=1): # values
                table.add_row(str(idx), *item)
        else:
            if add_0:
                table.add_row(0, add_0)
            for idx, item in enumerate(items, start=1): # values
                table.add_row(str(idx), item)

    console.print(table)

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

    table.add_row("0", "[Create new]", "")
    for idx, (row, score) in enumerate(candidates, start=1):
        table.add_row(str(idx), getattr(row, label_field), f"{score:.0f}%")
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
    if allow_new_label is not None:
        table.add_row("0", f"[{allow_new_label}]", *["" for _ in columns[1:]])
    for idx, row in enumerate(rows, start=1):
        values = [str(getattr(row, attr) or "") for _, attr in columns]
        table.add_row(str(idx), *values)
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
    columns = ["Option"]
    if allow_blank:
        _create_table(options, columns, title, add_0 = ["[Skip]"])
    else:
        _create_table(options, columns, title)

    choice = typer.prompt("Select a number", type=int)
    if allow_blank and choice == 0:
        return None
    return options[choice - 1]

def _select_existing_track():
    return _select_from_menu(
        TRACK_OPTIONS,
        title="Is this a work experience or education experience?",
        allow_blank=False,
    )

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

def _select_existing_position(session) -> Optional[PositionORM]:
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

def _select_existing_degree(session) -> Optional[DegreeORM]:
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
        company_id = company.company_id

    console.print(f"[green]Done! Company ID: {company_id}[/green]")

@app.command("add-school")
def add_school() -> None:
    """Interactively add a single school record (standalone, no degree/project required)."""
    with get_session() as session:
        school = _resolve_school(session)
        school_id = school.school_id

    console.print(f"[green]Done! School ID: {school_id}[/green]")

@app.command("add-position")
def add_position() -> None:
    """Interactively add a single position record under an existing company (standalone, no project required)."""
    with get_session() as session:
        company = _select_existing_company(session)
        position = _resolve_position(session, company)
        position_id = position.position_id

    console.print(f"[green]Done! Position ID: {position_id}[/green]")


@app.command("add-degree")
def add_degree() -> None:
    """Interactively add a single degree record under an existing school (standalone, no project required)."""
    with get_session() as session:
        school = _select_existing_school(session)
        degree = _resolve_degree(session, school)
        degree_id = degree.degree_id

    console.print(f"[green]Done! Degree ID: {degree_id}[/green]")

# ---------------------------------------------------------------------------
# Step 2: mid-level entity (position / degree)
# ---------------------------------------------------------------------------

def _draft_responsibility() -> tuple[str, Optional[str]]:
    """Returns (responsibility_text, responsibility_source)."""
    path: str = None
    description: str = None
    has_source = typer.confirm(
        "Do you have a source document to link the responsibility description to (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                description = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")
    else:
        description = typer.prompt("Briefly describe the scope of responsibilities for this position")

    client = get_llm_client()
    response = client.generate_text(
        system_prompt=RESPONSIBILITY_SYSTEM_PROMPT,
        user_prompt=build_responsibility_prompt_from_source(description),
    )
    final_text = _confirm_or_edit(response, "Responsibility summary", RESPONSIBILITY_REGENERATE_SYSTEM_PROMPT)
    return final_text, path



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
    path: str = None
    description: str = None
    has_source = typer.confirm(
        "Do you have a source document to base the degree description on (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                description = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")
    else:
        description = typer.prompt("Briefly describe the focus/scope of this degree program")

        client = get_llm_client()
        response = client.generate_text(
            system_prompt=DEGREE_DESCRIPTION_SYSTEM_PROMPT,
            user_prompt=build_degree_description_prompt_from_source(description),
        )
        final_text = _confirm_or_edit(response, "Degree description", DEGREE_DESCRIPTION_REGENERATE_SYSTEM_PROMPT)
        return final_text, path
    

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

def _select_existing_degree(session):
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

    path: str = None
    description: str = None
    has_source = typer.confirm(
        "Do you have a source document to link the project description to (file path or URL)?",
        default=False,
    )
    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
            try:
                description = read_source(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")
    else:
        description = typer.prompt("Briefly describe the scope of responsibilities for this position")

        client = get_llm_client()
        response = client.generate_text(
            system_prompt=PROJECT_CONTEXT_SYSTEM_PROMPT,
            user_prompt=build_project_context_prompt_from_source(description),
        )

        project_context = _confirm_or_edit(response, "Project background narrative", PROBLEM_REGENERATE_SYSTEM_PROMPT)

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

# ---------------------------------------------------------------------------
# add-project command (v1.9: flat position/degree selector first)
# ---------------------------------------------------------------------------

@app.command("add-project")
def add_project() -> None:
    """Interactively add a new project record under a work or education experience."""
    project_id: Optional[UUID] = None
    track = _select_existing_track()
    with get_session() as session:
        if track == "work":
            position = _select_existing_position(session)
            if position is None:
                company = _select_existing_company(session)
                if company is None:
                    company = _resolve_company(session)
                position = _resolve_position(session, company)
            project = _resolve_project(session, position_id=position.position_id, degree_id=None)
        else:
            degree = _select_existing_degree(session)
            if degree is None:
                school = _select_existing_school(session)
                if school is None:
                    school = _resolve_school(session)
                degree = _resolve_degree(session, school)
            project = _resolve_project(session, position_id=None, degree_id=degree.degree_id)

        project_id = project.project_id

    console.print(f"[green]Done! New project ID: {project_id}[/green]")
    return project_id

# ---------------------------------------------------------------------------
# add-claim: interactive project selector (flat, join-based, single menu --
# Institution / Role / Project columns only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# add-claim: Step 1 -- problem
# ---------------------------------------------------------------------------

def _resolve_problem(session, project: ProjectORM, desc: str | None = None, impact: str | None = None) -> ProblemORM:
    if desc is None and impact is None:
        description = typer.prompt("Describe the problem encountered in this project (no need to describe the solution)")
        client = get_llm_client()
        response = client.generate_text(
            system_prompt=PROBLEM_SYSTEM_PROMPT,
            user_prompt=build_problem_prompt(description),
        )
        problem_description = _confirm_or_edit(response, "Problem statement", PROBLEM_REGENERATE_SYSTEM_PROMPT)

        stakeholder_impact = typer.prompt(
            "Which stakeholders were affected by this problem? (optional)", default="", show_default=False
        )
    else:
        problem_description = desc
        stakeholder_impact = impact

    return create_row(
        session,
        ProblemORM,
        {
            "project_id": project.project_id,
            "problem_description": problem_description,
            "stakeholder_impact": stakeholder_impact or None,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 2 -- action
# ---------------------------------------------------------------------------

def _resolve_action(session, problem: ProblemORM, desc: str | None = None, scope: str | None = None) -> ActionORM:
    if desc is None and scope is None:
        description = typer.prompt("What did you do about this problem? (describe only the action itself)")
        client = get_llm_client()
        response = client.generate_text(
            system_prompt=ACTION_SYSTEM_PROMPT,
            user_prompt=build_action_prompt(description),
        )
        action_description = _confirm_or_edit(response, "Action statement", ACTION_REGENERATE_SYSTEM_PROMPT)

        role_scope = _select_from_menu(
            ROLE_SCOPE_OPTIONS,
            title="What was your role in this action?",
            allow_blank=False,
        )
    else:
        description = desc
        role_scope = scope

    return create_row(
        session,
        ActionORM,
        {
            "problem_id": problem.problem_id,
            "action_description": action_description,
            "role_scope": role_scope,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 3 -- claim (description)
# ---------------------------------------------------------------------------

def _resolve_claim(session, action: ActionORM, desc: str | None = None) -> ClaimORM:
    if desc is None:
        raw_input_text = typer.prompt("What concrete result did this action produce? (describe only the result, don't repeat the action)")
        client = get_llm_client()
        response = client.generate_text(
            system_prompt=CLAIM_SYSTEM_PROMPT,
            user_prompt=build_result_claim_prompt(raw_input_text),
        )
        description = _confirm_or_edit(response, "Claim statement", CLAIM_REGENERATE_SYSTEM_PROMPT)
    else:
        description = desc

    return create_row(
        session,
        ClaimORM,
        {
            "action_id": action.action_id,
            "claim_description": description,
        },
    )

# ---------------------------------------------------------------------------
# add-claim: Step 3.5 -- metrics
# ---------------------------------------------------------------------------

def _review_metric_drafts(drafts: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    console.print("\n[bold]Metric (LLM draft)[/bold]")
    columns = ["Description", "Value"]
    if drafts:
        _create_table(drafts, columns)
    else:
        console.print("[dim](LLM did not detect any quantifiable metrics)[/dim]")

    finalized: list[tuple[str, str, str, str]] = []
    edited = confirm_fields_multiple(drafts, columns, ["You are rewriter for metric name. Rewrite the name.", "You are rewriter for metric value. Make it to romaric numbers."])
    for edited_desc, edited_value in edited:
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f'Confidence level for "{edited_desc}"?',
            allow_blank=False,
        )
        finalized.append((edited_desc, edited_value, confidence))

    while typer.confirm("Manually add another metric? (add anything the LLM missed here)", default=False):
        m_desc = typer.prompt("Description (what this number represents)")
        m_value = typer.prompt("Value (what is the difference in numbers/quantifiable statements)")
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
EVIDENCE_TYPE_TO_LEVEL: list[tuple[str, str]] = [
    ("Formal documentation (contract / official report / system screenshot)", "L1"),
    ("Third-party verification (manager review / client feedback / formal evaluation)", "L1"),
    ("Data record (analytics report / system log / dashboard)", "L2"),
    ("Internal team documentation (meeting notes / email / Slack conversation)", "L2"),
    ("Verbal account / recollection (no written record, memory only)", "L3"),
    ("Inference / estimate (reasonable inference without direct evidence)", "L4"),
]

def _select_evidence_type() -> tuple[str, str]:
    columns = ["Evidence type", "Corresponding level"]
    _create_table(EVIDENCE_TYPE_TO_LEVEL, columns, title="Select evidence type (level will be assigned automatically)")

    idx = typer.prompt("Select a number", type=int)
    evidence_type, evidence_level = EVIDENCE_TYPE_TO_LEVEL[idx - 1]
    return evidence_type, evidence_level

def _resolve_one_evidence(session, claim: ClaimORM, project: ProjectORM | None = None) -> EvidenceORM: 

    if project and project.project_context_source:
        if typer.prompt(
            f"Would you like to use {project.project_context_source} as evidence source?", 
            default="N"        
        ):
            evidence_location = project.project_context_source
        else:
            evidence_location = typer.prompt("Any other evidence location/link? (e.g. file path, URL) (optional)")
    else:
        evidence_location = typer.prompt("Evidence location/link? (e.g. file path, URL)") # not optional here
    evidence_type, evidence_level = _select_evidence_type()
    console.print(f"[dim]Corresponding evidence level: {evidence_level} (auto-derived from evidence type)[/dim]")

    return create_row(
        session,
        EvidenceORM,
        {
            "claim_id": claim.claim_id,
            "evidence_type": evidence_type,
            "evidence_location": evidence_location, # or None,
            "evidence_level": evidence_level,
        },
    )

def _resolve_evidences(session, claim: ClaimORM, project: ProjectORM | None = None) -> list[EvidenceORM]:
    evidences: list[EvidenceORM] = []
    while typer.confirm("Any evidences?", default="N"):
        evidences.append(_resolve_one_evidence(session, claim, project))
    return evidences

# ---------------------------------------------------------------------------
# add-claim: Step 5 -- tags
# ---------------------------------------------------------------------------

def _confirm_or_edit_tags(proposed: list[tuple[str, str]]) -> list[tuple[str, str]]:
    console.print("\n[bold]Tag drafts (generated by LLM from this record's full content)[/bold]")
    columns = ["Tag", "Category"]
    _create_table(proposed, columns)

    edited = confirm_fields_multiple(proposed, columns, [
        "You aew a tag naming helper. Please modify the input to the terminology.", 
        "You are a tag category classifier. Please identify the corresponding name into either 'tool', 'capability', or 'context'"
    ])

    result: list[tuple[str, str]] = []
    for name, category in edited:
        if name and category in TAG_CATEOGRY:
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

def _select_existing_project(session, orm) -> Optional[ProjectORM]:
    if isinstance(orm, PositionORM):
        rows = (
            session.query(ProjectORM)
            .join(PositionORM, ProjectORM.position_id == PositionORM.position_id)
            .join(CompanyORM, PositionORM.company_id == CompanyORM.company_id)
            .filter(PositionORM.position_id == orm.position_id)
        )
    elif isinstance(orm, DegreeORM):
        rows = (
            session.query(ProjectORM)
            .join(DegreeORM, ProjectORM.degree_id == DegreeORM.degree_id)
            .join(SchoolORM, DegreeORM.school_id == SchoolORM.school_id)
            .filter(DegreeORM.degree_id == orm.degree_id)
        )
    if not rows:
        return None
    
    table = Table(title="Select an existing project") #, or create a new one")
    table.add_column("#")
    table.add_column("Project")
    table.add_column("Project Context")
    table.add_row("0", "[Create new]", "")
    for idx, row in enumerate(rows, start=1):
        table.add_row(str(idx), row.project_name or "", row.project_context or "")
    console.print(table)

    choice = typer.prompt("Select a number", type=int)
    if choice == 0:
        return None
    return rows[choice - 1]

def _find_project(
    session,
    project_id: UUID = typer.Option(
        None,
        "--project-id",
        help="UUID of the project to add a claim to. If omitted, an interactive selector is shown.",
    ),   
) -> ProjectORM:
    return session.query(ProjectORM).filter(ProjectORM.project_id == project_id)

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
    with get_session() as session:
        project = _find_project(session, project_id)
        if not project_id:
            track = _select_existing_track()
            if track == "work":
                position = _select_existing_position(session)
                if position is None:
                    company = _select_existing_company(session)
                    position = _resolve_position(session, company)
                project = _select_existing_project(session, position)
                if project is None:
                    project = _resolve_project(session, position_id=position.position_id, degree_id=None)
            else:
                degree = _select_existing_degree(session)
                if degree is None:
                    school = _select_existing_school(session)
                    degree = _resolve_degree(session, school)
                project = _select_existing_project(session, degree)
                if project is None:
                    project = _resolve_project(session, position_id=None, degree_id=degree.degree_id)
        problem = _resolve_problem(session, project)
        action = _resolve_action(session, problem)
        claim = _resolve_claim(session, action)
        _resolve_metrics(session, claim)
        evidences = _resolve_evidences(session, claim)
        _resolve_tags(session, claim, problem, action, evidences, project)
        _resolve_governance(session, claim)

        claim_id = claim.claim_id

    fill_pending_support_levels()

    console.print(f"[green]Done! New claim ID: {claim_id}[/green]")

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
    source: str = project.project_context_source
    has_source: bool = False
    path: str

    if source is not None:
         has_source = typer.confirm(
                f"Found a source is your project. Would you like to derive claims from {source}?",
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


def add_claims(
        project_id: UUID = typer.Option(
        None,
        "--project-id",
        help="UUID of the project to add a claim to.", # If omitted, an interactive selector is shown.",
    ),
):
    """Bulk add claims to database through source files or URLs."""
    project: ProjectORM = None
    with get_session() as session:
        project = _find_project(session, project_id) # if project_id else _select_existing_project(session)

    raw_claims = _resolve_claims(project) 
    for idx, raw_claim in enumerate(raw_claims):
        confirmed = {}
        for key, val, prompt in zip(raw_claim.items(), [PROBLEM_REGENERATE_SYSTEM_PROMPT, None, ACTION_REGENERATE_SYSTEM_PROMPT, None, CLAIM_REGENERATE_SYSTEM_PROMPT]): 
            confirmed_val = _confirm_or_edit(val, key, prompt)
            confirmed[key] = confirmed_val

        with get_session() as session:
            problem = _resolve_problem(session, project, confirmed["problem_description"], confirmed["stakeholder_impact"])
            action = _resolve_action(session, problem, confirmed["action_description"], confirmed["role_scope"])
            claim = _resolve_claim(session, action, confirmed["claim_description"])
            _resolve_metrics(session, claim)
            evidences = _resolve_evidences(session, claim, project)
            _resolve_tags(session, claim, problem, action, evidences, project)
            _resolve_governance(session, claim)

        fill_pending_support_levels()
        console.print(f"[green] {project.project_name} has {(idx+1)} claim(s) written into database [/green]")


@app.command("add-project-claims")
def add_project_claims() -> None:
    """Bulk add a project and its claims to database through source files or URLs."""
    project_id = add_project()
    add_claims(project_id)

if __name__ == "__main__":
    app()
