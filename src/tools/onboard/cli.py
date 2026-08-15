"""
onboard/cli.py
Version: v1.9 (2026-08-10)

Changelog:
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
    ClaimMetricORM,
    ClaimORM,
    ClaimTagORM,
    CompanyORM,
    DegreeORM,
    EvidenceORM,
    GovernanceORM,
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
from core.file_reader import FileReadError, UnsupportedFileTypeError, read_source_file
from core.llm_client import get_llm_client
from onboard.prompts import (
    ACTION_SYSTEM_PROMPT,
    EVIDENCE_TYPE_TO_LEVEL,
    METRIC_CONFIDENCE_OPTIONS,
    METRIC_EXTRACTION_SYSTEM_PROMPT,
    METRIC_TYPE_OPTIONS,
    PROBLEM_SYSTEM_PROMPT,
    PROJECT_CONTEXT_SYSTEM_PROMPT,
    RESPONSIBILITY_SYSTEM_PROMPT,
    RESULT_CLAIM_SYSTEM_PROMPT,
    ROLE_SCOPE_OPTIONS,
    TAG_SYSTEM_PROMPT,
    VALUE_TYPE_OPTIONS,
    build_action_prompt,
    build_metric_extraction_prompt,
    build_problem_prompt,
    build_project_context_prompt,
    build_responsibility_prompt_from_description,
    build_responsibility_prompt_from_source,
    build_result_claim_prompt,
    build_tag_prompt,
    parse_metric_response,
    parse_tag_response,
)

app = typer.Typer()
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
# add-project (v1.9): flat position/degree selectors
# ---------------------------------------------------------------------------
# Joins position+company (work track) or degree+school (education track)
# directly, skipping the separate "pick/create company" then "pick/create
# position" steps whenever existing rows are available. Returns None in
# TWO cases, both of which signal the caller to fall back to the original
# _resolve_company -> _resolve_position (or _resolve_school ->
# _resolve_degree) stepwise flow:
#   1. No rows exist at all yet (menu isn't even shown).
#   2. The user explicitly picks "[Create new]" (option 0), even though
#      existing rows were available -- the escape hatch for genuinely
#      new companies/positions.


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
        "Do you have a source document to link the responsibility description to (file path)?",
        default=False,
    )

    if has_source:
        while True:
            path = typer.prompt("Enter file path (.txt / .md / .pdf)")
            try:
                source_text = read_source_file(path)
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


def _resolve_degree(session, school: SchoolORM) -> DegreeORM:
    existing = list_rows(session, DegreeORM, school_id=school.school_id)
    if existing:
        chosen = _select_from_list(
            existing,
            [("Degree/Program", "degree_name"), ("Subject", "subject"), ("Start date", "start_date")],
            title=f"Existing education records at {school.school_name}",
            allow_new_label="Add new education record",
        )
        if chosen is not None:
            return chosen

    degree_name = typer.prompt("Degree/program name")
    subject = typer.prompt("Subject", default="", show_default=False)
    start_dt = _prompt_date("Start date (YYYY-MM-DD)")
    end_dt = _prompt_date("End date (YYYY-MM-DD, leave blank)", allow_blank=True)

    return create_row(
        session,
        DegreeORM,
        {
            "school_id": school.school_id,
            "degree_name": degree_name,
            "subject": subject or None,
            "start_date": start_dt,
            "end_date": end_dt,
        },
    )


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
    description = typer.prompt("Briefly describe the background/context of this project")

    client = get_llm_client()
    response = client.generate_text(
        system_prompt=PROJECT_CONTEXT_SYSTEM_PROMPT,
        user_prompt=build_project_context_prompt(description),
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
    baseline_metric = typer.prompt(
        "What was the baseline metric/state before this problem? (optional)", default="", show_default=False
    )

    return create_row(
        session,
        ProblemORM,
        {
            "project_id": project.project_id,
            "description": problem_description,
            "stakeholder_impact": stakeholder_impact or None,
            "baseline_metric": baseline_metric or None,
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


def _review_metric_drafts(drafts: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    console.print("\n[bold]Metric drafts (extracted by LLM from the result statement)[/bold]")
    if drafts:
        table = Table()
        table.add_column("#")
        table.add_column("Type")
        table.add_column("Label")
        table.add_column("Value")
        for idx, (m_type, m_label, m_value) in enumerate(drafts, start=1):
            table.add_row(str(idx), m_type, m_label, m_value)
        console.print(table)
    else:
        console.print("[dim](LLM did not detect any quantifiable metrics)[/dim]")

    finalized: list[tuple[str, str, str, str]] = []
    for m_type, m_label, m_value in drafts:
        keep = typer.confirm(f'Keep the metric "{m_label}: {m_value}"?', default=True)
        if not keep:
            continue
        edited_label = typer.prompt("Label", default=m_label)
        edited_value = typer.prompt("Value", default=m_value)
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f'Confidence level for "{edited_label}"?',
            allow_blank=False,
        )
        finalized.append((m_type, edited_label, edited_value, confidence))

    while typer.confirm("Manually add another metric? (add anything the LLM missed here)", default=False):
        m_type = _select_from_menu(METRIC_TYPE_OPTIONS, title="Metric type?", allow_blank=False)
        m_label = typer.prompt("Label (what this number represents)")
        m_value = typer.prompt("Value")
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f'Confidence level for "{m_label}"?',
            allow_blank=False,
        )
        finalized.append((m_type, m_label, m_value, confidence))

    return finalized


def _resolve_metrics(session, claim: ClaimORM) -> list[ClaimMetricORM]:
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=METRIC_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_metric_extraction_prompt(claim.description),
    )
    drafts = parse_metric_response(response.text)
    finalized = _review_metric_drafts(drafts)

    metric_orms: list[ClaimMetricORM] = []
    for m_type, m_label, m_value, confidence in finalized:
        metric_orms.append(
            create_row(
                session,
                ClaimMetricORM,
                {
                    "claim_id": claim.claim_id,
                    "metric_type": m_type,
                    "metric_label": m_label,
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
            parts.append(f"Degree/Program: {degree.degree_name}" + (f", Subject: {degree.subject}" if degree.subject else ""))
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
            problem_description=problem.description,
            action_description=action.description,
            description=claim.description,
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
        evidence_ids = [evidence.evidence_id for evidence in evidences]
        metric_summaries = [f"{m.metric_label}: {m.metric_value}" for m in metrics]
        tag_names = [tag.tag_name for tag in tags]

    fill_pending_support_levels()

    console.print(f"[green]Done! New claim ID: {claim_id}[/green]")
    console.print(f"[green]Evidence IDs ({len(evidence_ids)} total): {', '.join(str(e) for e in evidence_ids)}[/green]")
    if metric_summaries:
        console.print(f"[green]Metrics: {'; '.join(metric_summaries)}[/green]")
    if tag_names:
        console.print(f"[green]Tags: {', '.join(tag_names)}[/green]")
    console.print("[green]Governance support_level: computed and filled in.[/green]")


if __name__ == "__main__":
    app()
