"""
onboard/cli.py
Typer CLI entrypoint for the onboard module.

Implements:
- `add-project`: walks through Step 0-4 (company/school -> position/degree
  -> project), single transactional write.
- `add-claim`: walks through Step 0-5 (select project -> problem -> action ->
  claim/description -> evidence -> tags), single transactional write.

NOTE (Aug 2026): the education-track mid-level entity is called `degree`
in the actual Postgres schema (table `degree`, column `degree_id`,
`degree_name`), NOT `education`/`education_id`.

Design notes (per Aug 2026 review):
- --project-id is OPTIONAL on add-claim. If omitted, the user is walked
  through an interactive selector that mirrors the add-project UX
  (company/school -> position/degree -> project), reusing existing
  list_rows()/Table-based selection logic already proven in add-project.
- description prompt wording no longer overlaps with action's wording
  (see onboard/prompts.py RESULT_CLAIM_SYSTEM_PROMPT).
- evidence_type is a fixed numbered menu (not free text), and evidence_level
  is auto-derived from the chosen type via EVIDENCE_TYPE_TO_LEVEL.
- Tags are NOT typed by the user. After problem/action/claim/evidence are
  collected, the LLM reads that same in-memory content (no re-pasting) and
  proposes tags; the user reviews/edits the proposed list. Tagging itself
  is a mandatory step of add-claim -- not a yes/no gate -- only the exact
  tag content is adjustable.
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
    PositionORM,
    ProblemORM,
    ProjectORM,
    SchoolORM,
    TagORM,
    create_row,
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

def _prompt_date(label: str, allow_blank: bool = False) -> Optional[date]:
    while True:
        raw = typer.prompt(label, default="" if allow_blank else ...)
        if allow_blank and raw.strip() == "":
            return None
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]日期格式錯誤，請用 YYYY-MM-DD[/red]")


def _confirm_or_edit(draft: str, label: str) -> str:
    console.print(f"\n[bold]{label}（LLM 草稿）[/bold]")
    console.print(draft)
    edited = typer.prompt(
        "按 Enter 確認採用，或直接輸入新文字覆蓋",
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
    table = Table(title=f"找到 {len(candidates)} 筆相似的既有紀錄")
    table.add_column("#")
    table.add_column("名稱")
    table.add_column("相似度")

    for idx, (row, score) in enumerate(candidates, start=1):
        table.add_row(str(idx), getattr(row, label_field), f"{score:.0f}%")
    table.add_row("0", "[新增一筆]", "")
    console.print(table)

    choice = typer.prompt("請選擇編號", type=int)
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

    choice = typer.prompt("請選擇編號", type=int)
    if allow_new_label is not None and choice == 0:
        return None
    return rows[choice - 1]


def _select_from_menu(options: list[str], title: str, allow_blank: bool = False) -> Optional[str]:
    """
    Fixed-choice numbered menu -- used for fields where free-typing would
    lead to inconsistent values across records (e.g. role_scope, value_type,
    metric_confidence). If `allow_blank` is True, option 0 lets the user
    skip the field entirely (returns None).
    """
    table = Table(title=title)
    table.add_column("#")
    table.add_column("選項")
    start = 0 if allow_blank else 1
    if allow_blank:
        table.add_row("0", "[留空]")
    for idx, option in enumerate(options, start=1):
        table.add_row(str(idx), option)
    console.print(table)

    choice = typer.prompt("請選擇編號", type=int)
    if allow_blank and choice == 0:
        return None
    return options[choice - 1]


# ---------------------------------------------------------------------------
# Step 1: upper-level entity (company / school)
# ---------------------------------------------------------------------------

def _resolve_company(session) -> CompanyORM:
    name = typer.prompt("公司名稱")
    candidates = fuzzy_find_candidates(session, CompanyORM, "company_name", name)
    if candidates:
        chosen = _select_from_fuzzy_candidates(candidates, "company_name")
        if chosen is not None:
            return chosen

    location = typer.prompt("地點", default="", show_default=False)
    return create_row(
        session,
        CompanyORM,
        {"company_name": name, "location": location or None},
    )


def _resolve_school(session) -> SchoolORM:
    name = typer.prompt("學校名稱")
    candidates = fuzzy_find_candidates(session, SchoolORM, "school_name", name)
    if candidates:
        chosen = _select_from_fuzzy_candidates(candidates, "school_name")
        if chosen is not None:
            return chosen

    location = typer.prompt("地點", default="", show_default=False)
    return create_row(
        session,
        SchoolORM,
        {"school_name": name, "location": location or None},
    )


# ---------------------------------------------------------------------------
# Step 2: mid-level entity (position / degree)
# ---------------------------------------------------------------------------

def _draft_responsibility() -> tuple[str, Optional[str]]:
    """Returns (responsibility_text, responsibility_source)."""
    client = get_llm_client()
    has_source = typer.confirm("是否有原始資料可連結職責描述（檔案路徑）？", default=False)

    if has_source:
        while True:
            path = typer.prompt("請輸入檔案路徑（.txt / .md / .pdf）")
            try:
                source_text = read_source_file(path)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                console.print(f"[red]{exc}[/red]")

        response = client.generate_text(
            system_prompt=RESPONSIBILITY_SYSTEM_PROMPT,
            user_prompt=build_responsibility_prompt_from_source(source_text),
        )
        final_text = _confirm_or_edit(response.text, "職責摘要")
        return final_text, path

    description = typer.prompt("請簡述這個職位的職責範圍")
    response = client.generate_text(
        system_prompt=RESPONSIBILITY_SYSTEM_PROMPT,
        user_prompt=build_responsibility_prompt_from_description(description),
    )
    final_text = _confirm_or_edit(response.text, "職責摘要")
    return final_text, None


def _resolve_position(session, company: CompanyORM) -> PositionORM:
    existing = list_rows(session, PositionORM, company_id=company.company_id)
    if existing:
        chosen = _select_from_list(
            existing,
            [("職稱", "title"), ("部門", "department"), ("起始日期", "start_date")],
            title=f"{company.company_name} 現有職位",
            allow_new_label="新增職位",
        )
        if chosen is not None:
            return chosen

    title = typer.prompt("職稱")
    department = typer.prompt("部門", default="", show_default=False)
    start_dt = _prompt_date("起始日期 (YYYY-MM-DD)")
    end_dt = _prompt_date("結束日期 (YYYY-MM-DD，可留空表示至今)", allow_blank=True)
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
            [("學位/學程", "degree_name"), ("專業", "subject"), ("起始日期", "start_date")],
            title=f"{school.school_name} 現有學業紀錄",
            allow_new_label="新增學業紀錄",
        )
        if chosen is not None:
            return chosen

    degree_name = typer.prompt("學位/學程名稱")
    subject = typer.prompt("專業", default="", show_default=False)
    start_dt = _prompt_date("起始日期 (YYYY-MM-DD)")
    end_dt = _prompt_date("結束日期 (YYYY-MM-DD，可留空)", allow_blank=True)

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
    project_name = typer.prompt("專案名稱")
    start_dt = _prompt_date("專案起始日期 (YYYY-MM-DD)")
    end_dt = _prompt_date("專案結束日期 (YYYY-MM-DD，可留空)", allow_blank=True)
    description = typer.prompt("簡述這個專案的背景/情境")

    client = get_llm_client()
    response = client.generate_text(
        system_prompt=PROJECT_CONTEXT_SYSTEM_PROMPT,
        user_prompt=build_project_context_prompt(description),
    )
    project_context = _confirm_or_edit(response.text, "專案背景敘述")

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
    return typer.confirm("\n確認寫入資料庫？", default=True)


# ---------------------------------------------------------------------------
# add-project command
# ---------------------------------------------------------------------------

@app.command("add-project")
def add_project() -> None:
    """互動式新增一筆工作或學業經驗下的專案紀錄。"""
    track = typer.prompt("這是工作經驗還是學業經驗？ [work/education]").strip().lower()
    if track not in ("work", "education"):
        console.print("[red]請輸入 work 或 education[/red]")
        raise typer.Exit(code=1)

    project_id: Optional[UUID] = None

    with get_session() as session:
        if track == "work":
            company = _resolve_company(session)
            position = _resolve_position(session, company)
            project = _resolve_project(session, position_id=position.position_id, degree_id=None)
        else:
            school = _resolve_school(session)
            degree = _resolve_degree(session, school)
            project = _resolve_project(session, position_id=None, degree_id=degree.degree_id)

        if not _confirm_final_write():
            console.print("[yellow]已取消，變更不會寫入。[/yellow]")
            raise typer.Exit(code=0)

        project_id = project.project_id
        # transaction commits automatically on successful exit of get_session()

    console.print(f"[green]完成！新專案 ID：{project_id}[/green]")


# ---------------------------------------------------------------------------
# add-claim: interactive project selector
# ---------------------------------------------------------------------------

def _select_project_interactively(session) -> ProjectORM:
    """
    Mirrors the add-project UX: pick company/school first (narrows the
    search space), then position/degree under it, then project under
    that. All lookups are indexed FK queries via list_rows() -- no fuzzy
    search needed, and the UX is identical to what the user already knows
    from add-project.
    """
    track = typer.prompt("這個專案屬於工作經驗還是學業經驗？ [work/education]").strip().lower()
    if track not in ("work", "education"):
        console.print("[red]請輸入 work 或 education[/red]")
        raise typer.Exit(code=1)

    if track == "work":
        companies = list_rows(session, CompanyORM)
        if not companies:
            console.print("[red]目前沒有任何公司紀錄，請先執行 add-project[/red]")
            raise typer.Exit(code=1)
        company = _select_from_list(
            companies, [("公司名稱", "company_name")], title="請選擇公司"
        )

        positions = list_rows(session, PositionORM, company_id=company.company_id)
        if not positions:
            console.print("[red]這個公司底下沒有任何職位紀錄[/red]")
            raise typer.Exit(code=1)
        position = _select_from_list(
            positions,
            [("職稱", "title"), ("部門", "department"), ("起始日期", "start_date")],
            title=f"{company.company_name} 現有職位",
        )

        projects = list_rows(session, ProjectORM, position_id=position.position_id)
        parent_label = position.title
    else:
        schools = list_rows(session, SchoolORM)
        if not schools:
            console.print("[red]目前沒有任何學校紀錄，請先執行 add-project[/red]")
            raise typer.Exit(code=1)
        school = _select_from_list(
            schools, [("學校名稱", "school_name")], title="請選擇學校"
        )

        degrees = list_rows(session, DegreeORM, school_id=school.school_id)
        if not degrees:
            console.print("[red]這個學校底下沒有任何學業紀錄[/red]")
            raise typer.Exit(code=1)
        degree = _select_from_list(
            degrees,
            [("學位/學程", "degree_name"), ("專業", "subject"), ("起始日期", "start_date")],
            title=f"{school.school_name} 現有學業紀錄",
        )

        projects = list_rows(session, ProjectORM, degree_id=degree.degree_id)
        parent_label = degree.degree_name

    if not projects:
        console.print(f"[red]{parent_label} 底下沒有任何專案紀錄，請先執行 add-project[/red]")
        raise typer.Exit(code=1)

    return _select_from_list(
        projects,
        [("專案名稱", "project_name"), ("起始日期", "start_date")],
        title=f"{parent_label} 現有專案",
    )


# ---------------------------------------------------------------------------
# add-claim: Step 1 -- problem
# ---------------------------------------------------------------------------

def _resolve_problem(session, project: ProjectORM) -> ProblemORM:
    description = typer.prompt("請描述這個專案中遇到的問題（不需描述解決方式）")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=PROBLEM_SYSTEM_PROMPT,
        user_prompt=build_problem_prompt(description),
    )
    problem_description = _confirm_or_edit(response.text, "問題敘述")

    stakeholder_impact = typer.prompt("這個問題影響了哪些關係人？（可留空）", default="", show_default=False)
    baseline_metric = typer.prompt("問題發生前的基準數據/狀態？（可留空）", default="", show_default=False)

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
    description = typer.prompt("你針對這個問題做了什麼？（只描述行動本身）")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=ACTION_SYSTEM_PROMPT,
        user_prompt=build_action_prompt(description),
    )
    action_description = _confirm_or_edit(response.text, "行動敘述")

    role_scope = _select_from_menu(
        ROLE_SCOPE_OPTIONS,
        title="你在這個行動中的角色？",
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
    raw_input_text = typer.prompt("這個行動帶來了什麼具體結果？（只描述結果，不要重複行動本身）")
    client = get_llm_client()
    response = client.generate_text(
        system_prompt=RESULT_CLAIM_SYSTEM_PROMPT,
        user_prompt=build_result_claim_prompt(raw_input_text),
    )
    description = _confirm_or_edit(response.text, "結果敘述")

    # value_type / metric / metric_confidence no longer live on claim itself --
    # see _resolve_metrics() below, which supports multiple quantifiable
    # metrics per claim via claim_metric.
    return create_row(
        session,
        ClaimORM,
        {
            "action_id": action.action_id,
            "description": description,
        },
    )


# ---------------------------------------------------------------------------
# add-claim: Step 3.5 -- metrics (LLM-extracted from the claim description,
# supports multiple metrics per claim)
# ---------------------------------------------------------------------------
# LLM only extracts metrics explicitly present in the text -- it does not
# estimate/invent values, and it never outputs metric_confidence (LLM
# self-reported confidence is poorly calibrated; confidence here is always
# a manual human judgment made during review).

def _review_metric_drafts(drafts: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    """
    Shows LLM-extracted metric drafts (type, label, value) and lets the
    user accept/edit/delete each one, plus add any the LLM missed. Returns
    a final list of (metric_type, metric_label, metric_value, metric_confidence)
    tuples -- confidence is always picked by the user here, never by the LLM.
    """
    console.print("\n[bold]量化指標草稿（LLM 從結果敘述中抽取）[/bold]")
    if drafts:
        table = Table()
        table.add_column("#")
        table.add_column("類型")
        table.add_column("標籤")
        table.add_column("數值")
        for idx, (m_type, m_label, m_value) in enumerate(drafts, start=1):
            table.add_row(str(idx), m_type, m_label, m_value)
        console.print(table)
    else:
        console.print("[dim]（LLM 未偵測到任何可量化指標）[/dim]")

    finalized: list[tuple[str, str, str, str]] = []
    for m_type, m_label, m_value in drafts:
        keep = typer.confirm(f"保留「{m_label}: {m_value}」這筆指標？", default=True)
        if not keep:
            continue
        edited_label = typer.prompt("標籤", default=m_label)
        edited_value = typer.prompt("數值", default=m_value)
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f"「{edited_label}」這筆數據的確信程度？",
            allow_blank=False,
        )
        finalized.append((m_type, edited_label, edited_value, confidence))

    while typer.confirm("要手動新增一筆指標嗎？（LLM 沒抓到的可以在這裡補）", default=False):
        m_type = _select_from_menu(METRIC_TYPE_OPTIONS, title="指標類型？", allow_blank=False)
        m_label = typer.prompt("標籤（這個數字代表什麼）")
        m_value = typer.prompt("數值")
        confidence = _select_from_menu(
            METRIC_CONFIDENCE_OPTIONS,
            title=f"「{m_label}」這筆數據的確信程度？",
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
# add-claim: Step 4 -- evidence (a claim may have multiple pieces of
# evidence, e.g. one L1 direct artifact plus a supporting L3 mention)
# ---------------------------------------------------------------------------

def _select_evidence_type() -> tuple[str, str]:
    choices = list(EVIDENCE_TYPE_TO_LEVEL.keys())
    table = Table(title="請選擇證據類型（等級將自動對應）")
    table.add_column("#")
    table.add_column("證據類型")
    table.add_column("對應等級")
    for idx, choice in enumerate(choices, start=1):
        table.add_row(str(idx), choice, EVIDENCE_TYPE_TO_LEVEL[choice])
    console.print(table)

    idx = typer.prompt("請選擇編號", type=int)
    evidence_type = choices[idx - 1]
    evidence_level = EVIDENCE_TYPE_TO_LEVEL[evidence_type]
    return evidence_type, evidence_level


def _resolve_one_evidence(session, claim: ClaimORM) -> EvidenceORM:
    evidence_type, evidence_level = _select_evidence_type()
    evidence_location = typer.prompt(
        "證據存放位置/連結？（例如檔案路徑、連結，可留空）", default="", show_default=False
    )

    console.print(f"[dim]對應證據等級：{evidence_level}（依證據類型自動判定）[/dim]")

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
    """
    A single claim may have multiple pieces of evidence (e.g. a direct L1
    artifact plus a supporting L3 mention from a performance review). This
    loops _resolve_one_evidence() until the user says there's nothing more
    to add -- at least one evidence entry is required per claim.
    """
    evidences: list[EvidenceORM] = [_resolve_one_evidence(session, claim)]
    while typer.confirm("這個 claim 還有其他證據要補充嗎？", default=False):
        evidences.append(_resolve_one_evidence(session, claim))
    return evidences


# ---------------------------------------------------------------------------
# add-claim: Step 5 -- tags (LLM-generated, mandatory step)
# ---------------------------------------------------------------------------
# Problem 1 fix: tags are generated by the LLM from the claim's full content,
# not typed by the user.
# Problem 2 fix: this step is not a yes/no gate -- it always runs as part of
# add-claim. Only the proposed tag list itself is editable.
# Problem 3 fix: the LLM prompt is built directly from problem/action/claim/
# evidence text already held in memory during this run -- the user never
# re-pastes anything.

def _confirm_or_edit_tags(proposed: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Shows the LLM-proposed tag list and lets the user accept as-is, or
    replace it with their own comma-separated list. Returns a list of
    (tag_name, tag_category) tuples.
    """
    console.print("\n[bold]標籤草稿（LLM 根據這筆紀錄的完整內容生成）[/bold]")
    table = Table()
    table.add_column("#")
    table.add_column("標籤")
    table.add_column("類別")
    for idx, (name, category) in enumerate(proposed, start=1):
        table.add_row(str(idx), name, category)
    console.print(table)

    edited = typer.prompt(
        "按 Enter 確認採用以上標籤，或輸入新的標籤列表（格式：標籤1|類別1, 標籤2|類別2, ...）",
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
    """
    Resolves the claim's linked static (relatively unchanging) data --
    company/position for work-track projects, or school/degree for
    education-track projects -- by walking up from the already-selected
    project. This context is passed into tag generation so the LLM can
    infer domain/job_function tags implied by *where* the work happened
    (e.g. company industry, job title), not just from the problem/action/
    result text.
    """
    if project.position_id is not None:
        position = session.get(PositionORM, project.position_id)
        company = session.get(CompanyORM, position.company_id) if position else None
        parts = []
        if company is not None:
            parts.append(f"公司：{company.company_name}" + (f"（{company.location}）" if company.location else ""))
        if position is not None:
            parts.append(f"職稱：{position.title}" + (f"，部門：{position.department}" if position.department else ""))
        return "；".join(parts)

    if project.degree_id is not None:
        degree = session.get(DegreeORM, project.degree_id)
        school = session.get(SchoolORM, degree.school_id) if degree else None
        parts = []
        if school is not None:
            parts.append(f"學校：{school.school_name}" + (f"（{school.location}）" if school.location else ""))
        if degree is not None:
            parts.append(f"學位/學程：{degree.degree_name}" + (f"，專業：{degree.subject}" if degree.subject else ""))
        return "；".join(parts)

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
    evidence_types_joined = "、".join(sorted({e.evidence_type for e in evidences}))

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
# add-claim command
# ---------------------------------------------------------------------------

@app.command("add-claim")
def add_claim(
    project_id: Optional[UUID] = typer.Option(
        None,
        "--project-id",
        help="要新增 claim 的 project UUID。若省略，將進入互動式選擇流程。",
    )
) -> None:
    """互動式新增一筆 problem -> action -> claim -> metric(s) -> evidence(s) -> tag 紀錄。"""
    claim_id: Optional[UUID] = None
    evidence_ids: list[UUID] = []
    metric_summaries: list[str] = []
    tag_names: list[str] = []

    with get_session() as session:
        if project_id is not None:
            project = session.get(ProjectORM, project_id)
            if project is None:
                console.print(f"[red]找不到 project_id：{project_id}[/red]")
                raise typer.Exit(code=1)
        else:
            project = _select_project_interactively(session)

        console.print(f"[bold]已選擇專案：{project.project_name}[/bold]\n")

        problem = _resolve_problem(session, project)
        action = _resolve_action(session, problem)
        claim = _resolve_claim(session, action)
        metrics = _resolve_metrics(session, claim)
        evidences = _resolve_evidences(session, claim)
        tags = _resolve_tags(session, claim, problem, action, evidences, project)

        if not _confirm_final_write():
            console.print("[yellow]已取消，變更不會寫入。[/yellow]")
            raise typer.Exit(code=0)

        claim_id = claim.claim_id
        evidence_ids = [evidence.evidence_id for evidence in evidences]
        metric_summaries = [f"{m.metric_label}: {m.metric_value}" for m in metrics]
        tag_names = [tag.tag_name for tag in tags]
        # transaction commits automatically on successful exit of get_session()

    console.print(f"[green]完成！新 claim ID：{claim_id}[/green]")
    console.print(f"[green]證據 ID（共 {len(evidence_ids)} 筆）：{', '.join(str(e) for e in evidence_ids)}[/green]")
    if metric_summaries:
        console.print(f"[green]量化指標：{'; '.join(metric_summaries)}[/green]")
    if tag_names:
        console.print(f"[green]標籤：{', '.join(tag_names)}[/green]")


if __name__ == "__main__":
    app()
