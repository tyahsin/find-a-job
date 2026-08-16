"""
resume_builder.py
Version: v12 (2026-08-16)

Changelog:
- v12 (2026-08-16) -- QA fixes:
  1. sections["projects"] is now only written when project_entries is
     non-empty (`if project_entries: sections["projects"] = project_entries`).
     Previously it was written unconditionally, so a resume with zero
     education-track claims would get an empty "projects" heading rendered
     with nothing under it.
  2. build_education_entries() signature simplified: DROPPED the
     degree_ids_with_projects parameter entirely. Previous logic filtered
     relevant_degrees down to only degrees with at least one selected
     project underneath them -- this directly contradicted the purpose of
     education_selector.py's orphan-degree path (Path B: degrees with ZERO
     claims/projects attached, judged relevant to the JD by a dedicated LLM
     call). That filter silently discarded every orphan degree
     education_selector.py worked to surface, making that file's Path B
     logic dead code in practice. Fixed by using relevant_degrees directly,
     with no post-filtering -- select_relevant_degrees() (Path A union
     Path B) is now the single source of truth for which degrees appear in
     the education section. Whether a degree also has a rendered project
     entry is now completely independent of whether it appears in
     education -- these were two separate questions that a shared filter
     had incorrectly coupled together.

     This also removes a computational dependency: build_education_entries()
     no longer needs build_project_entries() to run first (it no longer
     consumes degree_ids_with_projects at all), and no longer re-queries
     DegreeORM from the database (relevant_degrees are already loaded ORM
     objects) -- only SchoolORM is queried, by degree.school_id, to resolve
     institution names. This replaces the previous
     `session.query(DegreeORM, SchoolORM).join(...)` re-fetch with a single
     `session.query(SchoolORM).filter(school_id.in_(...))` lookup.
- v11 (2026-08-16): Reverted build_education_entries()'s sort key from
  end_date back to start_date.
- v10 (2026-08-16): Added build_project_entries() -- education-track claims
  (is_work_track=False) grouped by project_name into a "projects" section.
- v9 (2026-08-15) -- BUGFIX: cv.summary is NOT a valid top-level field in
  RenderCV's schema. Fixed by moving summary into sections dict as a
  TextEntry (single-element string list), as the first section key.
- v8 (2026-08-15): [BUGGY] Added summary param wrongly as cv_block["summary"].
- v7 (2026-08-15): sections dict always includes "education" as a key.
- v6 (2026-08-15): Removed degree.degree_description -> entry["summary"].
- v5 (2026-08-15): format_highlight() commented out.
- v4 (2026-08-15): work-track filtering, education entries added.
- v3 (2026-08-15): headline optional, identity/network integration.
- v1 (2026-08-10, inferred): Initial resume YAML assembly + rendercv CLI call.

把選定的經歷 + 身份 + 連結 + 教育 + 專案 + 摘要從資料庫組裝成履歷內容 YAML，交給
rendercv 官方套件排版編譯成最終 PDF。
"""

import pathlib
import subprocess
import sys
from collections import defaultdict

import yaml

from core.db import get_session, ClaimMetricORM, ClaimTagORM, TagORM, IdentityORM, NetworkORM, DegreeORM, SchoolORM
from gencv.experience_matcher import fetch_experience_contexts, to_uuid_list

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
print(f"[INFO] Using templates from {TEMPLATES_DIR.resolve()}")


def build_experience_entries(
    session, claim_ids: list[str], description_overrides: dict[str, str] | None = None
) -> list[dict]:
    """依 claim_id 清單取得完整背景，篩出 WORK TRACK（is_work_track=True）子集，
    依組織+職稱分組，組成 rendercv ExperienceEntry 清單。"""
    contexts = fetch_experience_contexts(session, claim_ids)
    description_overrides = description_overrides or {}

    work_claim_ids = [cid for cid in claim_ids if contexts.get(cid, {}).get("is_work_track")]

    metrics_by_claim: dict[str, list] = defaultdict(list)
    if work_claim_ids:
        metric_rows = (
            session.query(ClaimMetricORM)
            .filter(ClaimMetricORM.claim_id.in_(to_uuid_list(work_claim_ids)))
            .all()
        )
        for m in metric_rows:
            metrics_by_claim[str(m.claim_id)].append(m)

    grouped: dict[tuple, list[str]] = defaultdict(list)
    meta: dict[tuple, dict] = {}

    for claim_id in work_claim_ids:
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        key = (ctx["org_name"], ctx["title"], ctx["start_date"], ctx["end_date"])
        highlight = description_overrides.get(claim_id, ctx["claim_description"])
        grouped[key].append(highlight)
        meta[key] = ctx

    entries = []
    for (org_name, title, start_date, end_date), highlights in grouped.items():
        entry = {
            "company": org_name,
            "position": title,
            "highlights": highlights,
        }
        if start_date:
            entry["start_date"] = str(start_date)
            entry["end_date"] = str(end_date) if end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_project_entries(
    session, claim_ids: list[str], description_overrides: dict[str, str] | None = None
) -> list[dict]:
    """依 claim_id 清單取得完整背景，篩出 EDUCATION TRACK（is_work_track=False）
    子集，依 project_name 分組，組成 rendercv ProjectEntry 清單。

    (v12) 不再回傳 degree_ids_with_projects -- build_education_entries() 已
    改用 relevant_degrees 作為唯一依據，不再需要這份集合去做過濾。
    """
    contexts = fetch_experience_contexts(session, claim_ids)
    description_overrides = description_overrides or {}

    project_claim_ids = [cid for cid in claim_ids if not contexts.get(cid, {}).get("is_work_track")]

    if not project_claim_ids:
        return []

    grouped: dict[tuple, list[str]] = defaultdict(list)
    meta: dict[tuple, dict] = {}

    for claim_id in project_claim_ids:
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        key = (ctx["project_name"], ctx["start_date"], ctx["end_date"])
        highlight = description_overrides.get(claim_id, ctx["claim_description"])
        grouped[key].append(highlight)
        meta[key] = ctx

    entries = []
    for (project_name, start_date, end_date), highlights in grouped.items():
        entry = {
            "name": project_name,
            "highlights": highlights,
        }
        if start_date:
            entry["start_date"] = str(start_date)
            entry["end_date"] = str(end_date) if end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_education_entries(session, relevant_degrees: list[DegreeORM]) -> list[dict]:
    """把 education_selector.select_relevant_degrees() 選出的 DegreeORM 清單
    （Path A：claim-backed 學位，聯集 Path B：orphan 學位，經 LLM 判斷與 JD
    相關）直接組成 education section，不再額外篩選。

    (v12) 移除了 degree_ids_with_projects 過濾 -- 該過濾會把 Path B 選出的
    orphan degree（本來就沒有任何 claim/project）全部排除，等於讓
    education_selector.py 的 orphan-degree 判斷變成死代碼。一個學位要不要
    顯示在 education section，現在完全由 select_relevant_degrees() 的聯集
    結果決定；該學位底下是否也有 project 顯示在 sections["projects"]，是
    完全獨立的另一個問題（build_project_entries() 各自決定）。

    只查 SchoolORM 取得 institution 名稱（relevant_degrees 裡的 DegreeORM
    物件已經是 loaded 狀態，不需要重新 JOIN 查一次 degree 表）。

    只放 institution/area/degree/start_date/end_date（degree 本身的靜態
    資訊，不含任何 claim 內容）。排序依 start_date 倒序。
    """
    if not relevant_degrees:
        return []

    school_ids = [d.school_id for d in relevant_degrees if d.school_id]
    schools = (
        {s.school_id: s.school_name for s in session.query(SchoolORM).filter(SchoolORM.school_id.in_(school_ids)).all()}
        if school_ids else {}
    )

    entries = []
    for degree in relevant_degrees:
        entry: dict = {
            "institution": schools.get(degree.school_id),
            "area": degree.degree_name,
        }
        if degree.degree_level:
            entry["degree"] = degree.degree_level
        if degree.start_date:
            entry["start_date"] = str(degree.start_date)
        entry["end_date"] = str(degree.end_date) if degree.end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_skills_section(session, claim_ids: list[str]) -> list[dict]:
    """從關聯的標籤中，依 skill/domain/job_function 三類分組，輸出 rendercv OneLineEntry 清單。"""
    if not claim_ids:
        return []

    tag_rows = (
        session.query(TagORM)
        .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .all()
    )

    grouped: dict[str, set] = defaultdict(set)
    for tag in tag_rows:
        grouped[tag.tag_category].add(tag.tag_name)

    entries = []
    label_map = {"skill": "Skills", "domain": "Domain Knowledge", "job_function": "Functions"}
    for category in ["skill", "domain", "job_function"]:
        if grouped.get(category):
            entries.append({
                "label": label_map[category],
                "details": ", ".join(sorted(grouped[category])),
            })
    return entries


def build_social_networks_section(networks: list[NetworkORM]) -> list[dict]:
    """把 NetworkORM 清單轉成 rendercv SocialNetwork 清單（network/username 欄位）。"""
    return [
        {"network": n.network, "username": n.username}
        for n in networks
    ]


def build_resume_yaml(
    claim_ids: list[str],
    identity: IdentityORM,
    output_path: str,
    headline: str | None = None,
    networks: list[NetworkORM] | None = None,
    relevant_degrees: list[DegreeORM] | None = None,
    description_overrides: dict[str, str] | None = None,
    name: str | None = None,
    summary: str | None = None,
) -> str:
    """組出 rendercv 看得懂的 cv YAML（只含 cv: 區塊），寫入 output_path。

    (v12) build_project_entries() 跟 build_education_entries() 現在互相
    獨立，沒有執行順序依賴 -- 前者只需要 claim_ids，後者只需要
    relevant_degrees（來自 select_relevant_degrees()），兩者互不影響。

    summary 為選填的履歷開場摘要，寫進 sections["summary"]（單元素字串 list，
    RenderCV 的 TextEntry 寫法）。None 代表使用者選擇省略。
    """
    networks = networks or []
    relevant_degrees = relevant_degrees or []

    with get_session() as session:
        experience_entries = build_experience_entries(session, claim_ids, description_overrides)
        project_entries = build_project_entries(session, claim_ids, description_overrides)
        education_entries = build_education_entries(session, relevant_degrees)
        skills_entries = build_skills_section(session, claim_ids)

    cv_block: dict = {
        "name": name if name is not None else identity.name,
        "email": identity.email,
    }
    if headline:
        cv_block["headline"] = headline
    if identity.phone:
        cv_block["phone"] = identity.phone
    if identity.location:
        cv_block["location"] = identity.location

    social_networks_entries = build_social_networks_section(networks)
    if social_networks_entries:
        cv_block["social_networks"] = social_networks_entries

    sections: dict = {}
    if summary:
        sections["summary"] = [summary]
    sections["education"] = education_entries
    if project_entries:
        sections["projects"] = project_entries
    sections["experience"] = experience_entries
    sections["skills"] = skills_entries

    cv_block["sections"] = sections

    resume_content = {"cv": cv_block}

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resume_content, f, allow_unicode=True, sort_keys=False)

    print(
        f"Generated {output_path} with {len(experience_entries)} experience entries, "
        f"{len(project_entries)} project entries, and {len(education_entries)} education entries."
    )
    return output_path


def render_pdf(
    cv_yaml_path: pathlib.Path,
    design_path: pathlib.Path | None = None,
    locale_path: pathlib.Path | None = None,
    settings_path: pathlib.Path | None = None,
) -> None:
    """呼叫 `rendercv render` CLI，把 cv YAML 排版編譯成 PDF。"""
    design_path = design_path or TEMPLATES_DIR / "design.yaml"
    locale_path = locale_path or TEMPLATES_DIR / "locale.yaml"
    settings_path = settings_path or TEMPLATES_DIR / "settings.yaml"

    for f in [cv_yaml_path, design_path, locale_path, settings_path]:
        if not f.exists():
            print(f"[ERROR] 找不到檔案: {f}", file=sys.stderr)
            sys.exit(1)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    pdf_out_path = settings["settings"]["render_command"]["pdf_path"]

    cmd = [
        "rendercv", "render", str(cv_yaml_path),
        "--settings", str(settings_path),
        "--design", str(design_path),
        "--locale-catalog", str(locale_path)
    ]

    print("[INFO] Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] rendercv 執行失敗:", file=sys.stderr)
        if result.stderr:
            print("--- stderr ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print("--- stdout ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[OK] PDF 已產出: {pdf_out_path}")
    if result.stdout:
        print(result.stdout)

    return pdf_out_path
