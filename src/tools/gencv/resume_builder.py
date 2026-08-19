"""
resume_builder.py
Version: v14 (2026-08-16)

Changelog:
- v14 (2026-08-16): build_skills_section() now filters to
tag_category == "skill" ONLY before calling skill_grouper.
group_skills_for_jd() -- domain and job_function tags are no longer
passed into the skills section grouping at all. Rationale (per user
decision, 2026-08-16): domain (industry knowledge) and job_function
(role-type) tags describe WHAT DOMAIN/ROLE the candidate operates in,
not a technical skill -- mixing them into a "skills" section grouped by
JD-specific presentation logic (skill_grouper.py) was conflating two
different kinds of information. These two tag categories are now
surfaced instead via summary_writer.py v2's optional
domain_job_function_tags parameter (auxiliary context for the resume
summary, not a forced-inclusion requirement). Added a new helper,
fetch_domain_job_function_tags(), so pipeline.py can collect these tags
and pass them to summary_writer.draft_and_confirm_summary() without
resume_builder.py and summary_writer.py needing to share internal query
logic.
- v13 (2026-08-16): build_skills_section()/build_resume_yaml() gained a
required job_description parameter, threading through to the new
gencv/skill_grouper.py module -- a single batch LLM call (capped at 4
groups) replaces the previous hardcoded skill/domain/job_function
label_map, followed by a mandatory interactive human confirmation step.
- v12 (2026-08-16) -- QA fixes: sections["projects"] only written when
non-empty; build_education_entries() signature simplified (dropped
degree_ids_with_projects), using relevant_degrees directly as the single
source of truth for the education section.
- v11 (2026-08-16): Reverted build_education_entries()'s sort key from
end_date back to start_date.
- v10 (2026-08-16): Added build_project_entries().
- v9 (2026-08-15): summary moved into sections dict as a TextEntry.
- v1 (2026-08-10, inferred): Initial resume YAML assembly + rendercv CLI call.

把選定的經歷 + 身份 + 連結 + 教育 + 專案 + 摘要 + JD-specific skills 從資料庫組裝成
履歷內容 YAML，交給 rendercv 官方套件排版編譯成最終 PDF。
"""

import pathlib
import subprocess
import sys
from collections import defaultdict

import yaml

from core.db import get_session, ClaimMetricORM, ClaimTagORM, TagORM, IdentityORM, NetworkORM, DegreeORM, SchoolORM
from gencv.experience_matcher import fetch_experience_contexts, to_uuid_list
from gencv.skill_grouper import group_skills_for_jd

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
    子集，依 project_name 分組，組成 rendercv ProjectEntry 清單。"""
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
    直接組成 education section，不再額外篩選。只放
    institution/area/degree/start_date/end_date。排序依 start_date 倒序。"""
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


def build_skills_section(session, claim_ids: list[str], job_description: str) -> list[dict]:
    """從關聯的標籤中取得 tag_name，交給 skill_grouper.group_skills_for_jd()
    依這份 JD 重新分組（上限4組，含未分類 tag 的互動式指派、以及新增分組的
    互動指令），輸出 rendercv OneLineEntry 清單。

    (v14) 只撈 tag_category == "skill" 的 tag -- domain / job_function
    類 tag 已改由 summary_writer.py 的 domain_job_function_tags 參數處理
    （見 fetch_domain_job_function_tags()），不再進入 skills section。
    """
    if not claim_ids:
        return []

    tag_rows = (
        session.query(TagORM)
        .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .filter(TagORM.tag_category == "skill")
        .all()
    )

    seen_tags: set[str] = set()
    tags_with_category: list[tuple[str, str]] = []
    for tag in tag_rows:
        if tag.tag_name not in seen_tags:
            seen_tags.add(tag.tag_name)
            tags_with_category.append((tag.tag_name, tag.tag_category))

    if not tags_with_category:
        return []

    grouping = group_skills_for_jd(tags_with_category, job_description)

    return [
        {"label": label, "details": ", ".join(tags)}
        for label, tags in grouping.items()
        if tags
    ]


def fetch_domain_job_function_tags(session, claim_ids: list[str]) -> list[str]:
    """(v14, new) 撈出 claim_ids 關聯到的 domain / job_function 類 tag_name
    清單（去重），供 pipeline.py 傳給
    summary_writer.draft_and_confirm_summary() 當輔助訊號使用。與
    build_skills_section() 互斥 -- 該函式只處理 tag_category == "skill"，
    這裡處理另外兩類，兩者合起來剛好覆蓋全部 tag_category，不重疊。
    """
    if not claim_ids:
        return []

    tag_rows = (
        session.query(TagORM)
        .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .filter(TagORM.tag_category.in_(["domain", "job_function"]))
        .all()
    )

    seen: set[str] = set()
    result: list[str] = []
    for tag in tag_rows:
        if tag.tag_name not in seen:
            seen.add(tag.tag_name)
            result.append(tag.tag_name)
    return result


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
    job_description: str,
    headline: str | None = None,
    networks: list[NetworkORM] | None = None,
    relevant_degrees: list[DegreeORM] | None = None,
    description_overrides: dict[str, str] | None = None,
    name: str | None = None,
    summary: str | None = None,
) -> str:
    """組出 rendercv 看得懂的 cv YAML（只含 cv: 區塊），寫入 output_path。

    job_description 為必填參數 -- build_skills_section() 需要它來做
    JD-specific 的 skill 分組（僅 tag_category == "skill" 的 tag，v14）。

    summary 為選填的履歷開場摘要，寫進 sections["summary"]（單元素字串 list，
    RenderCV 的 TextEntry 寫法）。None 代表使用者選擇省略。
    """
    networks = networks or []
    relevant_degrees = relevant_degrees or []

    with get_session() as session:
        experience_entries = build_experience_entries(session, claim_ids, description_overrides)
        project_entries = build_project_entries(session, claim_ids, description_overrides)
        education_entries = build_education_entries(session, relevant_degrees)
        skills_entries = build_skills_section(session, claim_ids, job_description)

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
