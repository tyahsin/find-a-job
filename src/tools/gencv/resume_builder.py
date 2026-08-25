"""
resume_builder.py
Version: v17 (2026-08-19)

Changelog:
- v17 (2026-08-19): Per user request (2026-08-19):
  1. build_resume_yaml() renamed to build_cv_yaml() (name only, no
     signature/behavior change).
  2. render_pdf() no longer reads settings.yaml to determine the PDF
     output path -- that logic is fully removed. `--settings` is still
     passed to the rendercv CLI as before (settings.yaml may still hold
     other rendercv_settings the user cares about, e.g. bold_keywords),
     but the PDF destination is now ALWAYS computed in Python and passed
     via `--pdf-path`, never read back out of the YAML file.
  3. render_pdf()'s pdf_path parameter is now REQUIRED (previously
     optional) -- there is no more "fall back to settings.yaml's
     pdf_path" behavior, so a caller-supplied path is mandatory.
  4. build_default_output_path() replaced by build_cv_pdf_path(), which
     implements the new required naming scheme:
     - folder: {OUTPUT_DIR}/{job_title}_{company_name}/
     - filename: cv_{name}_{job_title}_{company_name}.pdf
     - full path: {OUTPUT_DIR}/{job_title}_{company_name}/cv_{name}_{job_title}_{company_name}.pdf
     OUTPUT_DIR comes from core.config.get_cv_config() (reads OUTPUT_DIR
     from .env). The folder is created via pdf_path.parent.mkdir(...) at
     render_pdf() call time if it doesn't exist yet -- build_cv_pdf_path()
     itself only computes the path, it doesn't touch the filesystem.
- v16 (2026-08-19): Renamed render_pdf()'s pdf_output_path parameter to
  pdf_path, and switched to core.config.get_cv_config() (renamed from
  get_apply_config() in config.py v3).
- v15 (2026-08-19): render_pdf() gained an optional pdf_output_path
  parameter, passed to `rendercv render` via the `--pdf-path` CLI flag.
- v14 (2026-08-16): build_skills_section() now filters to
  tag_category == "skill" ONLY before calling skill_grouper.
  group_skills_for_jd().
- v13 (2026-08-16): build_skills_section()/build_resume_yaml() gained a
  required job_description parameter, threading through to
  gencv/skill_grouper.py.
- v12 (2026-08-16) -- QA fixes: sections["projects"] only written when
  non-empty; build_education_entries() signature simplified.
- v11 (2026-08-16): Reverted build_education_entries()'s sort key.
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

from core.config import get_cv_config
from core.db import get_session, MetricORM, ClaimTagORM, TagORM, ClaimORM, ActionORM, ProblemORM, ProjectORM, CompanyORM, SchoolORM
from gencv.experience_matcher import fetch_experience_contexts, to_uuid_list
from gencv.skill_grouper import group_skills_for_jd
from gencv.identity_selector import select_identity_from_jd, select_networks_from_jd
from gencv.education_selector import select_relevant_degrees
from gencv.summary_writer import write_summary


TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"

def build_identity_entry(job_description, designated_name=None):
    identity = select_identity_from_jd(job_description)
    name = identity.name if designated_name is None else designated_name
    visa_status = identity.visa_status if 'citizenship' not in str.lower(identity.visa_status) else None
    return name, identity.email, identity.phone, identity.location, visa_status

def build_network_entries(job_description):
    networks = select_networks_from_jd(job_description)
    return [
        {"network": n.network, "username": n.username}
        for n in networks
    ]

def build_experience_entries(session, claim_ids, description_overrides=None):
    """依 claim_id 清單取得完整背景，篩出 WORK TRACK（is_work_track=True）子集，
    依組織+職稱分組，組成 rendercv ExperienceEntry 清單。"""
    contexts = fetch_experience_contexts(session, claim_ids)
    description_overrides = description_overrides or {}

    work_claim_ids = [cid for cid in claim_ids if contexts.get(cid, {}).get("is_work_track")]

    metrics_by_claim = defaultdict(list)
    if work_claim_ids:
        metric_rows = (
            session.query(MetricORM)
            .filter(MetricORM.claim_id.in_(to_uuid_list(work_claim_ids)))
            .all()
        )
        for m in metric_rows:
            metrics_by_claim[str(m.claim_id)].append(m)

    grouped = defaultdict(list)
    meta = {}

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
            "location": session.query(CompanyORM.location).filter(CompanyORM.company_name == org_name).scalar(),
            "highlights": highlights,
        }
        if start_date:
            entry["start_date"] = str(start_date)
            entry["end_date"] = str(end_date) if end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_project_entries(session, claim_ids, description_overrides=None):
    """依 claim_id 清單取得完整背景，篩出 EDUCATION TRACK（is_work_track=False）
    子集，依 project_name 分組，組成 rendercv ProjectEntry 清單。"""
    contexts = fetch_experience_contexts(session, claim_ids)
    description_overrides = description_overrides or {}

    project_claim_ids = [cid for cid in claim_ids if not contexts.get(cid, {}).get("is_work_track")]

    if not project_claim_ids:
        return []

    grouped = defaultdict(list)

    for claim_id in project_claim_ids:
        ctx = contexts.get(claim_id)
        if not ctx:
            continue

        tag_rows = (
                session.query(TagORM)
                .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
                .join(ClaimORM, ClaimORM.claim_id == ClaimTagORM.claim_id)
                .join(ActionORM, ActionORM.action_id == ClaimORM.action_id)
                .join(ProblemORM, ProblemORM.problem_id == ActionORM.problem_id)
                .join(ProjectORM, ProjectORM.project_id == ProblemORM.project_id)
                .filter(ProjectORM.project_name==ctx["project_name"])
                .filter(TagORM.tag_category == "tool")
                .all()
            )
        tools = ', '.join([tag.tag_name for i, tag in enumerate(tag_rows) if i < 3])
        
        key = (ctx["project_name"], ctx["start_date"], ctx["end_date"], ctx["project_context"], tools)
        highlight = description_overrides.get(claim_id, ctx["claim_description"])

        grouped[key].append(highlight)

    entries = []
    for (project_name, start_date, end_date, project_context, tools), highlights in grouped.items():
        entry = {
            "name": project_name,
            "tools": tools,
            "summary": project_context,
            "highlights": highlights,
        }
        if start_date:
            entry["start_date"] = str(start_date)
            entry["end_date"] = str(end_date) if end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_education_entries(session, relevant_degrees): 
    """把 education_selector.select_relevant_degrees() 選出的 DegreeORM 清單
    直接組成 education section，不再額外篩選。只放
    institution/area/degree/start_date/end_date。排序依 start_date 倒序。"""
    if not relevant_degrees:
        return []

    entries = []
    for degree in relevant_degrees:
        entry = {
            "institution": session.query(SchoolORM.school_name).filter(SchoolORM.school_id==degree.school_id).scalar(),
            "area": degree.degree_name,
            "location": session.query(SchoolORM.location).filter(SchoolORM.school_id==degree.school_id).scalar(),
        }
        if degree.degree_level:
            entry["degree"] = degree.degree_level
        if degree.start_date:
            entry["start_date"] = str(degree.start_date)
            entry["end_date"] = str(degree.end_date) if degree.end_date else "present"
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_skill_entries(session, claim_ids, job_description):
    """從關聯的標籤中取得 tag_name，交給 skill_grouper.group_skills_for_jd()
    依這份 JD 重新分組（上限4組，含未分類 tag 的互動式指派、以及新增分組的
    互動指令），輸出 rendercv OneLineEntry 清單。
    """
    if not claim_ids:
        return []

    tag_rows = (
        session.query(TagORM)
        .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .filter(TagORM.tag_category.in_(["tool", "capability"])) # tyahsin: need some re-thinking
        .all()
    )

    seen_tags = set()
    tags_with_category = []
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

def build_summary_entry(experience_entries, project_entries, education_entries, job_description):
    """組裝 summary section 所需素材，呼叫 summary_writer 進行 LLM 草擬 + 人工確認。

    (本輪變更) 拿掉原本的 domain/job_function 輔助訊號 -- tag_category 改版
    成 tool/capability/context 三軸後，舊訊號已無對應資料來源；改用
    skills_entries 頂替又會跟 skills section 內容重複曝光。權衡之下直接
    拿掉這個訊號：final_bullets + education_entries 已是履歷資訊量最大的
    兩塊，輔助訊號本來就是 non-forced 的錦上添花，拿掉不影響 summary 的
    核心品質，換來簽名更簡單、不必為了一個弱訊號多查一次 DB 或製造重複。
    """
    bullets = [
        h for entry in experience_entries for h in entry.get("highlights", [])
    ] + [
        h for entry in project_entries for h in entry.get("highlights", [])
    ]

    return write_summary(job_description, bullets, education_entries)

def build_cv_pdf_path(name: str, job_title: str, company_name: str) -> pathlib.Path:
    """(v17, replaces build_default_output_path) 依規則組出完整 PDF 輸出路徑：

    - OUTPUT_DIR 來自 core.config.get_cv_config()（讀 .env 的 OUTPUT_DIR，
      未設定則預設 ~/gencv_applications）。
    - folder: {OUTPUT_DIR}/{job_title}_{company_name}/
    - filename: cv_{name}_{job_title}_{company_name}.pdf
    - 回傳完整路徑: {OUTPUT_DIR}/{job_title}_{company_name}/cv_{name}_{job_title}_{company_name}.pdf

    這個函式只計算路徑字串，不建立資料夾 -- 資料夾的建立交給
    render_pdf()（呼叫時才 mkdir，若已存在則不動作）。
    name/job_title/company_name 皆做輕量清洗（去頭尾空白、空白轉底線），
    不是完整的 slug 函式，其他特殊字元不會被處理。
    """
    cv_config = get_cv_config()

    def _clean(value: str, fallback: str) -> str:
        return (value or fallback).strip().replace(" ", "_")

    safe_name = _clean(name, "UnknownName")
    safe_job_title = _clean(job_title, "UnknownRole")
    safe_company = _clean(company_name, "UnknownCompany")

    folder = cv_config.output_dir / f"{safe_job_title}_{safe_company}"
    filename = f"cv_{safe_name}_{safe_job_title}_{safe_company}.pdf"
    return folder / filename


def build_cv_yaml(
    company_name,
    job_title,
    job_description,
    claim_ids,
    output_path,
    designated_name=None,
    headline=None,
    description_overrides=None,
):
    """(v17, renamed from build_resume_yaml) 組出 rendercv 看得懂的 cv YAML
    （只含 cv: 區塊），寫入 output_path。

    job_description 為必填參數 -- build_skills_section() 需要它來做
    JD-specific 的 skill 分組（僅 tag_category == "skill" 的 tag）。

    summary 為選填的履歷開場摘要，寫進 sections["summary"]（單元素字串 list，
    RenderCV 的 TextEntry 寫法）。None 代表使用者選擇省略。
    """

    name, email, phone, location, visa_status = build_identity_entry(job_description, designated_name)
    networks_entries = build_network_entries(job_description)
    with get_session() as session:
        experience_entries = build_experience_entries(session, claim_ids, description_overrides)
        project_entries = build_project_entries(session, claim_ids, description_overrides)
        relevant_degrees = select_relevant_degrees(session, claim_ids, job_description)
        education_entries = build_education_entries(session, relevant_degrees)
        skills_entries = build_skill_entries(session, claim_ids, job_description)
    summary_entry = build_summary_entry(experience_entries, project_entries, education_entries, job_description)

    cv_block = {
        "name": name,
        "headline": f"{headline}" if visa_status is None else f"{headline} ({visa_status})",
        "email": email,
        "phone": phone,
        "location": location,
    }
    if networks_entries:
        cv_block["social_networks"] = networks_entries

    sections = {}
    sections["summary"] = [summary_entry]
    sections["skills"] = skills_entries
    sections["experience"] = experience_entries
    sections["education"] = education_entries
    if project_entries:
        sections["projects"] = project_entries

    cv_block["sections"] = sections

    cv_content = {"cv": cv_block}

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cv_content, f, allow_unicode=True, sort_keys=False)

    print(
        f"Generated {output_path} with {len(experience_entries)} experience entries, "
        f"{len(project_entries)} project entries, and {len(education_entries)} education entries."
    )

    return build_cv_pdf_path(name, job_title, company_name)


def render_pdf(
    cv_yaml_path: pathlib.Path,
    pdf_path: pathlib.Path,
    design_path: pathlib.Path | None = None,
    locale_path: pathlib.Path | None = None,
    settings_path: pathlib.Path | None = None,
):
    """呼叫 `rendercv render` CLI，把 cv YAML 排版編譯成 PDF。

    (v17) pdf_path 現在是必填參數 -- 不再從 settings.yaml 讀取任何
    pdf_path 值來當備援，PDF 輸出路徑一律由呼叫端算好傳入（通常搭配
    build_cv_pdf_path()），並透過 `--pdf-path` CLI 參數告訴 rendercv。
    `--settings` 仍會照常傳給 rendercv CLI -- settings.yaml 裡其他
    render 設定（非 pdf_path 的部分）不受影響，只是不再從裡面讀值。

    會自動建立 pdf_path 所在資料夾（若不存在）。
    """
    print(f"[INFO] Using templates from {TEMPLATES_DIR.resolve()}")
    design_path = design_path or TEMPLATES_DIR / "design.yaml"
    locale_path = locale_path or TEMPLATES_DIR / "locale.yaml"
    settings_path = settings_path or TEMPLATES_DIR / "settings.yaml"

    pdf_path = pathlib.Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # save .yaml for reference 
    yaml_data = {}
    yaml_path = pdf_path.parent / pdf_path.split("/")[-1].replace("pdf", "yaml")

    for file in [cv_yaml_path, design_path, locale_path, settings_path]:
        if not file.exists():
            print(f"[ERROR] 找不到檔案: {f}", file=sys.stderr)
            sys.exit(1)
        with open(file, "r", encoding='utf-8') as f:
            data = yaml.safe_load(file)
            if data:
                yaml_data[file.split('/')[-1].split('.')[0]] = data
    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False)

    cmd = [
        "rendercv", "render", str(cv_yaml_path),
        "--settings", str(settings_path),
        "--design", str(design_path),
        "--locale-catalog", str(locale_path),
        "--pdf-path", str(pdf_path),
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

    if result.stdout:
        print(result.stdout)

    return pdf_path
