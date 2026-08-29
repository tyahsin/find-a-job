"""
gencv/resume_builder.py
"""

import pathlib
import subprocess
import sys
import yaml
from datetime import date
import tempfile

from core.config import get_cv_config
from core.db import (
    get_session, 
    DegreeORM, ClaimTagORM, TagORM, ClaimORM, ActionORM, ProblemORM, ProjectORM, CompanyORM, SchoolORM, PositionORM, JobORM, IdentityORM, NetworkORM
)
from gencv.claim_selector import select_claims
from gencv.claim_ranker import narrow_all_claims
from gencv.claim_rewriter import rewrite_and_confirm
# from gencv.network_selector import select_networks_from_jd
from gencv.orphan_selector import judge_orphan_positions, judge_orphan_degrees
from gencv.skill_grouper import group_skills_for_jd
from gencv.summary_writer import write_summary


TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"


def build_identity_entry(job_location, designated_name=None):
    with get_session() as session:
        identity = (
            session.query(IdentityORM).
            filter(IdentityORM.location.like(f"%{job_location.split(',')[-1]}")). # same country
            one_or_none()
        )
        if not identity:
            identity = (
                session.query(IdentityORM).
                filter(IdentityORM.is_default == True). # default identity
                one_or_none()
            )
    name = identity.name if designated_name is None else designated_name
    visa_status = identity.visa_status if identity.visa_status and 'citizenship' not in str.lower(identity.visa_status) else None
    return name, identity.email, identity.phone, identity.location, visa_status


def build_network_entries(job_description):
    # networks = select_networks_from_jd(job_description)
    # return [
    #     {"network": n.network, "username": n.username}
    #     for n in networks
    # ]
    with get_session() as session:
        network = session.query(NetworkORM).filter(NetworkORM.network=='LinkedIn').one_or_none()
        return [{"network": network.network, "username": network.username}]


def build_experience_entries(items, job_description):
    entries = []

    def _add_entry(company, position):
        entries.append({
            "company": company.company_name,
            "position": position.title,
            "location": company.location,
            "start_date": str(position.start_date),
            "end_date": str(position.end_date) if position.end_date else "present"
        })

    def _positions_overlap(pos_a, pos_b):
        end_a = pos_a.end_date or date.max
        end_b = pos_b.end_date or date.max
        return pos_a.start_date <= end_b and pos_b.start_date <= end_a        

    # fetch all position information joined by company information
    with get_session() as session:
        orms = (
            session.query(CompanyORM, PositionORM).
            join(PositionORM, PositionORM.company_id==CompanyORM.company_id).
            order_by(PositionORM.start_date.desc()).
            all()
        )

    # group items to highlights
    highlighted_positions, orphan_position_candidates = [], {}
    for company, position in orms:
        highlights = [
            item["description"] for item in items if item["group_key"] == position.position_id
        ]
        if highlights:
            _add_entry(company, position)
            entries[-1]["highlights"] = highlights
            highlighted_positions.append(position)
        else:
            orphan_position_candidates[position] = company

    # fetch orphan position
    orphan_positions = {
        position: company for position, company in orphan_position_candidates.items() if not any(_positions_overlap(position, hp) for hp in highlighted_positions) 
    }

    # judge orphan positions
    judged_positions = judge_orphan_positions(list(orphan_positions.keys()), job_description)
    if judged_positions:
        for judged_position in judged_positions:
            _add_entry(orphan_positions[judged_position], judged_position)
            entries[-1]["summary"] = judged_position.responsibility
        
    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def _match_tags(project_ids, tag_category):
    with get_session() as session:
        return (
            session.query(TagORM)
            .join(ClaimTagORM, ClaimTagORM.tag_id == TagORM.tag_id)
            .join(ClaimORM, ClaimORM.claim_id == ClaimTagORM.claim_id)
            .join(ActionORM, ActionORM.action_id == ClaimORM.action_id)
            .join(ProblemORM, ProblemORM.problem_id == ActionORM.problem_id)
            .join(ProjectORM, ProjectORM.project_id == ProblemORM.project_id)
            .filter(ProjectORM.project_id.in_(project_ids))
            .filter(TagORM.tag_category.in_(tag_category))
            .all()
        )


def build_project_entries(items):
    entries = []

    # fetch all project information and tag information
    with get_session() as session:
        orms = session.query(ProjectORM).filter(ProjectORM.degree_id.isnot(None)).all()
    
    # group items to highlights
    for project in orms:
        highlights = [
            item["description"] for item in items if item["group_key"] == project.project_id
        ]
        if highlights:
            tags = _match_tags([project.project_id], ['tool'])
            tools = ', '.join([tag.tag_name for i, tag in enumerate(tags) if i < 3]) # how to make sure top-K tags are what I want
            entries.append({
                "name": project.project_name,
                "tools": tools,
                "summary": project.project_context,
                "highlights": highlights,
                "start_date": str(project.start_date),
                "end_date": str(project.end_date) if project.end_date else "present"
            })

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_education_entries(items, job_description): 
    entries = []

    def _add_entry(school, degree):
        highlights = degree.degree_honors + degree.degree_awards # haven't done thesis and others but better do that
        entries.append({
            "institution": school.school_name,
            "area": degree.degree_name,
            "location": school.location,
            "degree": degree.degree_level,
            "highlights": highlights, # honors and awards
            "start_date": str(degree.start_date),
            "end_date": str(degree.end_date) if degree.end_date else "present"
        })

    # fetch all degree information joined by school information
    with get_session() as session:
        orms = (
            session.query(SchoolORM, DegreeORM, ProjectORM.project_id)
            .join(SchoolORM, SchoolORM.school_id == DegreeORM.school_id)
            .join(ProjectORM, ProjectORM.degree_id == DegreeORM.degree_id)  # academic project related degree only
            .all()
        )

    # channel degrees into chosen and not
    chosen_projects = set([item["group_key"] for item in items if item["track"] == "degree"]) # project_id
    chosen_degrees, orphan_degrees = [], {}
    for school, degree, project_id in orms:
        if project_id in chosen_projects:
            if degree.degree_id not in chosen_degrees:
                _add_entry(school, degree)
                chosen_degrees.append(degree.degree_id)
                orphan_degrees.pop(degree, None)  # to remove degree wrongly identified as orphan but actually has project chosen in the later orm records
        else:
            if degree.degree_id not in chosen_degrees and degree not in orphan_degrees:
                orphan_degrees[degree] = school

    # judge orphan degrees
    judged_degrees = judge_orphan_degrees(list(orphan_degrees.keys()), job_description)
    if judged_degrees:
        for judged_degree in judged_degrees:
            _add_entry(orphan_degrees[judged_degree], judged_degree)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_skill_entries(items, job_description):

    # find all project information joined by position information
    project_ids = []
    with get_session() as session:
        orms = (
            session.query(ProjectORM.position_id, ProjectORM.project_id).
            filter(ProjectORM.position_id.isnot(None)).
            all()
        )
    group_key_map = {orm.position_id: orm.project_id for orm in orms}
    for item in items:
        if item["track"] == "degree": # group_key is project_id
            project_ids.append(item["group_key"])
        else: # group_key is position_id
            if item["group_key"] in group_key_map and item["group_key"] not in project_ids: # 1 position may have N projects
                project_ids.append(group_key_map[item["group_key"]])

    tag_rows = _match_tags(project_ids, ['tool', 'capability'])

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
    bullets = [
        h for entry in experience_entries for h in entry.get("highlights", [])
    ] + [
        h for entry in project_entries for h in entry.get("highlights", [])
    ]

    return [write_summary(job_description, bullets, education_entries)]


def build_cv_blocks(
    name,
    headline,
    info,
    networks_entries, 
    summary_entry,
    # skills_entries, 
    experience_entries, 
    education_entries, 
    project_entries, 
    cv_yaml_path,
):
    email, phone, location, visa_status = info
    
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
    sections["summary"] = summary_entry
    # sections["skills"] = skills_entries
    sections["experience"] = experience_entries
    sections["education"] = education_entries
    if project_entries:
        sections["projects"] = project_entries

    cv_block["sections"] = sections

    cv_content = {"cv": cv_block}

    with open(cv_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cv_content, f, allow_unicode=True, sort_keys=False)

    print(
        f"Generated {cv_yaml_path} with {len(experience_entries)} experience entries, "
        f"{len(project_entries)} project entries, and {len(education_entries)} education entries."
    )


def build_cv_pdf_path(name: str, job_title: str, company_name: str) -> pathlib.Path:
    cv_config = get_cv_config()

    def _clean(value: str, fallback: str) -> str:
        return (value or fallback).strip().replace(" ", "_")

    safe_name = _clean(name, "UnknownName")
    safe_job_title = _clean(job_title, "UnknownRole")
    safe_company = _clean(company_name, "UnknownCompany")

    folder = cv_config.output_dir / f"{safe_job_title}_{safe_company}"
    filename = f"cv_{safe_name}_{safe_job_title}_{safe_company}.pdf"
    return pathlib.Path(folder / filename)


def render_pdf(
    pdf_path: pathlib.Path,
    cv_yaml_path: pathlib.Path | None = None,
    design_path: pathlib.Path | None = None,
    locale_path: pathlib.Path | None = None,
    settings_path: pathlib.Path | None = None,
):
    print(f"[INFO] Using templates from {TEMPLATES_DIR.resolve()}")
    cv_yaml_path = cv_yaml_path or pdf_path.parent / "cv.yaml"
    design_path = design_path or TEMPLATES_DIR / "design.yaml"
    locale_path = locale_path or TEMPLATES_DIR / "locale.yaml"
    settings_path = settings_path or TEMPLATES_DIR / "settings.yaml"

    for file in [cv_yaml_path, design_path, locale_path, settings_path]:
        print(file)
        if not file.exists():
            print(f"[ERROR] File not found: {file}", file=sys.stderr)
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        typst_path = pathlib.Path(tmp_dir) / f"{pdf_path.stem}.typ"
        cmd = [
            "rendercv", "render", str(cv_yaml_path),
            "--settings", str(settings_path),
            "--design", str(design_path),
            "--locale-catalog", str(locale_path),
            "--pdf-path", str(pdf_path),
            "--typst-path", str(typst_path),
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


def generate_cv(
    job: JobORM,
    designated_name: str | None = None,
    designated_headline: str | None = None,
) -> pathlib.Path:
    
    # decalre JobORM related variables
    company_name = job.company_name
    job_title = job.job_title
    headline = job_title if designated_headline is None else designated_headline
    job_location = job.job_location
    job_description = job.job_description

    # build identity & basic information
    name, *info = build_identity_entry(job_location, designated_name)
    pdf_path = build_cv_pdf_path(name, job_title, company_name)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = pdf_path.parent / "claim_ranker.log"
    cv_yaml_path = pdf_path.parent / "cv.yaml"

    # build networks
    networks_entries = build_network_entries(job_description)

    # select claims
    selected = select_claims(job_description)
    if not selected["work_track"] and not selected["academic_track"]:
        raise ValueError("篩選後沒有任何經歷通過（可能被治理層擋下，或 LLM 判定皆不相關）。")

    # rank claims
    narrowed, contexts = narrow_all_claims(selected, job_description, log_path)
    if not narrowed:
        raise ValueError("MECE narrowing 後沒有任何 item 留下，無法產生履歷內容。")

    # rewrite claims
    rewritten = rewrite_and_confirm(narrowed, contexts, job_description)

    # build others
    experience_entries = build_experience_entries(rewritten, job_description)
    project_entries = build_project_entries(rewritten)
    education_entries = build_education_entries(rewritten, job_description)
    # skills_entries = build_skill_entries(rewritten, job_description)
    summary_entry = build_summary_entry(experience_entries, project_entries, education_entries, job_description)

    # build cv blocks
    build_cv_blocks(
        name,
        headline,
        info,
        networks_entries, 
        summary_entry,
        # skills_entries, 
        experience_entries, 
        education_entries, 
        project_entries, 
        cv_yaml_path,
    )

    # render cv
    render_pdf(pdf_path=pdf_path)

    return pdf_path