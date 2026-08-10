"""
resume_builder.py
把選定的經歷(claim)從資料庫組裝成履歷內容 YAML,交給 rendercv 官方套件(pip 安裝、
CLI 呼叫)排版編譯成最終 PDF。

輸出格式已對照 RenderCV 官方 YAML schema 逐欄位核對:
  ExperienceEntry: company / position / start_date / end_date / location / summary / highlights
  OneLineEntry:    label / details
start_date 缺值時不寫入該 key(而非顯式寫 null);end_date 缺值時一律填 "present"。

design/locale/settings 三份凍結設定檔命名為 design.yaml / locale.yaml / settings.yaml,
放在 gencv/templates/ 底下(可從 test.yaml 拆出對應區塊)。

前置需求:
    pip install rendercv
    templates/design.yaml / locale.yaml / settings.yaml 三份凍結設定檔。

實作備註(2026-08-10):本檔案是內部模組,不再提供 __main__ 命令列入口 -- 對外唯一
執行入口是 gencv/cli.py(python -m gencv.cli generate ...)。
"""

import pathlib
import subprocess
import sys
from collections import defaultdict

import yaml

from core.db import get_session, ClaimMetricORM, ClaimTagORM, TagORM
from gencv.experience_matcher import fetch_experience_contexts, to_uuid_list

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# 資料組裝:資料庫 -> rendercv 的 cv YAML 內容
# ---------------------------------------------------------------------------

def format_highlight(claim_description: str, metrics: list) -> str:
    """把一筆經歷的描述 + 其數據指標組成一條 highlight 字串。
    數字用 **bold** 標記 -- rendercv 原生支援 highlight 內的 Markdown 語法。
    claim_description 可能是原句,也可能是 experience_rewriter.py 產出、
    使用者已人工確認過的改寫版,metric 數字一律機械式附加,不受改寫影響。"""
    text_out = claim_description
    metric_frags = [
        f"**{m.metric_value}** ({m.metric_label})" for m in metrics if m.metric_value
    ]
    if metric_frags:
        text_out = f"{text_out} — " + ", ".join(metric_frags)
    return text_out


def build_experience_entries(
    session, claim_ids: list[str], description_overrides: dict[str, str] | None = None
) -> list[dict]:
    """依 claim_id 清單取得完整背景,依組織+職稱分組,組成 rendercv ExperienceEntry 清單。"""
    contexts = fetch_experience_contexts(session, claim_ids)
    description_overrides = description_overrides or {}

    metrics_by_claim: dict[str, list] = defaultdict(list)
    if claim_ids:
        metric_rows = (
            session.query(ClaimMetricORM)
            .filter(ClaimMetricORM.claim_id.in_(to_uuid_list(claim_ids)))
            .all()
        )
        for m in metric_rows:
            metrics_by_claim[str(m.claim_id)].append(m)

    grouped: dict[tuple, list[str]] = defaultdict(list)
    meta: dict[tuple, dict] = {}

    for claim_id in claim_ids:
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        key = (ctx["org_name"], ctx["title"], ctx["start_date"], ctx["end_date"])
        description_text = description_overrides.get(claim_id, ctx["claim_description"])
        highlight = format_highlight(description_text, metrics_by_claim.get(claim_id, []))
        grouped[key].append(highlight)
        meta[key] = ctx

    entries = []
    for (org_name, title, start_date, end_date), highlights in grouped.items():
        ctx = meta[(org_name, title, start_date, end_date)]
        entry = {
            "company": org_name,
            "position": title,
            "highlights": highlights,
        }
        if start_date:
            entry["start_date"] = str(start_date)
        entry["end_date"] = str(end_date) if end_date else "present"
        if ctx.get("project_context"):
            entry["summary"] = ctx["project_context"]
        entries.append(entry)

    entries.sort(key=lambda e: e.get("start_date", ""), reverse=True)
    return entries


def build_skills_section(session, claim_ids: list[str]) -> list[dict]:
    """從關聯的標籤中,依 skill/domain/job_function 三類分組,輸出 rendercv OneLineEntry 清單。"""
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


def build_resume_yaml(
    claim_ids: list[str], headline: str, name: str, output_path: str,
    description_overrides: dict[str, str] | None = None,
) -> str:
    """組出 rendercv 看得懂的 cv YAML(只含 cv: 區塊),寫入 output_path。"""
    with get_session() as session:
        experience_entries = build_experience_entries(session, claim_ids, description_overrides)
        skills_entries = build_skills_section(session, claim_ids)

    resume_content = {
        "cv": {
            "name": name,
            "headline": headline,
            "sections": {
                "experience": experience_entries,
                "skills": skills_entries,
            },
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resume_content, f, allow_unicode=True, sort_keys=False)

    print(f"Generated {output_path} with {len(experience_entries)} experience entries.")
    return output_path


# ---------------------------------------------------------------------------
# 排版編譯:呼叫已安裝的 rendercv CLI
# ---------------------------------------------------------------------------

def render_pdf(
    cv_yaml_path: pathlib.Path,
    pdf_out_path: pathlib.Path,
    design_path: pathlib.Path | None = None,
    locale_path: pathlib.Path | None = None,
    settings_path: pathlib.Path | None = None,
) -> None:
    """呼叫 `rendercv render` CLI,把 cv YAML 排版編譯成 PDF。"""
    design_path = design_path or TEMPLATES_DIR / "design.yaml"
    locale_path = locale_path or TEMPLATES_DIR / "locale.yaml"
    settings_path = settings_path or TEMPLATES_DIR / "settings.yaml"

    for f in [cv_yaml_path, design_path, locale_path, settings_path]:
        if not f.exists():
            print(f"[ERROR] 找不到檔案: {f}", file=sys.stderr)
            sys.exit(1)

    pdf_out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rendercv", "render", str(cv_yaml_path),
        "--design", str(design_path),
        "--locale-catalog", str(locale_path),
        "--settings", str(settings_path),
        "--pdf-path", str(pdf_out_path),
        "--dont-generate-markdown",
        "--dont-generate-html",
        "--dont-generate-png",
    ]

    print("[INFO] Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] rendercv 執行失敗:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[OK] PDF 已產出: {pdf_out_path}")
    if result.stdout:
        print(result.stdout)
