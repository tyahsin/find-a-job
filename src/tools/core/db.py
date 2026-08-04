"""
core/db.py
SQLAlchemy engine/session setup + ORM models for the Source of Truth schema:
company, position, school, degree, project (Step 0-4 of add-project),
plus problem, action, claim (result_claim), evidence, tag, claim_tag
(Step 0-5 of add-claim).

Also provides a generic `select_or_create()` helper used by the onboard
CLI flow to either reuse an existing row or create a new one, plus typed
insert helpers for each confirmed table.

Fuzzy matching (via RapidFuzz) is used when reusing existing company/school
rows, since names may be typed slightly differently across sessions
(e.g. "Google" vs "Google Inc.").

NOTE (Aug 2026): the education-track mid-level entity is called `degree`
in the actual Postgres schema (table `degree`, column `degree_id`,
`degree_name`), NOT `education`/`education_id`.

NOTE (Aug 2026): expire_on_commit=False is set on the sessionmaker to avoid
DetachedInstanceError -- our CLI flows read attributes off ORM objects
(e.g. project.project_id) after the `with get_session()` block has exited
and committed/closed the session. Without this flag, SQLAlchemy marks all
attributes as "expired" on commit, so any read after the session closes
triggers an attribute refresh against a now-closed session, which raises
DetachedInstanceError. Since this app is single-user, single-threaded CLI
with no concurrent access to these objects, disabling commit-time expiry is
safe and avoids the error entirely.

governance_metadata is intentionally NOT modeled here yet -- add it once
its schema is finalized.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Type, TypeVar

from rapidfuzz import fuzz, process
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import get_db_config

# ---------------------------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------------------------

_db_config = get_db_config()
engine = create_engine(_db_config.sqlalchemy_url, future=True)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,  # avoid DetachedInstanceError on post-commit attribute reads
)


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Yields a session wrapped in a transaction. Commits on success,
    rolls back on any exception -- used to guarantee that multi-table
    writes (e.g. add-project's company/school -> position/degree -> project,
    or add-claim's problem -> action -> claim -> evidence -> tag) are
    all-or-nothing.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Column:
    return Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# ---------------------------------------------------------------------------
# 公司層 / 學校層
# ---------------------------------------------------------------------------

class CompanyORM(Base):
    __tablename__ = "company"

    company_id = _uuid_pk()
    company_name = Column(Text)
    location = Column(Text)


class SchoolORM(Base):
    __tablename__ = "school"

    school_id = _uuid_pk()
    school_name = Column(Text)
    location = Column(Text)


# ---------------------------------------------------------------------------
# 職位層 (work track)
# ---------------------------------------------------------------------------

class PositionORM(Base):
    __tablename__ = "position"

    position_id = _uuid_pk()
    company_id = Column(PGUUID(as_uuid=True), ForeignKey("company.company_id"))
    title = Column(Text)
    department = Column(Text)
    responsibility = Column(Text)
    responsibility_source = Column(Text, nullable=True)
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)


# ---------------------------------------------------------------------------
# 學業層 (degree -- work-equivalent track for education, per latest schema)
# ---------------------------------------------------------------------------

class DegreeORM(Base):
    __tablename__ = "degree"

    degree_id = _uuid_pk()
    school_id = Column(PGUUID(as_uuid=True), ForeignKey("school.school_id"))
    degree_name = Column(Text)  # e.g. MA, MS, PhD, Short Program, etc.
    subject = Column(Text)      # e.g. Major, Minor, area of study
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)


# ---------------------------------------------------------------------------
# 專案層 (shared, position_id XOR degree_id)
# ---------------------------------------------------------------------------

class ProjectORM(Base):
    __tablename__ = "project"
    __table_args__ = (
        CheckConstraint(
            "(position_id IS NOT NULL AND degree_id IS NULL) OR "
            "(position_id IS NULL AND degree_id IS NOT NULL)",
            name="project_exactly_one_parent",
        ),
    )

    project_id = _uuid_pk()
    position_id = Column(PGUUID(as_uuid=True), ForeignKey("position.position_id"), nullable=True)
    degree_id = Column(PGUUID(as_uuid=True), ForeignKey("degree.degree_id"), nullable=True)
    project_name = Column(Text)
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)
    project_context = Column(Text)


# ---------------------------------------------------------------------------
# 問題層 (Problem) -- belongs to a project
# ---------------------------------------------------------------------------

class ProblemORM(Base):
    __tablename__ = "problem"

    problem_id = _uuid_pk()
    project_id = Column(PGUUID(as_uuid=True), ForeignKey("project.project_id"))
    description = Column(Text)
    stakeholder_impact = Column(Text, nullable=True)
    baseline_metric = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# 行動層 (Action) -- belongs to a problem
# ---------------------------------------------------------------------------

class ActionORM(Base):
    __tablename__ = "action"

    action_id = _uuid_pk()
    problem_id = Column(PGUUID(as_uuid=True), ForeignKey("problem.problem_id"))
    description = Column(Text)
    role_scope = Column(Text, nullable=True)  # e.g. Led / Collaborated / Supported


# ---------------------------------------------------------------------------
# 結果宣告層 (ResultClaim) -- belongs to an action
# ---------------------------------------------------------------------------

class ClaimORM(Base):
    __tablename__ = "claim"

    claim_id = _uuid_pk()
    action_id = Column(PGUUID(as_uuid=True), ForeignKey("action.action_id"))
    description = Column(Text)  # CV bullet 的核心敘述 (formerly result_statement)
    # NOTE (Aug 2026): value_type / metric / metric_confidence moved OUT of this
    # table into claim_metric (one-to-many). A single claim was found in
    # practice to often carry more than one quantifiable metric, and not
    # every metric cleanly fits the fixed value_type categories -- see
    # ClaimMetricORM below.


class ClaimMetricORM(Base):
    """
    One-to-many: a single claim can carry multiple quantifiable metrics
    (e.g. "processing time dropped from 5 days to 2 days" AND "customer
    satisfaction rose ~15%" in the same description text).

    Rows here are LLM-extracted DRAFTS from the claim description text, always
    reviewed/edited by the user before being written -- metric_confidence
    in particular is a human judgment call, never taken from the LLM's own
    self-reported confidence (LLM self-confidence is known to be poorly
    calibrated).
    """
    __tablename__ = "claim_metric"

    metric_id = _uuid_pk()
    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"))
    metric_type = Column(Text)        # 效率提升 / 成本節省 / 品質改善 / 其他
    metric_label = Column(Text)       # 這個數字代表什麼，例如「處理時間」「客戶滿意度」
    metric_value = Column(Text)       # 數值本身，含單位/描述，例如「從5天降到2天」
    metric_confidence = Column(Text)  # 精確 / 估算 / 大致印象 -- 一律由使用者手動選定


# ---------------------------------------------------------------------------
# 證據層 (Evidence) -- belongs to a claim
# ---------------------------------------------------------------------------

class EvidenceORM(Base):
    __tablename__ = "evidence"

    evidence_id = _uuid_pk()
    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"))
    evidence_type = Column(Text)
    evidence_location = Column(Text, nullable=True)
    evidence_level = Column(Text)  # L1 (最強) -> L4 (最弱), auto-derived from evidence_type


# ---------------------------------------------------------------------------
# 標籤層 (Tag) -- LLM-generated, human-confirmed; many-to-many with claim
# ---------------------------------------------------------------------------

class TagORM(Base):
    __tablename__ = "tag"

    tag_id = _uuid_pk()
    tag_name = Column(Text, unique=True, nullable=False)
    tag_category = Column(Text, nullable=False)  # skill | domain | job_function


class ClaimTagORM(Base):
    __tablename__ = "claim_tag"

    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"), primary_key=True)
    tag_id = Column(PGUUID(as_uuid=True), ForeignKey("tag.tag_id"), primary_key=True)


# ---------------------------------------------------------------------------
# Generic select-or-create helper (exact match)
# ---------------------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=Base)


def select_or_create(
    session: Session,
    model: Type[ModelT],
    match_filters: dict,
    create_values: Optional[dict] = None,
) -> ModelT:
    """
    Exact-match lookup. If found, return it; otherwise create a new row.
    Use this for fields where exact equality is the right semantics
    (e.g. matching an existing position_id you already selected, or
    reusing an existing tag by exact tag_name).
    """
    query = session.query(model)
    for field, value in match_filters.items():
        query = query.filter(getattr(model, field) == value)
    existing = query.first()
    if existing is not None:
        return existing

    values = create_values if create_values is not None else match_filters
    new_obj = model(**values)
    session.add(new_obj)
    session.flush()
    return new_obj


def list_rows(session: Session, model: Type[ModelT], **filters) -> Sequence[ModelT]:
    """Generic query helper -- used e.g. to list existing positions under a company."""
    query = session.query(model)
    for field, value in filters.items():
        query = query.filter(getattr(model, field) == value)
    return query.all()


# ---------------------------------------------------------------------------
# Fuzzy select-or-create (for company_name / school_name style fields)
# ---------------------------------------------------------------------------

FUZZY_MATCH_THRESHOLD = 90  # 0-100 similarity score; tune based on real usage


def fuzzy_find_candidates(
    session: Session,
    model: Type[ModelT],
    name_field: str,
    query_name: str,
    limit: int = 5,
) -> list[tuple[ModelT, float]]:
    """
    Returns up to `limit` existing rows whose `name_field` is similar to
    `query_name`, sorted by similarity score (highest first). Used to show
    the user a disambiguation prompt like:

        Did you mean one of these existing companies?
        1. Google Inc. (94% match)
        2. Google LLC (88% match)
        [n] None of these, create new

    This does NOT auto-create or auto-select -- the caller (CLI layer)
    decides what to do with the candidates, keeping the "no silent
    overwrite" governance principle intact.
    """
    all_rows = session.query(model).all()
    if not all_rows:
        return []

    choices = {row: getattr(row, name_field) or "" for row in all_rows}
    matches = process.extract(
        query_name,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )
    # rapidfuzz's process.extract with a dict returns (choice_value, score, key)
    results = [(row, score) for _value, score, row in matches]
    return [(row, score) for row, score in results if score >= FUZZY_MATCH_THRESHOLD]


def create_row(session: Session, model: Type[ModelT], values: dict) -> ModelT:
    """Straight insert, no matching -- used once the CLI layer has confirmed
    the user really wants a brand new row (no fuzzy match accepted)."""
    new_obj = model(**values)
    session.add(new_obj)
    session.flush()
    return new_obj
