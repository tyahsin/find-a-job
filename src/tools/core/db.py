"""
core/db.py
Version: v1.4 (2026-08-19)

Changelog:
- v1.4 (2026-08-19): Removed GeneratedOutputORM (generated_outputs table).
  Rationale (per user decision, 2026-08-19): that table assumed "every
  pipeline generation is worth recording", but actual usage is "a
  generation may be unsatisfactory, get hand-edited as a yaml, and
  re-rendered -- only the version actually submitted matters". Replaced
  with JobORM (job) and ApplicationORM (application), corresponding to
  003_apply_tracking additions in 001_init_schema.sql:
  - JobORM: one row per job posting (company_name/job_title/job_location/
    job_description/source_url/date_posted/contact_name/contact_email).
    No governance concept -- every row is a candidate, matching the same
    pattern as CompanyORM/SchoolORM above.
  - ApplicationORM: one row per submitted application, referencing job_id
    (company_name/job_title read through this FK, not duplicated).
    file_path stores the path to whichever CV version was actually used
    (which may be hand-edited after generate_cv() produced a first
    draft -- pipeline.py itself no longer writes anything here
    automatically; only a deliberate "record this application" action
    does). current_stage is a fixed CHECK vocabulary (mirrors the
    existing role_scope/metric_type pattern in this codebase).
    interview_round is a plain INTEGER counter, incremented by CLI-layer
    logic (apply/application_log.py) when current_stage transitions to
    'Interviewing' -- NOT a database trigger, consistent with this
    project's "business logic stays in Python, not hidden in the
    database" principle (see fill_pending_support_levels() below for
    the same philosophy applied to governance).
- v1.3 (2026-08-10): Removed the global Session-level after_commit event
  listener introduced in v1.2. That mechanism fired on EVERY session
  commit across the whole app (add-project, add-company, add-position,
  add-degree, the backfill script, etc.), which is architecturally wider
  than warranted -- governance is conceptually tied ONLY to claims, so
  only add-claim should ever trigger a support_level fill. Replaced with
  fill_pending_support_levels(), a plain public function with no event
  registration. onboard/cli.py's add_claim() now calls this explicitly,
  once, right after its own `with get_session()` block finishes (i.e.
  after all claim-related steps -- problem/action/claim/metrics/evidence/
  tags/governance -- have been written and committed). No other command
  calls it, so add-project/add-company/etc. never trigger this check at
  all, matching the "governance only exists for claims" mental model.
  compute_support_level() itself is unchanged -- only the trigger
  mechanism moved from implicit (global event) to explicit (direct call
  from the one flow that actually needs it).
- v1.2 (2026-08-10): [SUPERSEDED by v1.3] Added an automatic after_commit
  hook that fired on every session commit app-wide.
- v1.1 (2026-08-10): Added compute_support_level() -- the rule-based
  aggregator for governance.support_level, combining evidence_level
  (multiple rows, OR logic -- take the STRONGEST) and metric_confidence
  (multiple rows, AND logic -- take the WEAKEST), then taking the WEAKER
  of those two combined results.
- v1.0 (2026-08-04, inferred): Initial engine/session setup + ORM models.

SQLAlchemy engine/session setup + ORM models for the Source of Truth schema:
company, position, school, degree, project (Step 0-4 of add-project),
plus problem, action, claim (result_claim), claim_metric, evidence, tag,
claim_tag, governance, claim_chunks (Step 0-6 of add-claim), plus job and
application (apply/ module, Step 5 -- see v1.4 changelog above).

Also provides a generic `select_or_create()` helper used by the onboard
CLI flow to either reuse an existing row or create a new one, typed insert
helpers for each confirmed table, `compute_support_level()` for governance
aggregation, and `fill_pending_support_levels()` -- a function callers can
invoke explicitly once they know claim-related writes have committed (see
onboard/cli.py's add_claim(), the only caller).

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
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Type, TypeVar

from rapidfuzz import fuzz, process
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.sql import func

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
    or add-claim's problem -> action -> claim -> evidence -> tag -> governance)
    are all-or-nothing.

    NOTE (v1.3): this function does NOT automatically fill in
    governance.support_level on commit. That is the caller's
    responsibility -- see fill_pending_support_levels() below, which
    onboard/cli.py's add_claim() calls explicitly right after this
    context manager exits. Other flows (add-project, add-company, etc.)
    never call it, since they never touch governance rows.
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
# Identity layer / Network layer
# ---------------------------------------------------------------------------

class IdentityORM(Base):
    __tablename__ = "identity"
    __table_args__ = (
        Index(
            "identity_only_one_default",
            "is_default",
            unique=True,
            postgresql_where=text("is_default = true"),
        ),
    )

    identity_id = _uuid_pk()
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    phone = Column(Text, nullable=True)
    location = Column(Text, nullable=True)  # current residence, e.g. "Tokyo, Japan"
    visa_status = Column(Text, nullable=True)  # e.g. "Requires sponsorship"
    is_default = Column(Boolean, nullable=False, default=False)

class NetworkORM(Base):
    __tablename__ = "network"

    network_id = _uuid_pk()
    network = Column(Text, nullable=False)  # e.g. "GitHub", "LinkedIn", "Behance"
    username = Column(Text, nullable=False)  # account handle, e.g. "yahsintsai"
    target_position = Column(ARRAY(Text), nullable=False)  # e.g. ["Software Engineer", "Data Scientist"]

# ---------------------------------------------------------------------------
# Company layer / School layer
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
# Position layer (work track)
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
# Degree layer (education track)
# ---------------------------------------------------------------------------

class DegreeORM(Base):
    __tablename__ = "degree"

    degree_id = _uuid_pk()
    school_id = Column(PGUUID(as_uuid=True), ForeignKey("school.school_id"))
    degree_level = Column(Text)  # e.g. MA, MS, PhD, Short Program, etc.
    degree_type = Column(Text)  # e.g. Major, Minor, Concentration, etc.
    degree_name = Column(Text)  # the subject/area itself, e.g. "Computer Science"
    degree_description = Column(Text, nullable=True)
    degree_description_source = Column(Text, nullable=True)  # NULL => LLM-generated, no source doc
    degree_honors = Column(ARRAY(Text), nullable=False) 
    degree_awards = Column(ARRAY(Text), nullable=False) 
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)

# ---------------------------------------------------------------------------
# Project layer (shared, position_id XOR degree_id)
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
    project_context_source = Column(Text, nullable=True)

# ---------------------------------------------------------------------------
# Problem layer -- belongs to a project
# ---------------------------------------------------------------------------

class ProblemORM(Base):
    __tablename__ = "problem"

    problem_id = _uuid_pk()
    project_id = Column(PGUUID(as_uuid=True), ForeignKey("project.project_id"))
    problem_description = Column(Text)
    stakeholder_impact = Column(Text, nullable=True)
    baseline_metric = Column(Text, nullable=True)

# ---------------------------------------------------------------------------
# Action layer -- belongs to a problem
# ---------------------------------------------------------------------------

class ActionORM(Base):
    __tablename__ = "action"

    action_id = _uuid_pk()
    problem_id = Column(PGUUID(as_uuid=True), ForeignKey("problem.problem_id"))
    action_description = Column(Text)
    role_scope = Column(Text, nullable=True)  # e.g. Led / Collaborated / Supported

# ---------------------------------------------------------------------------
# Claim layer -- belongs to an action
# ---------------------------------------------------------------------------

class ClaimORM(Base):
    __tablename__ = "claim"

    claim_id = _uuid_pk()
    action_id = Column(PGUUID(as_uuid=True), ForeignKey("action.action_id"))
    claim_description = Column(Text)  # core CV bullet content (formerly result_statement)

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
    metric_type = Column(Text)  # Efficiency Improvement / Cost Saving / Quality Improvement / Other
    metric_label = Column(Text)  # what the number represents, e.g. "processing time"
    metric_value = Column(Text)  # original number/description, e.g. "reduced from 5 days to 2 days"
    metric_confidence = Column(Text)  # Precise / Estimated / Rough impression -- always human-selected

# ---------------------------------------------------------------------------
# Evidence layer -- belongs to a claim
# ---------------------------------------------------------------------------

class EvidenceORM(Base):
    __tablename__ = "evidence"

    evidence_id = _uuid_pk()
    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"))
    evidence_type = Column(Text)
    evidence_location = Column(Text, nullable=True)
    evidence_level = Column(Text)  # L1 (strongest) -> L4 (weakest), auto-derived from evidence_type

# ---------------------------------------------------------------------------
# Tag layer -- LLM-generated, human-confirmed; many-to-many with claim
# ---------------------------------------------------------------------------

class TagORM(Base):
    __tablename__ = "tag"

    tag_id = _uuid_pk()
    tag_name = Column(Text, unique=True, nullable=False)
    tag_category = Column(Text, nullable=False) 

class ClaimTagORM(Base):
    __tablename__ = "claim_tag"

    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"), primary_key=True)
    tag_id = Column(PGUUID(as_uuid=True), ForeignKey("tag.tag_id"), primary_key=True)

# ---------------------------------------------------------------------------
# Governance layer -- belongs to a claim
# ---------------------------------------------------------------------------

class GovernanceORM(Base):
    __tablename__ = "governance"

    metadata_id = _uuid_pk()
    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"))
    support_level = Column(Text)  # aggregated evidence/metric strength, L1-L4 -- see compute_support_level() / fill_pending_support_levels()
    use_policy = Column(Text, nullable=True)  # usage restriction, e.g. senior roles only / specific industry
    notes_for_llm = Column(Text, nullable=True)  # operating instructions for the generation model
    allowed_for_generation = Column(Boolean, default=True)  # hard governance switch

# ---------------------------------------------------------------------------
# Claim chunks layer (pgvector) -- belongs to a claim
# ---------------------------------------------------------------------------
# NOTE: VECTOR type requires the pgvector-sqlalchemy package's Vector column
# type. If not yet installed (`pip install pgvector`), fall back to Text
# below; swap the embedding column to Vector(1536) once the package is
# installed.

try:
    from pgvector.sqlalchemy import Vector
    _EMBEDDING_COLUMN = Column(Vector(1536))
except ImportError:
    _EMBEDDING_COLUMN = Column(Text)  # fallback, pgvector-sqlalchemy not yet installed

class ClaimChunkORM(Base):
    __tablename__ = "claim_chunks"

    chunk_id = _uuid_pk()
    claim_id = Column(PGUUID(as_uuid=True), ForeignKey("claim.claim_id"))
    support_level = Column(Text, nullable=True)  # a claim may have multiple evidence rows; used to aggregate
    content = Column(Text)
    embedding = _EMBEDDING_COLUMN

# ---------------------------------------------------------------------------
# Job layer -- apply/ module (2026-08-19, replaces generated_outputs -- see
# v1.4 changelog above). No governance concept -- every row is a candidate,
# same pattern as CompanyORM/SchoolORM.
# ---------------------------------------------------------------------------

class JobORM(Base):
    __tablename__ = "job"

    job_id = _uuid_pk()
    company_name = Column(Text)
    job_title = Column(Text)
    job_location = Column(Text)
    job_description = Column(Text)  # full JD text
    source_url = Column(Text)
    date_posted = Column(Date)
    contact_name = Column(Text)
    contact_email = Column(Text)

# ---------------------------------------------------------------------------
# Application layer -- apply/ module (2026-08-19, replaces generated_outputs)
# ---------------------------------------------------------------------------
# current_stage is a fixed CHECK vocabulary, same pattern as
# role_scope/metric_type elsewhere in this schema. interview_round is a
# plain counter incremented by CLI-layer logic (apply/application_log.py)
# when current_stage transitions to 'Interviewing' -- deliberately NOT a
# database trigger, consistent with fill_pending_support_levels()'s
# "business logic stays in Python" philosophy above.

APPLICATION_CURRENT_STAGE_OPTIONS = (
    "Applied",
    "Screening",
    "Interviewing",
    "Offer",
    "Rejected",
    "Ghosted",
)

class ApplicationORM(Base):
    __tablename__ = "application"
    __table_args__ = (
        CheckConstraint(
            f"current_stage IN ({APPLICATION_CURRENT_STAGE_OPTIONS})",
            name="application_current_stage_valid",
        ),
    )

    application_id = _uuid_pk()
    job_id = Column(PGUUID(as_uuid=True), ForeignKey("job.job_id"))  # company_name/job_title read through this FK
    file_folder = Column(Text, nullable=True)  # path to the CV version actually used for this application
    current_stage = Column(Text, nullable=False, default="Applied")
    interview_round = Column(Integer, nullable=False, default=0)  # incremented by CLI logic, not a DB trigger
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

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

    if create_values is not None:
        new_obj = model(**create_values)
        session.add(new_obj)
        session.flush()
        return new_obj

    return None

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

# ---------------------------------------------------------------------------
# Governance support_level computation
# ---------------------------------------------------------------------------
# support_level is a RULE-BASED aggregate, never LLM-scored or freely
# user-selected. It combines two independent signals collected earlier in
# the add-claim flow, which answer two DIFFERENT questions and therefore
# aggregate in OPPOSITE directions across multiple rows:
#
# - evidence_level (L1 strongest -> L4 weakest), one per evidence row --
#   answers "did this actually happen, and how well is that documented?"
#   Aggregates with OR logic: take the STRONGEST level among them.
#
# - metric_confidence (Precise/Estimated/Rough impression), one per
#   claim_metric row -- answers "how accurate is the number we're
#   citing?" Aggregates with AND logic: take the WEAKEST confidence
#   among them.
#
# Final aggregation: support_level = the WEAKER of (evidence_component,
# metric_component).

_LEVEL_STRENGTH: dict[str, int] = {"L1": 4, "L2": 3, "L3": 2, "L4": 1}

_METRIC_CONFIDENCE_TO_LEVEL: dict[str, str] = {
    "Precise": "L1",
    "Estimated": "L3",
    "Rough impression": "L4",
}

def compute_support_level(
    evidence_levels: Sequence[str],
    metric_confidences: Sequence[str] = (),
) -> str:
    """
    Computes governance.support_level from the evidence and metric data
    for a claim. Called from two places:
    1. fill_pending_support_levels() below, explicitly, right after
       add-claim's transaction commits.
    2. scripts/backfill_governance.py, for one-off migration of claims
       that predate the governance layer entirely.

    `evidence_levels`: the evidence_level of every EvidenceORM row for this
    claim (must be non-empty). Aggregated with MAX (OR logic).

    `metric_confidences`: the metric_confidence of every ClaimMetricORM
    row for this claim. May be empty. Aggregated with MIN after mapping
    onto the L1-L4 scale (AND logic).

    Returns the weaker of the two aggregated components.
    """
    if not evidence_levels:
        raise ValueError("compute_support_level requires at least one evidence_level")

    evidence_component = max(evidence_levels, key=lambda lvl: _LEVEL_STRENGTH[lvl])

    if not metric_confidences:
        return evidence_component

    mapped_metric_levels = [
        _METRIC_CONFIDENCE_TO_LEVEL[mc]
        for mc in metric_confidences
        if mc in _METRIC_CONFIDENCE_TO_LEVEL
    ]
    if not mapped_metric_levels:
        return evidence_component

    metric_component = min(mapped_metric_levels, key=lambda lvl: _LEVEL_STRENGTH[lvl])

    return min([evidence_component, metric_component], key=lambda lvl: _LEVEL_STRENGTH[lvl])

# ---------------------------------------------------------------------------
# Support_level autofill -- explicit call, no event registration (v1.3)
# ---------------------------------------------------------------------------

def fill_pending_support_levels() -> int:
    """
    Scans for GovernanceORM rows with support_level IS NULL and fills them
    in, using a fresh session/transaction. Intended to be called explicitly
    by the caller right after it knows claim-related data (evidence,
    metrics, governance) has just been committed -- see onboard/cli.py's
    add_claim(), the only caller.

    Returns the number of governance rows that were filled in (0 if there
    was nothing pending).
    """
    fill_session = SessionLocal()
    try:
        pending = (
            fill_session.query(GovernanceORM)
            .filter(GovernanceORM.support_level.is_(None))
            .all()
        )
        if not pending:
            return 0

        filled_count = 0
        for governance in pending:
            evidences = list_rows(fill_session, EvidenceORM, claim_id=governance.claim_id)
            if not evidences:
                continue

            metrics = list_rows(fill_session, ClaimMetricORM, claim_id=governance.claim_id)
            governance.support_level = compute_support_level(
                evidence_levels=[e.evidence_level for e in evidences],
                metric_confidences=[m.metric_confidence for m in metrics],
            )
            filled_count += 1

        fill_session.commit()
        return filled_count
    finally:
        fill_session.close()
