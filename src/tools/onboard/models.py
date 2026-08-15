"""
onboard/models.py
Version: v1.2 (2026-08-10)

Changelog:
- v1.2 (2026-08-10): Schema-alignment fix. ClaimBase previously still had
  result_statement / value_type / metric / metric_confidence fields, but
  001_init_schema.sql's own changelog note already renamed/removed these
  on the DB side (result_statement -> description; value_type/metric/
  metric_confidence moved out to the new claim_metric one-to-many table).
  cli.py's _resolve_claim() already writes ClaimORM with only action_id +
  description, so this brings the Pydantic layer back in sync with what's
  actually being persisted. Also removed the orphaned `ValueType` Literal
  (revenue/cost/time/risk/process/team_enablement), which belonged only to
  the old value_type field and used a completely different category set
  than the metric_type vocabulary now in use (see prompts.py
  METRIC_TYPE_OPTIONS: Efficiency Improvement/Cost Saving/Quality
  Improvement/Other) -- keeping it around would have been actively
  misleading. Added ClaimMetricBase/ClaimMetricCreate/ClaimMetric to close
  the modeling gap for the claim_metric table, which exists in
  001_init_schema.sql and is already written to via ClaimMetricORM in
  cli.py, but previously had no Pydantic validation model at all.
- v1.1 (2026-08-10): Translated all internal comments/docstrings from
  Chinese to English. No logic changes in that pass.
- v1.0 (2026-08-04, inferred): Initial Pydantic models for the CONFIRMED
  portion of the Source of Truth schema.

Pydantic models for the Source of Truth schema (001_init_schema.sql):
company, position, school, degree, project, problem, action, claim,
claim_metric, evidence, governance, tag, claim_tag.

Still intentionally NOT modeled here: claim_chunks (pgvector embedding
layer -- no embedding-generation function exists yet, see core/llm_client.py
notes) and generated_outputs (output-tracking layer, belongs to the
gencv/application_log.py side of the pipeline rather than the onboard
data-entry side). Add those once their consuming code is ready to use them.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Literal
from uuid import UUID

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Company layer / School layer (Upper-level entities)
# ---------------------------------------------------------------------------

class CompanyBase(BaseModel):
    company_name: str
    location: Optional[str] = None

class CompanyCreate(CompanyBase):
    pass

class Company(CompanyBase):
    company_id: UUID

class SchoolBase(BaseModel):
    school_name: str
    location: Optional[str] = None

class SchoolCreate(SchoolBase):
    pass

class School(SchoolBase):
    school_id: UUID

# ---------------------------------------------------------------------------
# Position layer (work track)
# ---------------------------------------------------------------------------

class PositionBase(BaseModel):
    company_id: UUID
    title: str
    department: Optional[str] = None
    responsibility: Optional[str] = None
    responsibility_source: Optional[str] = None  # NULL => LLM-generated w/o source file
    start_date: date
    end_date: Optional[date] = None

    @model_validator(mode="after")
    def check_dates(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self

class PositionCreate(PositionBase):
    pass

class Position(PositionBase):
    position_id: UUID

# ---------------------------------------------------------------------------
# Degree layer (education track)
# ---------------------------------------------------------------------------

class DegreeBase(BaseModel):
    school_id: UUID
    degree_name: str  # e.g. "MA", "Exchange Program", "Summer Research Program"
    subject: Optional[str] = None
    start_date: date
    end_date: Optional[date] = None

    @model_validator(mode="after")
    def check_dates(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self

class DegreeCreate(DegreeBase):
    pass

class Degree(DegreeBase):
    degree_id: UUID

# ---------------------------------------------------------------------------
# Project layer (shared by both tracks, position_id XOR degree_id)
# ---------------------------------------------------------------------------

class ProjectBase(BaseModel):
    position_id: Optional[UUID] = None
    degree_id: Optional[UUID] = None
    project_name: str
    start_date: date
    end_date: Optional[date] = None
    project_context: Optional[str] = None  # LLM-drafted, human-confirmed narrative

    @model_validator(mode="after")
    def check_exclusive_parent(self):
        has_position = self.position_id is not None
        has_degree = self.degree_id is not None
        if has_position == has_degree:
            raise ValueError(
                "project must belong to exactly one of position_id or degree_id"
            )
        return self

    @model_validator(mode="after")
    def check_dates(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    project_id: UUID

# ---------------------------------------------------------------------------
# Composite payload used by the add-project CLI flow (Step 0-3 only)
# ---------------------------------------------------------------------------

class AddProjectPayload(BaseModel):
    """
    Bundles everything collected across Step 0-3 of `add-project` so the
    whole thing can be validated once before being written inside a single
    DB transaction. Does not touch problem/action/claim layers.
    """
    track: Literal["work", "education"]

    company: Optional[CompanyCreate] = None
    existing_company_id: Optional[UUID] = None
    school: Optional[SchoolCreate] = None
    existing_school_id: Optional[UUID] = None

    position: Optional[PositionCreate] = None
    existing_position_id: Optional[UUID] = None
    degree: Optional[DegreeCreate] = None
    existing_degree_id: Optional[UUID] = None

    project: ProjectCreate

    @model_validator(mode="after")
    def check_track_consistency(self):
        if self.track == "work":
            if not (self.company or self.existing_company_id):
                raise ValueError("work track requires company or existing_company_id")
            if not (self.position or self.existing_position_id):
                raise ValueError("work track requires position or existing_position_id")
            if self.school or self.existing_school_id or self.degree or self.existing_degree_id:
                raise ValueError("work track must not include education-side data")
        else:  # education
            if not (self.school or self.existing_school_id):
                raise ValueError("education track requires school or existing_school_id")
            if not (self.degree or self.existing_degree_id):
                raise ValueError("education track requires degree or existing_degree_id")
            if self.company or self.existing_company_id or self.position or self.existing_position_id:
                raise ValueError("education track must not include work-side data")
        return self

# ---------------------------------------------------------------------------
# Problem layer / Action layer
# ---------------------------------------------------------------------------

class ProblemBase(BaseModel):
    project_id: UUID
    description: str
    stakeholder_impact: Optional[str] = None
    baseline_metric: Optional[str] = None

class ProblemCreate(ProblemBase):
    pass

class Problem(ProblemBase):
    problem_id: UUID

class ActionBase(BaseModel):
    problem_id: UUID
    description: str
    role_scope: Optional[str] = None  # e.g. "Led"/"Collaborated"/"Supported"

class ActionCreate(ActionBase):
    pass

class Action(ActionBase):
    action_id: UUID

# ---------------------------------------------------------------------------
# Result/Claim layer
# ---------------------------------------------------------------------------
# Aligned (v1.2) with 001_init_schema.sql's current `claim` table shape:
# claim_id, action_id, description only. value_type/metric/metric_confidence
# no longer live here -- see ClaimMetricBase below for the one-to-many
# replacement (claim_metric table).

class ClaimBase(BaseModel):
    action_id: UUID
    description: str  # result statement -- the core content source for CV bullets

class ClaimCreate(ClaimBase):
    pass

class Claim(ClaimBase):
    claim_id: UUID

# ---------------------------------------------------------------------------
# Claim metric layer (added v1.2) -- a claim may carry MULTIPLE
# quantifiable metrics, and not every metric cleanly fits a single fixed
# category, which is why this was split out of `claim` into its own
# one-to-many table. metric_confidence is always a manual human judgment
# (see prompts.py METRIC_EXTRACTION_SYSTEM_PROMPT rule 3 -- LLM
# self-reported confidence is not trusted), never LLM-derived.
# ---------------------------------------------------------------------------

class ClaimMetricBase(BaseModel):
    claim_id: UUID
    metric_type: Optional[str] = None  # Efficiency Improvement / Cost Saving / Quality Improvement / Other
    metric_label: Optional[str] = None  # what the number represents, e.g. "processing time"
    metric_value: Optional[str] = None  # original number/description, e.g. "reduced from 5 days to 2 days"
    metric_confidence: Optional[str] = None  # Precise / Estimated / Rough impression -- always human-selected

class ClaimMetricCreate(ClaimMetricBase):
    pass

class ClaimMetric(ClaimMetricBase):
    metric_id: UUID

# ---------------------------------------------------------------------------
# Evidence layer -- one claim can have MANY evidence rows
# ---------------------------------------------------------------------------

EvidenceLevel = Literal["L1", "L2", "L3", "L4"]

class EvidenceBase(BaseModel):
    claim_id: UUID
    evidence_type: Optional[str] = None  # source file / structured record / self-report / inference
    evidence_location: Optional[str] = None
    evidence_level: EvidenceLevel

class EvidenceCreate(EvidenceBase):
    pass

class Evidence(EvidenceBase):
    evidence_id: UUID

# ---------------------------------------------------------------------------
# Governance layer -- one row per claim, aggregated from its evidence
# ---------------------------------------------------------------------------

class GovernanceBase(BaseModel):
    claim_id: UUID
    support_level: Optional[EvidenceLevel] = None  # rule-based aggregate (max), not LLM-scored
    use_policy: Optional[str] = None  # NULL => follow SKILLS.md global rules
    notes_for_llm: Optional[str] = None  # NULL => follow SKILLS.md global rules
    allowed_for_generation: bool = True

class GovernanceCreate(GovernanceBase):
    pass

class Governance(GovernanceBase):
    metadata_id: UUID

# ---------------------------------------------------------------------------
# Tag governance layer
# ---------------------------------------------------------------------------

TagCategory = Literal["skill", "domain", "job_function"]

class TagBase(BaseModel):
    tag_name: str
    tag_category: TagCategory
    aliases: list[str] = []

class TagCreate(TagBase):
    pass

class Tag(TagBase):
    tag_id: UUID

class ClaimTag(BaseModel):
    claim_id: UUID
    tag_id: UUID

# ---------------------------------------------------------------------------
# Composite payload used by the add-claim CLI flow
# ---------------------------------------------------------------------------

# Mirrors the add-project pattern: bundle everything collected across the
# interactive flow (problem -> action -> claim -> N metric rows -> N
# evidence rows -> tags) so it can be validated once before being written
# inside a single short DB transaction. governance.support_level is
# deliberately NOT part of this payload -- it is computed server-side via
# db.compute_support_level() from the evidence_list below, right before the
# governance row is written.
#
# metric_list added (v1.2) alongside evidence_list, now that ClaimMetric
# has a model -- a claim may have zero metrics (not every result is
# quantifiable) so this list is allowed to be empty, unlike evidence_list.

class AddClaimPayload(BaseModel):
    problem: Optional[ProblemCreate] = None
    existing_problem_id: Optional[UUID] = None

    action: Optional[ActionCreate] = None
    existing_action_id: Optional[UUID] = None

    claim: ClaimCreate
    metric_list: list[ClaimMetricCreate] = []
    evidence_list: list[EvidenceCreate]
    tag_names: list[str] = []  # resolved to tag_id via fuzzy match at write time

    governance_use_policy: Optional[str] = None
    governance_notes_for_llm: Optional[str] = None
    governance_allowed_for_generation: bool = True

    @model_validator(mode="after")
    def check_problem_consistency(self):
        if not (self.problem or self.existing_problem_id):
            raise ValueError("must provide either problem or existing_problem_id")
        if self.problem and self.existing_problem_id:
            raise ValueError("provide only one of problem or existing_problem_id, not both")
        return self

    @model_validator(mode="after")
    def check_action_consistency(self):
        if not (self.action or self.existing_action_id):
            raise ValueError("must provide either action or existing_action_id")
        if self.action and self.existing_action_id:
            raise ValueError("provide only one of action or existing_action_id, not both")
        return self

    @model_validator(mode="after")
    def check_evidence_nonempty(self):
        if len(self.evidence_list) == 0:
            raise ValueError(
                "a claim must have at least one evidence entry "
                "(even a self-reported / L4 one) -- claims with zero "
                "evidence should not exist per governance principles"
            )
        return self
