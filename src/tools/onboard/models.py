"""
onboard/models.py
Pydantic models for the CONFIRMED portion of the Source of Truth schema
(001_init_schema.sql): company, position, school, degree, project.

Everything downstream of `project` (problem / action / claim /
evidence / governance / tag / claim_tag / claim_chunks / generated_outputs)
is still being finalized and intentionally NOT modeled here yet. Add those
once their schema is locked.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Literal
from uuid import UUID

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# 公司層 / 學校層 (Upper-level entities)
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
# 職位層 (Position -- work track)
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
# 學業層 (Degree -- education track)
# ---------------------------------------------------------------------------

class DegreeBase(BaseModel):
    school_id: UUID
    degree_name: str  # e.g. "碩士", "交換學程", "暑期研究學程"
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
# 專案層 (Project -- shared by both tracks, position_id XOR degree_id)
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
# 問題層 / 行動層 (Problem / Action)
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
    role_scope: Optional[str] = None  # e.g. "主導"/"協同"/"支援" (Led/Collaborated/Supported)


class ActionCreate(ActionBase):
    pass


class Action(ActionBase):
    action_id: UUID


# ---------------------------------------------------------------------------
# 結果/聲稱層 (Claim)
# ---------------------------------------------------------------------------

ValueType = Literal["revenue", "cost", "time", "risk", "process", "team_enablement"]


class ClaimBase(BaseModel):
    action_id: UUID
    result_statement: str
    value_type: Optional[ValueType] = None
    metric: Optional[str] = None
    metric_confidence: Optional[str] = None  # e.g. "精確值"/"估算"/"方向性"


class ClaimCreate(ClaimBase):
    pass


class Claim(ClaimBase):
    claim_id: UUID


# ---------------------------------------------------------------------------
# 證據層 (Evidence) -- one claim can have MANY evidence rows
# ---------------------------------------------------------------------------

EvidenceLevel = Literal["L1", "L2", "L3", "L4"]


class EvidenceBase(BaseModel):
    claim_id: UUID
    evidence_type: Optional[str] = None  # 原始檔案/結構化紀錄/自述/推論
    evidence_location: Optional[str] = None
    evidence_level: EvidenceLevel


class EvidenceCreate(EvidenceBase):
    pass


class Evidence(EvidenceBase):
    evidence_id: UUID


# ---------------------------------------------------------------------------
# 治理層 (Governance) -- one row per claim, aggregated from its evidence
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
# Tag 治理層
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
#
# Mirrors the add-project pattern: bundle everything collected across the
# interactive flow (problem -> action -> claim -> N evidence rows -> tags)
# so it can be validated once before being written inside a single short
# DB transaction. governance.support_level is deliberately NOT part of
# this payload -- it is computed server-side via db.compute_support_level()
# from the evidence_list below, right before the governance row is written.

class AddClaimPayload(BaseModel):
    problem: Optional[ProblemCreate] = None
    existing_problem_id: Optional[UUID] = None

    action: Optional[ActionCreate] = None
    existing_action_id: Optional[UUID] = None

    claim: ClaimCreate
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
