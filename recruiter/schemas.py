"""Pydantic schemas shared by every agent.

Each agent consumes and produces one of these models, so every hand-off in the
workflow is validated. LLM outputs are parsed into the same models as the
rule-based extractors, which keeps the two modes interchangeable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Importance(str, Enum):
    ESSENTIAL = "essential"
    DESIRABLE = "desirable"


class EvidenceStrength(str, Enum):
    """How convincingly a resume demonstrates a skill."""

    DEMONSTRATED = "demonstrated"  # used in experience/projects context
    LISTED = "listed"  # only appears in a skills list / summary
    MISSING = "missing"


# --------------------------------------------------------------------------- #
# Job requirements (Agent 1 output)
# --------------------------------------------------------------------------- #
class SkillRequirement(BaseModel):
    skill: str = Field(description="Canonical skill name")
    category: str = "general"
    importance: Importance = Importance.DESIRABLE
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalised weight; all weights sum to 1")
    alternatives: list[str] = Field(default_factory=list, description="Skills that satisfy the requirement equally (\"X or Y\")")
    rationale: str = ""

    @property
    def label(self) -> str:
        return self.skill + (f" (or {', '.join(self.alternatives)})" if self.alternatives else "")

    def accepted(self) -> list[str]:
        return [self.skill, *self.alternatives]


class JobProfile(BaseModel):
    title: str = "Untitled role"
    summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[SkillRequirement] = Field(default_factory=list)
    min_years_experience: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unspecified"] = "unspecified"
    bias_flags: list[str] = Field(default_factory=list, description="Potentially exclusionary JD wording")

    @property
    def essential_skills(self) -> list[SkillRequirement]:
        return [s for s in self.skills if s.importance == Importance.ESSENTIAL]

    @property
    def desirable_skills(self) -> list[SkillRequirement]:
        return [s for s in self.skills if s.importance == Importance.DESIRABLE]


# --------------------------------------------------------------------------- #
# Candidate profile (Agent 2 output)
# --------------------------------------------------------------------------- #
class SkillEvidence(BaseModel):
    skill: str
    category: str = "general"
    strength: EvidenceStrength = EvidenceStrength.LISTED
    snippets: list[str] = Field(default_factory=list, description="Verbatim resume excerpts supporting the skill")


class ExperienceEntry(BaseModel):
    title: str = ""
    organisation: str = ""
    start: str | None = None
    end: str | None = None
    months: int = 0
    highlights: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    candidate_id: str
    display_name: str = Field(description="Name shown to the recruiter (may be an anonymised alias)")
    source_file: str = ""
    headline: str = ""
    total_years_experience: float = 0.0
    skills: list[SkillEvidence] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list, description="Kinds of personal data removed before analysis")
    parse_warnings: list[str] = Field(default_factory=list)

    def skill_map(self) -> dict[str, SkillEvidence]:
        return {s.skill.lower(): s for s in self.skills}


# --------------------------------------------------------------------------- #
# Match assessment (Agent 3 output)
# --------------------------------------------------------------------------- #
class RequirementMatch(BaseModel):
    skill: str
    matched_as: str | None = Field(default=None, description="Candidate skill that satisfied it (may be an alternative)")
    importance: Importance
    weight: float
    strength: EvidenceStrength
    credit: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class Recommendation(str, Enum):
    """Advisory labels only. The tool never rejects a candidate."""

    STRONG = "Strong match - recommend interview"
    POTENTIAL = "Potential match - recruiter review"
    LIMITED = "Limited evidence - human review required"


class MatchAssessment(BaseModel):
    candidate_id: str
    display_name: str
    overall_score: float = Field(ge=0, le=100)
    skill_score: float = Field(ge=0, le=100)
    essential_coverage: float = Field(ge=0, le=100)
    relevant_years: float = 0.0
    experience_score: float = Field(ge=0, le=100)
    qualification_score: float = Field(ge=0, le=100)
    confidence: Literal["high", "medium", "low"] = "medium"
    recommendation: Recommendation = Recommendation.POTENTIAL
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rationale: str = ""
    rank: int = 0


# --------------------------------------------------------------------------- #
# Interview plan (Agent 4 output)
# --------------------------------------------------------------------------- #
class InterviewQuestion(BaseModel):
    question: str
    purpose: Literal["verify_evidence", "probe_gap", "behavioural", "role_specific"]
    target_skill: str | None = None
    what_good_looks_like: str = ""

    @field_validator("question")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("question too short")
        return v


class InterviewPlan(BaseModel):
    candidate_id: str
    questions: list[InterviewQuestion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Guardrail (Agent 5 output)
# --------------------------------------------------------------------------- #
class ComplianceIssue(BaseModel):
    severity: Literal["info", "warning", "blocked"]
    location: str
    message: str


class ComplianceReport(BaseModel):
    passed: bool = True
    issues: list[ComplianceIssue] = Field(default_factory=list)
    removed_questions: int = 0


# --------------------------------------------------------------------------- #
# Workflow trace + final result
# --------------------------------------------------------------------------- #
class AgentStep(BaseModel):
    agent: str
    status: Literal["ok", "fallback", "error"] = "ok"
    mode: Literal["llm", "rules", "hybrid"] = "rules"
    started_at: str = Field(default_factory=utcnow)
    duration_ms: float = 0.0
    detail: str = ""


class HumanReview(BaseModel):
    candidate_id: str
    decision: Literal["pending", "advance", "hold", "needs_info"] = "pending"
    reviewer: str = ""
    notes: str = ""
    updated_at: str = Field(default_factory=utcnow)


class ShortlistReport(BaseModel):
    run_id: str
    job: JobProfile
    candidates: list[CandidateProfile]
    assessments: list[MatchAssessment]
    interview_plans: dict[str, InterviewPlan] = Field(default_factory=dict)
    compliance: ComplianceReport = Field(default_factory=ComplianceReport)
    trace: list[AgentStep] = Field(default_factory=list)
    provider: str = "rules"
    created_at: str = Field(default_factory=utcnow)
    disclaimer: str = (
        "Decision-support output only. Scores summarise evidence found in the documents provided; "
        "they are not hiring decisions. A human recruiter must review every candidate."
    )
