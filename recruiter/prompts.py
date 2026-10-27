"""Prompt templates and the compact response schemas the LLM is asked to fill.

The response schemas are deliberately smaller than the internal domain models:
small local models (3-8B) follow short, flat schemas far more reliably. Agents
post-process these into the full domain models, normalise skills against the
taxonomy and verify every quoted piece of evidence against the source text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PROMPT_VERSION = "v1.2"

GUARDRAIL_PREAMBLE = (
    "You are part of a recruitment decision-support tool. Base every statement only on the text provided. "
    "Never infer or mention age, gender, ethnicity, nationality, religion, disability, marital or family status, "
    "pregnancy, sexual orientation or any other protected characteristic. Never recommend rejecting a candidate; "
    "a human recruiter makes all decisions."
)


# --------------------------------------------------------------------------- #
# Agent 1: Job Description Analyst
# --------------------------------------------------------------------------- #
class LLMSkill(BaseModel):
    skill: str
    priority: int = Field(ge=1, le=5, description="5 = critical to the role, 1 = minor nice-to-have")


class LLMJobExtraction(BaseModel):
    title: str
    summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[LLMSkill] = Field(default_factory=list)
    preferred_skills: list[LLMSkill] = Field(default_factory=list)
    min_years_experience: float | None = None
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unspecified"] = "unspecified"


JD_SYSTEM = (
    "You are an expert technical recruiter who turns job descriptions into structured requirement profiles. "
    + GUARDRAIL_PREAMBLE
)

JD_USER = """Analyse the job description below.

Rules:
- required_skills: skills, tools or competencies the text says are required / must-have / essential.
- preferred_skills: skills described as preferred, nice to have, a plus or bonus.
- Use short canonical skill names (e.g. "Python", "Docker", "Stakeholder Management"), one skill per entry.
- priority: 5 for skills central to the role (in the title or repeated), 3 for typical requirements, 1-2 for minor ones.
- min_years_experience: the minimum years of experience stated, or null if none is stated.
- responsibilities: up to 8 short bullet points, copied or tightly paraphrased.
- Do not invent requirements that are not in the text.

JOB DESCRIPTION:
\"\"\"
{job_text}
\"\"\"
"""


# --------------------------------------------------------------------------- #
# Agent 2: Resume Analyst
# --------------------------------------------------------------------------- #
class LLMResumeSkill(BaseModel):
    skill: str
    evidence_quote: str = Field(description="Exact phrase copied from the resume that shows this skill")
    used_in_work: bool = Field(description="True if the quote describes applying the skill in a job or project")


class LLMExperience(BaseModel):
    title: str = ""
    organisation: str = ""
    start: str | None = None
    end: str | None = None
    highlights: list[str] = Field(default_factory=list)


class LLMResumeExtraction(BaseModel):
    headline: str = ""
    total_years_experience: float | None = None
    skills: list[LLMResumeSkill] = Field(default_factory=list)
    experience: list[LLMExperience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


RESUME_SYSTEM = (
    "You are a meticulous resume analyst. You extract skills and experience together with verbatim evidence. "
    + GUARDRAIL_PREAMBLE
)

RESUME_USER = """Extract a structured profile from the (anonymised) resume below.

Rules:
- For every skill, evidence_quote MUST be copied exactly from the resume (5-25 words). If you cannot quote it, omit the skill.
- used_in_work is true only when the quote describes doing work with the skill, not just listing it.
- Focus on skills relevant to these job requirements, but include other clear skills too: {focus_skills}
- experience: each role with title, organisation, start/end as written (e.g. "Jan 2020", "Present"), up to 3 highlights.
- total_years_experience: sum of professional experience in years, or null if unclear.

RESUME:
\"\"\"
{resume_text}
\"\"\"
"""


# --------------------------------------------------------------------------- #
# Agent 3: Match Analyst (narrative only - the score is computed deterministically)
# --------------------------------------------------------------------------- #
class LLMMatchNarrative(BaseModel):
    strengths: list[str] = Field(default_factory=list, description="Up to 4 evidence-backed strengths")
    gaps: list[str] = Field(default_factory=list, description="Up to 4 missing or weakly evidenced requirements")
    rationale: str = Field(description="3-4 sentence neutral summary for a recruiter")


MATCH_SYSTEM = (
    "You are a fair, evidence-focused hiring analyst writing a short summary for a recruiter. " + GUARDRAIL_PREAMBLE
)

MATCH_USER = """Role: {title}
Computed match score: {score}/100 (essential coverage {coverage}%, experience {years} yrs vs {min_years} required)

Requirement-by-requirement evidence (from the resume):
{evidence_table}

Write strengths, gaps and a neutral rationale. Only refer to the evidence above; do not speculate about the person.
"""


# --------------------------------------------------------------------------- #
# Agent 4: Interview Question Generator
# --------------------------------------------------------------------------- #
class LLMQuestion(BaseModel):
    question: str
    purpose: Literal["verify_evidence", "probe_gap", "behavioural", "role_specific"]
    target_skill: str | None = None
    what_good_looks_like: str = ""


class LLMInterviewPlan(BaseModel):
    questions: list[LLMQuestion] = Field(default_factory=list)


INTERVIEW_SYSTEM = (
    "You are a structured-interview designer. You write job-related, competency-based questions that are "
    "identical in spirit for every candidate and never touch personal life. " + GUARDRAIL_PREAMBLE
)

INTERVIEW_USER = """Role: {title}
Candidate strengths (with evidence): {strengths}
Gaps or weak evidence: {gaps}

Write {n} interview questions:
- about half "verify_evidence" questions that ask the candidate to go deeper on a claimed skill,
- "probe_gap" questions for the gaps (phrased as an opportunity to show related experience),
- at least one "behavioural" question in STAR format relevant to the responsibilities: {responsibilities}
For each add what_good_looks_like: one sentence describing a strong answer.
"""
