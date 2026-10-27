"""Workflow orchestration.

An explicit, linear-with-fan-out state machine rather than a free-running agent
loop. Each stage has a single responsibility and a validated output, so the
pipeline is predictable, testable and cheap on CPU-only hardware:

    JD Analyst ─┐
                ├─> Resume Analyst (per candidate) ─> Evidence Matcher (per candidate)
                │       ─> Ranker ─> Interview Planner (top-k only) ─> Compliance Guardrail
                └──────────────────────────────────────────────────────────> ShortlistReport

The orchestrator owns retries/fallbacks (via ``Agent._execute``), the trace,
progress reporting and anonymisation aliases.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from recruiter.agents.compliance import ComplianceGuardrail
from recruiter.agents.interviewer import InterviewPlanner
from recruiter.agents.jd_analyst import JobDescriptionAnalyst
from recruiter.agents.matcher import EvidenceMatcher, ScoringConfig, rank
from recruiter.agents.resume_analyst import ResumeAnalyst
from recruiter.llm import LLMProvider, make_provider
from recruiter.privacy import redact
from recruiter.schemas import AgentStep, JobProfile, ShortlistReport

ProgressFn = Callable[[str, float], None]


@dataclass
class ResumeInput:
    filename: str
    text: str
    candidate_id: str = ""


@dataclass
class WorkflowConfig:
    provider_spec: str = "rules"
    temperature: float = 0.1
    blind_review: bool = True  # show "Candidate A" instead of names
    interview_top_k: int = 3
    questions_per_candidate: int = 6
    # Which agents may call the LLM. On CPU-only machines each call takes ~1-2 min with a 3B
    # model, so the default limits LLM use to the two stages where it adds most value.
    llm_stages: tuple[str, ...] = ("jd", "interview")
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @property
    def llm_narratives(self) -> bool:
        return "narrative" in self.llm_stages


class RecruitmentWorkflow:
    def __init__(self, config: WorkflowConfig | None = None, provider: LLMProvider | None = None):
        self.config = config or WorkflowConfig()
        self.provider = provider if provider is not None else make_provider(self.config.provider_spec, self.config.temperature)
        stages = set(self.config.llm_stages)

        def for_stage(stage: str) -> LLMProvider | None:
            return self.provider if stage in stages else None

        self.jd_agent = JobDescriptionAnalyst(for_stage("jd"))
        self.resume_agent = ResumeAnalyst(for_stage("resume"))
        self.matcher = EvidenceMatcher(for_stage("narrative"), self.config.scoring)
        self.interviewer = InterviewPlanner(for_stage("interview"), self.config.questions_per_candidate)
        self.guardrail = ComplianceGuardrail()

    @property
    def provider_label(self) -> str:
        return self.provider.label if self.provider else "rules"

    def run(self, job_text: str, resumes: list[ResumeInput], progress: ProgressFn | None = None,
            job: JobProfile | None = None) -> ShortlistReport:
        if not job_text.strip() and job is None:
            raise ValueError("A job description is required.")
        if not resumes:
            raise ValueError("At least one resume is required.")

        trace: list[AgentStep] = []
        notify = progress or (lambda msg, frac: None)
        total_steps = 2 + 2 * len(resumes) + min(self.config.interview_top_k, len(resumes))
        done = 0

        def tick(msg: str) -> None:
            nonlocal done
            done += 1
            notify(msg, min(1.0, done / total_steps))

        t0 = time.perf_counter()
        notify("Analysing job description...", 0.0)
        if job is None:
            job = self.jd_agent.run(job_text)
            trace.append(self.jd_agent.last_step)
        else:  # a recruiter-edited profile is re-used verbatim (human in the loop)
            trace.append(AgentStep(agent=self.jd_agent.name, mode="rules", detail="recruiter-edited profile reused"))
        tick("Job profile ready")

        candidates, redacted = [], {}
        for i, r in enumerate(resumes):
            cid = r.candidate_id or f"C{i + 1:03d}"
            alias = f"Candidate {_letters(i)}" if self.config.blind_review else None
            notify(f"Reading resume {i + 1}/{len(resumes)}: {r.filename}", done / total_steps)
            profile = self.resume_agent.run(r.text, cid, r.filename, alias=alias, job=job)
            trace.append(self.resume_agent.last_step)
            candidates.append(profile)
            redacted[cid] = redact(r.text).text
            tick(f"Parsed {profile.display_name}")

        assessments = []
        for c in candidates:
            notify(f"Matching {c.display_name}...", done / total_steps)
            assessments.append(self.matcher.run(job, c))
            trace.append(self.matcher.last_step)
            tick(f"Scored {c.display_name}")
        assessments = rank(assessments)

        plans = {}
        by_id = {c.candidate_id: c for c in candidates}
        for a in assessments[: self.config.interview_top_k]:
            notify(f"Preparing interview plan for {a.display_name}...", done / total_steps)
            plans[a.candidate_id] = self.interviewer.run(job, by_id[a.candidate_id], a)
            trace.append(self.interviewer.last_step)
            tick(f"Interview plan for {a.display_name}")

        compliance = self.guardrail.run(job, candidates, assessments, plans, redacted)
        trace.append(self.guardrail.last_step)
        tick("Compliance review complete")

        trace.append(AgentStep(agent="Orchestrator", mode="rules",
                               duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                               detail=f"{len(resumes)} candidates, provider={self.provider_label}"))
        notify("Done", 1.0)
        return ShortlistReport(
            run_id=uuid.uuid4().hex[:12], job=job, candidates=candidates, assessments=assessments,
            interview_plans=plans, compliance=compliance, trace=trace, provider=self.provider_label,
        )


def rescore(report: ShortlistReport, job: JobProfile, scoring: ScoringConfig | None = None,
            interview_top_k: int = 3) -> ShortlistReport:
    """Re-run matching, ranking, interview planning and compliance on stored candidate
    profiles after a recruiter edits the job profile or scoring policy (no re-parsing, no LLM)."""
    t0 = time.perf_counter()
    matcher = EvidenceMatcher(None, scoring)
    interviewer = InterviewPlanner(None)
    trace = [AgentStep(agent="Recruiter", mode="rules", detail="job profile / scoring policy edited")]
    assessments = []
    for c in report.candidates:
        assessments.append(matcher.run(job, c))
        trace.append(matcher.last_step)
    assessments = rank(assessments)
    by_id = {c.candidate_id: c for c in report.candidates}
    plans = {}
    for a in assessments[:interview_top_k]:
        plans[a.candidate_id] = interviewer.run(job, by_id[a.candidate_id], a)
        trace.append(interviewer.last_step)
    guardrail = ComplianceGuardrail()
    compliance = guardrail.run(job, report.candidates, assessments, plans, {})
    trace.append(guardrail.last_step)
    trace.append(AgentStep(agent="Orchestrator", mode="rules", duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                           detail=f"re-scored {len(assessments)} candidates from run {report.run_id}"))
    return ShortlistReport(run_id=uuid.uuid4().hex[:12], job=job, candidates=report.candidates,
                           assessments=assessments, interview_plans=plans, compliance=compliance,
                           trace=trace, provider="rules (re-score)")


def _letters(i: int) -> str:
    out = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        out = chr(65 + rem) + out
    return out
