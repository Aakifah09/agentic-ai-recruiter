"""Agent 4 - Interview Question Generator.

Produces a structured interview plan per shortlisted candidate: questions that
verify claimed evidence, probe gaps, and assess role-relevant behaviours.
"""

from __future__ import annotations

from recruiter import prompts
from recruiter.agents.base import Agent
from recruiter.llm import structured_call
from recruiter.schemas import (
    CandidateProfile,
    EvidenceStrength,
    Importance,
    InterviewPlan,
    InterviewQuestion,
    JobProfile,
    MatchAssessment,
)
from recruiter.skills import related_skills


class InterviewPlanner(Agent):
    name = "Interview Planner"
    description = "Generates evidence-verification, gap-probing and behavioural interview questions."

    def __init__(self, provider=None, n_questions: int = 6):
        super().__init__(provider)
        self.n_questions = n_questions

    def run(self, job: JobProfile, candidate: CandidateProfile, assessment: MatchAssessment) -> InterviewPlan:
        plan = self._execute(
            (lambda: self._run_llm(job, candidate, assessment)) if self.provider else None,
            lambda: self._run_rules(job, candidate, assessment),
        )
        if self.last_step:
            self.last_step.agent = f"{self.name} [{candidate.display_name}]"
            self.last_step.detail = f"{len(plan.questions)} questions" + (
                f" | {self.last_step.detail}" if self.last_step.detail else "")
        return plan

    def _run_rules(self, job: JobProfile, candidate: CandidateProfile, a: MatchAssessment) -> InterviewPlan:
        qs: list[InterviewQuestion] = []
        by_weight = sorted(a.requirement_matches, key=lambda m: -m.weight)

        for m in [m for m in by_weight if m.strength == EvidenceStrength.DEMONSTRATED][:2]:
            snippet = m.evidence[0][:140] if m.evidence else ""
            qs.append(InterviewQuestion(
                question=(f"Your resume says: \"{snippet}\". Walk me through how you used {m.skill} there - "
                          "what you personally built, the key design decisions, and the measurable outcome."),
                purpose="verify_evidence", target_skill=m.skill,
                what_good_looks_like=f"Specific, first-person detail on {m.skill}: architecture, trade-offs, metrics and lessons learned.",
            ))

        for m in [m for m in by_weight if m.strength == EvidenceStrength.LISTED and m.importance == Importance.ESSENTIAL][:1]:
            qs.append(InterviewQuestion(
                question=f"{m.skill} is listed in your skills. Describe a concrete problem you solved with {m.skill} and how you approached it.",
                purpose="role_specific", target_skill=m.skill,
                what_good_looks_like=f"A real example demonstrating hands-on depth in {m.skill}, not just familiarity.",
            ))

        for m in [m for m in by_weight if m.credit < 0.5 and m.importance == Importance.ESSENTIAL][:2]:
            related = ", ".join(sorted(related_skills(m.skill))[:2])
            alt = f" or with related tools such as {related}" if related else ""
            qs.append(InterviewQuestion(
                question=(f"This role relies on {m.skill}. Tell me about any experience you have with it{alt}, "
                          "and how you would get productive with it in your first months."),
                purpose="probe_gap", target_skill=m.skill,
                what_good_looks_like="Honest account of exposure, transferable experience and a credible learning plan.",
            ))

        responsibilities = job.responsibilities or ["deliver a complex piece of work with ambiguous requirements"]
        for resp in responsibilities[:2]:
            resp_clean = resp.rstrip(".").strip()
            resp_clean = resp_clean[0].lower() + resp_clean[1:] if resp_clean else resp_clean
            qs.append(InterviewQuestion(
                question=f"Tell me about a time you had to {resp_clean}. What was the situation, what did you do, and what was the result?",
                purpose="behavioural",
                what_good_looks_like="A structured STAR answer with a clear personal contribution and outcome.",
            ))
            if len(qs) >= self.n_questions:
                break
        return InterviewPlan(candidate_id=candidate.candidate_id, questions=qs[: self.n_questions])

    def _run_llm(self, job: JobProfile, candidate: CandidateProfile, a: MatchAssessment) -> InterviewPlan:
        result: prompts.LLMInterviewPlan = structured_call(
            self.provider, prompts.INTERVIEW_SYSTEM,
            prompts.INTERVIEW_USER.format(
                title=job.title, strengths="; ".join(a.strengths) or "none recorded",
                gaps="; ".join(a.gaps) or "none recorded", n=self.n_questions,
                responsibilities="; ".join(job.responsibilities[:5]) or "see role title",
            ),
            prompts.LLMInterviewPlan,
        )
        questions = []
        for q in result.questions:
            try:
                questions.append(InterviewQuestion(**q.model_dump()))
            except ValueError:
                continue
        if len(questions) < 3:
            raise ValueError("LLM returned too few usable questions")
        return InterviewPlan(candidate_id=candidate.candidate_id, questions=questions[: self.n_questions])
