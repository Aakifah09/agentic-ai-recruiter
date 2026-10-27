"""Agent 5 - Fairness & Compliance Guardrail.

Runs after every other agent and before anything reaches the recruiter:
  * flags exclusionary wording in the job description;
  * verifies all evidence snippets are grounded in the (redacted) resume text;
  * removes interview questions and narrative sentences that touch protected
    characteristics;
  * checks redaction was effective (no emails/phones left);
  * enforces that output is advisory - no automatic rejections.
"""

from __future__ import annotations

import re
import time

from recruiter.privacy import find_protected_terms, redact
from recruiter.schemas import (
    AgentStep,
    CandidateProfile,
    ComplianceIssue,
    ComplianceReport,
    InterviewPlan,
    JobProfile,
    MatchAssessment,
    Recommendation,
)
from recruiter.textutils import is_grounded

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_FORBIDDEN_DECISIONS = re.compile(r"\b(reject|do not hire|don't hire|not hire|unsuitable|disqualif)\w*", re.I)


class ComplianceGuardrail:
    name = "Compliance Guardrail"
    description = "Checks fairness, grounding and privacy; strips non-compliant content."

    def __init__(self) -> None:
        self.last_step: AgentStep | None = None

    def run(
        self,
        job: JobProfile,
        candidates: list[CandidateProfile],
        assessments: list[MatchAssessment],
        plans: dict[str, InterviewPlan],
        redacted_texts: dict[str, str],
    ) -> ComplianceReport:
        start = time.perf_counter()
        issues: list[ComplianceIssue] = []

        for flag in job.bias_flags:
            issues.append(ComplianceIssue(severity="warning", location="job description", message=flag))

        by_id = {c.candidate_id: c for c in candidates}
        for a in assessments:
            text = redacted_texts.get(a.candidate_id, "")
            cand = by_id.get(a.candidate_id)

            # Grounding: every evidence snippet must exist in the resume (skipped when the
            # source text is unavailable, e.g. re-scoring a stored run).
            for m in (a.requirement_matches if text else []):
                kept = [e for e in m.evidence if e.startswith("Related skill") or is_grounded(e.rstrip("."), text, 0.75)]
                if len(kept) < len(m.evidence):
                    issues.append(ComplianceIssue(severity="warning", location=a.display_name,
                                                  message=f"Removed ungrounded evidence for {m.skill}."))
                    m.evidence = kept

            # Narrative must not reference protected characteristics or make decisions.
            for field in ("strengths", "gaps"):
                items = getattr(a, field)
                clean = [s for s in items if not find_protected_terms(s) and not _FORBIDDEN_DECISIONS.search(s)]
                if len(clean) < len(items):
                    issues.append(ComplianceIssue(severity="blocked", location=a.display_name,
                                                  message=f"Removed {len(items) - len(clean)} {field} item(s) referencing protected or decision language."))
                setattr(a, field, clean)
            sentences = re.split(r"(?<=[.!?])\s+", a.rationale)
            clean_sentences = [s for s in sentences if not find_protected_terms(s) and not _FORBIDDEN_DECISIONS.search(s)]
            if len(clean_sentences) < len(sentences):
                issues.append(ComplianceIssue(severity="blocked", location=a.display_name,
                                              message="Removed rationale sentence(s) referencing protected characteristics or decisions."))
                a.rationale = " ".join(clean_sentences)

            if a.recommendation not in set(Recommendation):
                a.recommendation = Recommendation.LIMITED

            if cand and _EMAIL.search(" ".join(sum((s.snippets for s in cand.skills), []))):
                issues.append(ComplianceIssue(severity="warning", location=a.display_name,
                                              message="Contact details found in evidence; re-redacted."))
                for s in cand.skills:
                    s.snippets = [redact(x).text for x in s.snippets]

        removed = 0
        for cid, plan in plans.items():
            keep = []
            for q in plan.questions:
                terms = find_protected_terms(q.question)
                if terms:
                    removed += 1
                    issues.append(ComplianceIssue(severity="blocked", location=f"interview plan ({by_id[cid].display_name if cid in by_id else cid})",
                                                  message=f"Removed question touching protected topic(s): {', '.join(terms)}."))
                else:
                    keep.append(q)
            plan.questions = keep

        report = ComplianceReport(
            passed=not any(i.severity == "blocked" for i in issues) and not job.bias_flags,
            issues=issues, removed_questions=removed,
        )
        self.last_step = AgentStep(
            agent=self.name, mode="rules", duration_ms=round((time.perf_counter() - start) * 1000, 1),
            detail=f"{len(issues)} issue(s); {removed} question(s) removed",
        )
        return report
