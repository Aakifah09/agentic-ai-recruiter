"""Agent 3 - Evidence Matcher.

Scores each candidate against the job profile. The score is computed
deterministically from the evidence table so it is reproducible and auditable;
the LLM (when enabled) only writes the narrative, which the compliance agent
then checks.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from recruiter import prompts
from recruiter.agents.base import Agent
from recruiter.llm import structured_call
from recruiter.schemas import (
    CandidateProfile,
    EvidenceStrength,
    Importance,
    JobProfile,
    MatchAssessment,
    Recommendation,
    RequirementMatch,
    SkillEvidence,
)
from recruiter.skills import DEGREE_RANK, find_degrees, find_skills, related_skills


class ScoringConfig(BaseModel):
    """Recruiter-adjustable scoring policy (exposed in the UI)."""

    skill_weight: float = Field(0.65, ge=0, le=1)
    experience_weight: float = Field(0.20, ge=0, le=1)
    qualification_weight: float = Field(0.15, ge=0, le=1)
    demonstrated_credit: float = Field(1.0, ge=0, le=1)
    listed_credit: float = Field(0.5, ge=0, le=1)
    related_credit: float = Field(0.3, ge=0, le=1)
    strong_threshold: float = Field(70, ge=0, le=100)
    limited_threshold: float = Field(45, ge=0, le=100)
    min_essential_coverage_for_strong: float = Field(75, ge=0, le=100)

    def component_weights(self) -> tuple[float, float, float]:
        total = self.skill_weight + self.experience_weight + self.qualification_weight or 1.0
        return self.skill_weight / total, self.experience_weight / total, self.qualification_weight / total


class EvidenceMatcher(Agent):
    name = "Evidence Matcher"
    description = "Computes weighted, evidence-grounded match scores and explains them."

    def __init__(self, provider=None, config: ScoringConfig | None = None):
        super().__init__(provider)
        self.config = config or ScoringConfig()

    def run(self, job: JobProfile, candidate: CandidateProfile) -> MatchAssessment:
        assessment = score_candidate(job, candidate, self.config)
        narrative = self._execute(
            (lambda: self._llm_narrative(job, candidate, assessment)) if self.provider else None,
            lambda: rule_narrative(job, candidate, assessment),
        )
        assessment.strengths, assessment.gaps, assessment.rationale = narrative
        if self.last_step:
            self.last_step.agent = f"{self.name} [{candidate.display_name}]"
            self.last_step.detail = (f"score {assessment.overall_score:.1f}, essential coverage "
                                     f"{assessment.essential_coverage:.0f}%" +
                                     (f" | {self.last_step.detail}" if self.last_step.detail else ""))
        return assessment

    def _llm_narrative(self, job: JobProfile, candidate: CandidateProfile, a: MatchAssessment):
        rows = []
        for m in a.requirement_matches:
            ev = f' - evidence: "{m.evidence[0][:160]}"' if m.evidence else ""
            rows.append(f"- {m.skill} ({m.importance.value}, weight {m.weight:.2f}): {m.strength.value}{ev}")
        result: prompts.LLMMatchNarrative = structured_call(
            self.provider, prompts.MATCH_SYSTEM,
            prompts.MATCH_USER.format(
                title=job.title, score=round(a.overall_score), coverage=round(a.essential_coverage),
                years=a.relevant_years, min_years=job.min_years_experience or "not stated",
                evidence_table="\n".join(rows),
            ),
            prompts.LLMMatchNarrative,
        )
        strengths, gaps, rationale = rule_narrative(job, candidate, a)
        missing = [m.skill for m in a.requirement_matches if m.strength == EvidenceStrength.MISSING]
        llm_strengths = [s for s in result.strengths[:4] if not mentions_any(s, missing)]
        llm_rationale = result.rationale.strip()
        if contradicts_evidence(llm_rationale, missing):
            # Hallucination guard: the narrative claims a skill the evidence table marks missing.
            raise ValueError("LLM narrative contradicts the evidence table")
        return (llm_strengths or strengths, result.gaps[:4] or gaps, llm_rationale or rationale)


def score_candidate(job: JobProfile, candidate: CandidateProfile, config: ScoringConfig | None = None) -> MatchAssessment:
    config = config or ScoringConfig()
    have = candidate.skill_map()
    matches: list[RequirementMatch] = []
    for req in job.skills:
        ev = best_evidence(req.accepted(), have)
        if ev and ev.strength == EvidenceStrength.DEMONSTRATED:
            credit, strength, evidence = config.demonstrated_credit, ev.strength, ev.snippets
        elif ev:
            credit, strength, evidence = config.listed_credit, EvidenceStrength.LISTED, ev.snippets
        else:
            related = sorted(s for s in related_skills(req.skill) if s.lower() in have and s not in req.accepted())
            if related:
                credit, strength = config.related_credit, EvidenceStrength.LISTED
                evidence = [f"Related skill: {related[0]} - " + (have[related[0].lower()].snippets or [""])[0]]
            else:
                credit, strength, evidence = 0.0, EvidenceStrength.MISSING, []
        matches.append(RequirementMatch(skill=req.label, importance=req.importance, weight=req.weight,
                                        matched_as=ev.skill if ev else None,
                                        strength=strength, credit=round(credit, 3), evidence=evidence[:2]))

    total_w = sum(m.weight for m in matches) or 1.0
    skill_score = 100 * sum(m.weight * m.credit for m in matches) / total_w

    essential = [m for m in matches if m.importance == Importance.ESSENTIAL]
    coverage = 100 * sum(1 for m in essential if m.credit >= 0.5) / len(essential) if essential else 100.0

    # Only roles whose description uses at least one of the job's skills count as relevant experience.
    years = relevant_years(job, candidate)
    if job.min_years_experience:
        experience_score = 100 * min(1.0, years / job.min_years_experience)
    else:
        experience_score = 100.0 if years > 0 else 60.0

    qualification_score = qualification_fit(job, candidate)
    ws, we, wq = config.component_weights()
    overall = ws * skill_score + we * experience_score + wq * qualification_score

    if overall >= config.strong_threshold and coverage >= config.min_essential_coverage_for_strong:
        rec = Recommendation.STRONG
    elif overall < config.limited_threshold or coverage < 40:
        rec = Recommendation.LIMITED
    else:
        rec = Recommendation.POTENTIAL

    matched = [m for m in matches if m.strength != EvidenceStrength.MISSING]
    demonstrated = sum(1 for m in matched if m.strength == EvidenceStrength.DEMONSTRATED)
    if candidate.parse_warnings or len(candidate.skills) < 3 or (matched and demonstrated / len(matched) < 0.3):
        confidence = "low"
    elif matched and demonstrated / len(matched) >= 0.6:
        confidence = "high"
    else:
        confidence = "medium"

    return MatchAssessment(
        candidate_id=candidate.candidate_id, display_name=candidate.display_name,
        overall_score=round(overall, 1), skill_score=round(skill_score, 1), essential_coverage=round(coverage, 1),
        relevant_years=round(years, 1),
        experience_score=round(experience_score, 1), qualification_score=round(qualification_score, 1),
        confidence=confidence, recommendation=rec, requirement_matches=matches,
    )


_GAP_WORDS = re.compile(r"\b(gap|lack|lacks|missing|no evidence|not|limited|without|absent|explore|unclear|weak|however)\b", re.I)


def mentions_any(text: str, skills: list[str]) -> bool:
    return any(re.search(rf"(?<![\w]){re.escape(s.split(' (or')[0])}(?![\w])", text, re.I) for s in skills)


def contradicts_evidence(text: str, missing: list[str]) -> bool:
    """True if a sentence presents a missing skill as a strength (no gap wording in that sentence)."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if mentions_any(sentence, missing) and not _GAP_WORDS.search(sentence):
            return True
    return False


def best_evidence(accepted: list[str], have: dict[str, SkillEvidence]) -> SkillEvidence | None:
    found = [have[s.lower()] for s in accepted if s.lower() in have]
    if not found:
        return None
    return min(found, key=lambda e: e.strength != EvidenceStrength.DEMONSTRATED)


def relevant_years(job: JobProfile, candidate: CandidateProfile) -> float:
    if not candidate.experience:
        return candidate.total_years_experience  # undated history: fall back to the stated figure
    wanted = {a for req in job.skills for a in req.accepted()}
    months = sum(e.months for e in candidate.experience
                 if wanted & set(find_skills(" ".join([e.title, *e.highlights]))))
    return round(min(months / 12, candidate.total_years_experience or months / 12), 1)


def qualification_fit(job: JobProfile, candidate: CandidateProfile) -> float:
    score = 100.0
    if job.education_requirements:
        required = max(DEGREE_RANK.get(d, 0) for d in job.education_requirements)
        have = max((DEGREE_RANK.get(d, 0) for d in find_degrees(" ".join(candidate.education))), default=0)
        score = 100.0 if have >= required else 60.0 if have == required - 1 else 30.0
    if job.certifications:
        cand = " ".join(candidate.certifications).lower()
        hit = sum(1 for c in job.certifications if c.lower() in cand)
        score = 0.8 * score + 20.0 * hit / len(job.certifications)
    return score


def rule_narrative(job: JobProfile, candidate: CandidateProfile, a: MatchAssessment) -> tuple[list[str], list[str], str]:
    strengths, gaps = [], []
    for m in sorted(a.requirement_matches, key=lambda m: -m.weight):
        if m.strength == EvidenceStrength.DEMONSTRATED and len(strengths) < 4:
            strengths.append(f"{m.skill}: applied in work/projects - \"{m.evidence[0][:120]}\"" if m.evidence else m.skill)
        elif m.strength == EvidenceStrength.MISSING and len(gaps) < 4:
            gaps.append(f"No evidence of {m.skill} ({m.importance.value})")
        elif m.credit < 1 and m.evidence and m.evidence[0].startswith("Related skill") and len(gaps) < 4:
            gaps.append(f"{m.skill} not shown; transferable experience via {m.evidence[0].split(':')[1].split(' - ')[0].strip()}")
        elif m.strength == EvidenceStrength.LISTED and m.importance == Importance.ESSENTIAL and len(gaps) < 4:
            gaps.append(f"{m.skill} is listed but not evidenced in work examples")
    if job.min_years_experience and a.relevant_years < job.min_years_experience:
        gaps.append(f"{a.relevant_years:g} yrs relevant experience vs {job.min_years_experience:g}+ requested")

    n_ess = sum(1 for m in a.requirement_matches if m.importance == Importance.ESSENTIAL)
    n_ess_ok = round(a.essential_coverage / 100 * n_ess)
    rationale = (
        f"Evidence covers {n_ess_ok} of {n_ess} essential requirements ({a.essential_coverage:.0f}%) with a weighted "
        f"skill score of {a.skill_score:.0f}/100. Relevant experience is {a.relevant_years:g} years"
        + (f" against {job.min_years_experience:g}+ requested." if job.min_years_experience else ".")
        + (f" Key strengths: {', '.join(s.split(':')[0] for s in strengths[:3])}." if strengths else "")
        + (f" Areas to explore at interview: {', '.join(g.replace('No evidence of ', '').split(' (')[0] for g in gaps[:3])}." if gaps else "")
        + f" Confidence in this assessment is {a.confidence}."
    )
    return strengths, gaps, rationale


def rank(assessments: list[MatchAssessment]) -> list[MatchAssessment]:
    ordered = sorted(assessments, key=lambda a: (-a.overall_score, -a.essential_coverage, a.display_name))
    for i, a in enumerate(ordered, start=1):
        a.rank = i
    return ordered
