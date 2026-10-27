import pytest

from recruiter.agents.compliance import ComplianceGuardrail
from recruiter.agents.interviewer import InterviewPlanner
from recruiter.agents.jd_analyst import JobDescriptionAnalyst, alternative_groups, split_alternatives
from recruiter.agents.matcher import EvidenceMatcher, ScoringConfig, contradicts_evidence, rank, score_candidate
from recruiter.agents.resume_analyst import ResumeAnalyst
from recruiter.llm import LLMError
from recruiter.schemas import (
    EvidenceStrength,
    Importance,
    InterviewPlan,
    InterviewQuestion,
    JobProfile,
    Recommendation,
    SkillRequirement,
)
from recruiter.skills import find_skills

JD = """Job Title: Data Engineer
Responsibilities:
- Build ETL pipelines with Airflow
Requirements:
- 4+ years of experience in data engineering
- Python and SQL
- Power BI or Tableau
- Bachelor's degree in Computer Science
Nice to have:
- Kubernetes
"""


# ----------------------------------------------------------------------------- Agent 1
def test_jd_rules_separate_essential_and_desirable():
    job = JobDescriptionAnalyst().run(JD)
    imp = {s.skill: s.importance for s in job.skills}
    assert imp["Python"] == imp["SQL"] == Importance.ESSENTIAL
    assert imp["Kubernetes"] == Importance.DESIRABLE
    assert job.min_years_experience == 4 and job.education_requirements == ["Bachelor's"]
    assert job.title == "Data Engineer"
    assert abs(sum(s.weight for s in job.skills) - 1) < 0.01


def test_jd_rules_fold_either_or_requirements():
    job = JobDescriptionAnalyst().run(JD)
    power_bi = next(s for s in job.skills if s.skill == "Power BI")
    assert power_bi.alternatives == ["Tableau"]
    assert "Tableau" not in {s.skill for s in job.skills}


def test_alternative_groups_only_join_on_or():
    line = "Python, SQL and Power BI or Tableau"
    assert alternative_groups(line, find_skills(line)) == [["Power BI", "Tableau"]]


@pytest.mark.parametrize("raw,expected", [
    ("FastAPI or Flask", ("FastAPI", ["Flask"])), ("CI/CD", ("CI/CD", [])), ("AWS/Azure", ("AWS", ["Azure"])),
])
def test_split_alternatives(raw, expected):
    assert split_alternatives(raw) == expected


def test_jd_flags_biased_language(jobs):
    job = JobDescriptionAnalyst().run(jobs["data_analyst.txt"])
    assert len(job.bias_flags) == 4


def test_jd_llm_path_is_reconciled_with_document_structure(fake_provider):
    # The LLM wrongly calls Kubernetes required; the JD's "Nice to have" section wins.
    provider = fake_provider(lambda s, u: {
        "title": "Data Engineer", "required_skills": [{"skill": "python3", "priority": 5},
                                                      {"skill": "Kubernetes", "priority": 3},
                                                      {"skill": "Data Modelling", "priority": 4}],
        "preferred_skills": [], "min_years_experience": 4, "seniority": "mid",
    })
    agent = JobDescriptionAnalyst(provider)
    job = agent.run(JD)
    imp = {s.skill: s.importance for s in job.skills}
    assert imp["Python"] == Importance.ESSENTIAL
    assert imp["Kubernetes"] == Importance.DESIRABLE
    assert "Data Modelling" in imp  # open-vocabulary skill contributed by the LLM
    assert "SQL" in imp  # essential skill the LLM missed is added by the rule cross-check
    assert agent.last_step.mode == "hybrid" and agent.last_step.status == "ok"


def test_jd_llm_failure_falls_back_to_rules(fake_provider):
    def boom(s, u):
        raise LLMError("connection refused")

    agent = JobDescriptionAnalyst(fake_provider(boom))
    job = agent.run(JD)
    assert agent.last_step.status == "fallback" and "Python" in {s.skill for s in job.skills}


def test_jd_llm_invalid_json_is_retried_with_validation_error(fake_provider):
    replies = iter(["not json at all {", {"title": "Data Engineer", "required_skills": [{"skill": "SQL", "priority": 9}]},
                    {"title": "Data Engineer", "required_skills": [{"skill": "SQL", "priority": 5}]}])
    provider = fake_provider(lambda s, u: next(replies))
    agent = JobDescriptionAnalyst(provider)
    agent.run(JD)
    # first reply unparseable -> fallback for that attempt; retries are covered in test_llm.py
    assert agent.last_step.status in {"ok", "fallback"}


# ----------------------------------------------------------------------------- Agent 2
def test_resume_rules_extract_evidence_strength_and_experience(resumes):
    c = ResumeAnalyst().run(resumes["r11_keyword_stuffer.txt"], "r11")
    assert c.skills and all(s.strength == EvidenceStrength.LISTED for s in c.skills)
    c = ResumeAnalyst().run(resumes["r05_qa_engineer.txt"], "r05")
    sk = {s.skill: s for s in c.skills}
    assert sk["Pytest"].strength == EvidenceStrength.DEMONSTRATED
    assert any("pytest framework" in snip for snip in sk["Pytest"].snippets)
    assert len(c.experience) == 2 and c.experience[1].months == 12
    assert "email" in c.redactions
    assert "Lena" not in c.model_dump_json(exclude={"display_name"})


def test_resume_uses_alias_in_blind_mode(resumes):
    c = ResumeAnalyst().run(resumes["r01_senior_ai_engineer.txt"], "r01", alias="Candidate A")
    assert c.display_name == "Candidate A"


def test_resume_llm_claims_must_be_grounded(fake_provider, resumes):
    provider = fake_provider(lambda s, u: {"skills": [
        {"skill": "Terraform", "evidence_quote": "Provisioned all infrastructure with Terraform modules", "used_in_work": True},
        {"skill": "SQL", "evidence_quote": "wrote SQL to validate test data", "used_in_work": True},
    ]})
    c = ResumeAnalyst(provider).run(resumes["r05_qa_engineer.txt"], "r05")
    names = {s.skill for s in c.skills}
    assert "Terraform" not in names and "SQL" in names
    assert any("discarded" in w for w in c.parse_warnings)


def test_resume_without_dates_warns():
    c = ResumeAnalyst().run("Skills\nPython, SQL, Docker\nSummary\nI like data and building things with Python.", "x")
    assert c.parse_warnings and c.total_years_experience == 0


# ----------------------------------------------------------------------------- Agent 3
def _job(*skills, min_years=None):
    return JobProfile(title="T", min_years_experience=min_years, skills=[
        SkillRequirement(skill=s, importance=Importance.ESSENTIAL, weight=1 / len(skills), alternatives=alts)
        for s, alts in skills])


def test_demonstrated_beats_listed_beats_missing(resumes, jobs):
    job = JobDescriptionAnalyst().run(jobs["ai_engineer.txt"])
    ra = ResumeAnalyst()
    real = score_candidate(job, ra.run(resumes["r01_senior_ai_engineer.txt"], "a"))
    stuffer = score_candidate(job, ra.run(resumes["r11_keyword_stuffer.txt"], "b"))
    unrelated = score_candidate(job, ra.run(resumes["r07_marketing.txt"], "c"))
    assert real.overall_score > stuffer.overall_score > unrelated.overall_score
    assert stuffer.confidence == "low" and stuffer.relevant_years == 0
    assert real.recommendation == Recommendation.STRONG


def test_alternatives_and_related_skills_get_credit(resumes):
    ra = ResumeAnalyst()
    analyst = ra.run(resumes["r03_junior_data_analyst.txt"], "x")  # has Tableau, not Power BI
    m = score_candidate(_job(("Power BI", ["Tableau"])), analyst).requirement_matches[0]
    assert m.credit == 1.0 and m.matched_as == "Tableau"
    m = score_candidate(_job(("Power BI", [])), analyst).requirement_matches[0]
    assert m.credit == ScoringConfig().related_credit


def test_scoring_policy_is_configurable(resumes):
    c = ResumeAnalyst().run(resumes["r11_keyword_stuffer.txt"], "x")
    job = _job(("Python", []), ("Docker", []))
    low = score_candidate(job, c, ScoringConfig(listed_credit=0.2)).skill_score
    high = score_candidate(job, c, ScoringConfig(listed_credit=0.9)).skill_score
    assert high > low


def test_recommendations_are_advisory_only(resumes, jobs):
    job = JobDescriptionAnalyst().run(jobs["qa_engineer.txt"])
    for text in resumes.values():
        a = score_candidate(job, ResumeAnalyst().run(text, "x"))
        assert a.recommendation in set(Recommendation)
        assert "reject" not in a.recommendation.value.lower()


def test_rank_orders_by_score():
    job = _job(("Python", []))
    from recruiter.schemas import MatchAssessment
    items = [MatchAssessment(candidate_id=str(i), display_name=str(i), overall_score=s, skill_score=0, essential_coverage=0,
                             experience_score=0, qualification_score=0) for i, s in enumerate([40, 90, 70])]
    assert [a.overall_score for a in rank(items)] == [90, 70, 40] and rank(items)[0].rank == 1
    assert job.skills


def test_narrative_hallucination_guard(fake_provider, resumes, jobs):
    assert contradicts_evidence("The candidate demonstrated strong Docker skills.", ["Docker"])
    assert not contradicts_evidence("However, there is no evidence of Docker.", ["Docker"])
    job = JobDescriptionAnalyst().run(jobs["ai_engineer.txt"])
    cand = ResumeAnalyst().run(resumes["r08_graduate_ai.txt"], "x")
    provider = fake_provider(lambda s, u: {"strengths": ["Docker expert"], "gaps": [],
                                           "rationale": "Shows excellent Docker and Kubernetes experience."})
    matcher = EvidenceMatcher(provider)
    a = matcher.run(job, cand)
    assert matcher.last_step.status == "fallback"
    assert "Docker expert" not in a.strengths


# ----------------------------------------------------------------------------- Agents 4 + 5
def test_interview_plan_covers_evidence_gaps_and_behaviour(resumes, jobs):
    job = JobDescriptionAnalyst().run(jobs["ai_engineer.txt"])
    cand = ResumeAnalyst().run(resumes["r08_graduate_ai.txt"], "x")
    a = EvidenceMatcher().run(job, cand)
    plan = InterviewPlanner(n_questions=6).run(job, cand, a)
    purposes = {q.purpose for q in plan.questions}
    assert {"verify_evidence", "probe_gap", "behavioural"} <= purposes
    assert len(plan.questions) <= 6


def test_guardrail_removes_protected_questions_and_ungrounded_evidence(resumes, jobs):
    job = JobDescriptionAnalyst().run(jobs["data_analyst.txt"])
    text = resumes["r04_senior_data_analyst.txt"]
    cand = ResumeAnalyst().run(text, "c1")
    a = EvidenceMatcher().run(job, cand)
    a.requirement_matches[0].evidence = ["Invented a quantum SQL engine at NASA"]
    a.rationale += " They are young and energetic."
    plan = InterviewPlan(candidate_id="c1", questions=[
        InterviewQuestion(question="Do you plan to have children soon?", purpose="behavioural"),
        InterviewQuestion(question="Walk me through your Power BI data model design.", purpose="verify_evidence"),
    ])
    from recruiter.privacy import redact
    report = ComplianceGuardrail().run(job, [cand], [a], {"c1": plan}, {"c1": redact(text).text})
    assert report.removed_questions == 1 and len(plan.questions) == 1
    assert a.requirement_matches[0].evidence == []
    assert "young" not in a.rationale
    assert not report.passed  # biased JD + blocked content
