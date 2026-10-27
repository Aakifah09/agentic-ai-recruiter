import json

import pandas as pd
import pytest

from recruiter.evaluation import (
    counterfactual_variants,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    prf,
    run_evaluation,
    spearman,
)
from recruiter.llm import LLMError, extract_json, make_provider, structured_call
from recruiter.orchestrator import RecruitmentWorkflow, ResumeInput, WorkflowConfig, rescore
from recruiter.prompts import LLMMatchNarrative
from recruiter.reporting import shortlist_frame, to_docx, to_json, to_markdown
from recruiter.schemas import HumanReview, Importance
from recruiter.storage import Store


@pytest.fixture(scope="module")
def report(jobs, resumes):
    inputs = [ResumeInput(fn, text) for fn, text in resumes.items()]
    return RecruitmentWorkflow(WorkflowConfig(interview_top_k=3)).run(jobs["ai_engineer.txt"], inputs)


# ----------------------------------------------------------------------------- orchestrator
def test_workflow_end_to_end(report):
    assert len(report.assessments) == 11
    assert [a.rank for a in report.assessments] == list(range(1, 12))
    assert report.candidates[0].display_name == "Candidate A"  # blind review default
    assert len(report.interview_plans) == 3
    agents = {s.agent.split(" [")[0] for s in report.trace}
    assert {"Job Description Analyst", "Resume Analyst", "Evidence Matcher", "Interview Planner",
            "Compliance Guardrail", "Orchestrator"} <= agents
    assert report.provider == "rules"


def test_workflow_validates_inputs():
    wf = RecruitmentWorkflow()
    with pytest.raises(ValueError):
        wf.run("", [ResumeInput("a.txt", "x" * 100)])
    with pytest.raises(ValueError):
        wf.run("Job", [])


def test_llm_stages_control_which_agents_use_the_provider(fake_provider, jobs, resumes):
    calls = []
    provider = fake_provider(lambda s, u: calls.append(s) or {"title": "AI Engineer"})
    wf = RecruitmentWorkflow(WorkflowConfig(llm_stages=("jd",), interview_top_k=1), provider=provider)
    wf.run(jobs["ai_engineer.txt"], [ResumeInput("r", resumes["r01_senior_ai_engineer.txt"])])
    assert len(calls) == 1 and "job descriptions" in calls[0]


def test_progress_callback_reaches_completion(jobs, resumes):
    seen = []
    RecruitmentWorkflow().run(jobs["qa_engineer.txt"], [ResumeInput("r", resumes["r05_qa_engineer.txt"])],
                              progress=lambda m, f: seen.append(f))
    assert seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)


def test_rescore_with_recruiter_edited_profile(report):
    job = report.job.model_copy(deep=True)
    for s in job.skills:
        if s.skill == "Streamlit":
            s.importance, s.weight = Importance.ESSENTIAL, 0.5
    new = rescore(report, job)
    assert new.run_id != report.run_id and len(new.assessments) == len(report.assessments)
    assert new.trace[0].agent == "Recruiter"


# ----------------------------------------------------------------------------- storage + reporting
def test_store_roundtrip_reviews_and_experiments(report, tmp_path):
    store = Store(tmp_path / "t.db")
    store.save_run(report, "job text")
    assert store.list_runs()[0]["run_id"] == report.run_id
    assert store.load_run(report.run_id).assessments[0].overall_score == report.assessments[0].overall_score
    cid = report.assessments[0].candidate_id
    store.save_review(report.run_id, HumanReview(candidate_id=cid, decision="hold", reviewer="sam"))
    store.save_review(report.run_id, HumanReview(candidate_id=cid, decision="advance", reviewer="sam", notes="great"))
    assert store.reviews(report.run_id)[cid].decision == "advance"
    assert [r["decision"] for r in store.audit_log(report.run_id)] == ["hold", "advance"]
    store.save_experiment("e1", "exp", "rules", {"a": 1}, {"ndcg@3": 0.9})
    assert store.experiments()[0]["metrics"]["ndcg@3"] == 0.9
    store.delete_run(report.run_id)
    assert store.load_run(report.run_id) is None


def test_exports(report):
    reviews = {report.assessments[0].candidate_id: HumanReview(candidate_id=report.assessments[0].candidate_id,
                                                               decision="advance", notes="Strong RAG work")}
    df = shortlist_frame(report, reviews)
    assert isinstance(df, pd.DataFrame) and df.iloc[0]["recruiter_decision"] == "Advance to interview"
    md = to_markdown(report, reviews)
    assert "# Recruitment shortlist" in md and "Strong RAG work" in md and "Interview plan" in md
    assert json.loads(to_json(report, reviews))["human_reviews"]
    assert to_docx(report, reviews)[:2] == b"PK"  # docx is a zip
    assert "@example.com" not in md + to_json(report)


# ----------------------------------------------------------------------------- llm utilities
@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1,} hope that helps'])
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(Exception):
        extract_json("no json here")


def test_structured_call_retries_with_validation_feedback(fake_provider):
    replies = iter([{"strengths": "not a list"}, {"strengths": ["x"], "gaps": [], "rationale": "ok"}])
    provider = fake_provider(lambda s, u: next(replies))
    out = structured_call(provider, "sys", "user", LLMMatchNarrative)
    assert out.rationale == "ok" and "failed validation" in provider.prompts[1][1]


def test_structured_call_gives_up(fake_provider):
    with pytest.raises(LLMError):
        structured_call(fake_provider(lambda s, u: {"strengths": 3}), "s", "u", LLMMatchNarrative)


def test_make_provider():
    assert make_provider("rules") is None
    assert make_provider("ollama:llama3.2:latest").model == "llama3.2:latest"
    with pytest.raises(ValueError):
        make_provider("unknown:x")


# ----------------------------------------------------------------------------- evaluation
def test_metric_primitives():
    assert prf({"a", "b"}, {"b", "c"}) == (0.5, 0.5, 0.5)
    assert ndcg_at_k([3, 2, 1, 0], 3) == 1.0 and ndcg_at_k([0, 1, 3], 3) < 1.0
    assert precision_at_k([3, 1, 2], 3) == pytest.approx(2 / 3)
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pairwise_accuracy([3, 2, 1], [2, 1, 0]) == 1.0


def test_counterfactuals_change_only_protected_markers(resumes):
    variants = counterfactual_variants(resumes["r05_qa_engineer.txt"])
    assert len(variants) == 3 and all("pytest framework" in v for v in variants)


def test_evaluation_quality_gate():
    """Regression gate: the rule-based baseline must stay above these thresholds."""
    m = run_evaluation(fairness=True).metrics
    assert m["jd_skill_f1"] >= 0.9
    assert m["resume_skill_f1"] >= 0.9
    assert m["ndcg@3"] >= 0.9
    assert m["pairwise_accuracy"] >= 0.85
    assert m["evidence_grounding_rate"] == 1.0
    assert m["pii_leak_rate"] == 0.0
    assert m["counterfactual_max_score_shift"] == 0.0
