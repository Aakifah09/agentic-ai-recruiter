"""Quantitative evaluation of the agent workflow against hand-labelled ground truth.

Metrics
-------
* JD skill extraction        precision / recall / F1, importance (essential vs desirable) accuracy
* Resume skill extraction    micro precision / recall / F1
* Ranking quality            NDCG@3, NDCG@5, Precision@3 (grade >= 2), Spearman rho, pairwise accuracy
* Fairness                   counterfactual score shift when names / gender / origin markers change
* Privacy                    PII leakage rate in evidence and generated text
* Grounding                  share of evidence snippets found verbatim in the source resume
* Bias detection             JD bias flags found vs expected
* Efficiency                 latency per candidate
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from recruiter.agents.jd_analyst import JobDescriptionAnalyst
from recruiter.agents.matcher import score_candidate
from recruiter.agents.resume_analyst import ResumeAnalyst
from recruiter.documents import extract_file
from recruiter.orchestrator import RecruitmentWorkflow, ResumeInput, WorkflowConfig
from recruiter.privacy import redact
from recruiter.schemas import Importance, ShortlistReport
from recruiter.textutils import is_grounded

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
GOLD = ROOT / "data" / "eval" / "gold.json"


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def dcg(grades: list[float]) -> float:
    return float(sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(grades)))


def ndcg_at_k(ranked_grades: list[float], k: int) -> float:
    ideal = dcg(sorted(ranked_grades, reverse=True)[:k])
    return dcg(ranked_grades[:k]) / ideal if ideal else 0.0


def precision_at_k(ranked_grades: list[float], k: int, relevant: int = 2) -> float:
    top = ranked_grades[:k]
    return sum(1 for g in top if g >= relevant) / len(top) if top else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        order = arr.argsort()
        r = np.empty_like(arr)
        r[order] = np.arange(len(arr))
        for val in np.unique(arr):  # average ranks for ties
            idx = arr == val
            r[idx] = r[idx].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(scores: list[float], grades: list[float]) -> float:
    """Share of candidate pairs with different gold grades that the scores order correctly."""
    ok = total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if grades[i] == grades[j]:
                continue
            total += 1
            ok += (scores[i] - scores[j]) * (grades[i] - grades[j]) > 0
    return ok / total if total else 0.0


# --------------------------------------------------------------------------- #
# Counterfactual fairness
# --------------------------------------------------------------------------- #
COUNTERFACTUALS = [
    ("Alex Johnson", "Gender: Male", "Nationality: Canadian", "he"),
    ("Fatima Al-Sayed", "Gender: Female", "Nationality: Egyptian", "she"),
    ("Wei Zhang", "Pronouns: they/them", "Nationality: Chinese", "they"),
]


def counterfactual_variants(text: str) -> list[str]:
    """Same resume with a different name, gender and origin markers and pronouns."""
    lines = text.splitlines()
    body = "\n".join(lines[1:])
    variants = []
    for name, gender, origin, pronoun in COUNTERFACTUALS:
        b = re.sub(r"\b(they|he|she)\b", pronoun, body)
        variants.append(f"{name}\n{gender}\n{origin}\nDate of Birth: 01/01/1975\n{b}")
    return variants


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    experiment_id: str
    name: str
    provider: str
    metrics: dict[str, float]
    per_job: dict[str, dict] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_samples() -> tuple[dict[str, str], dict[str, str], dict]:
    jobs = {p.name: extract_file(p) for p in sorted((SAMPLES / "jobs").glob("*.txt"))}
    resumes = {p.name: extract_file(p) for p in sorted((SAMPLES / "resumes").glob("*.txt"))}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    return jobs, resumes, gold


def run_evaluation(config: WorkflowConfig | None = None, name: str = "baseline", progress=None,
                   fairness: bool = True) -> EvalResult:
    config = config or WorkflowConfig()
    jobs, resumes, gold = load_samples()
    workflow = RecruitmentWorkflow(config)
    notify = progress or (lambda msg, frac: None)

    jd_p, jd_r, jd_f, imp_acc, bias_hits = [], [], [], [], []
    ndcg3, ndcg5, p_at_3, rhos, pairs = [], [], [], [], []
    per_job: dict[str, dict] = {}
    latencies = []
    reports: list[ShortlistReport] = []

    for j_idx, (job_name, job_text) in enumerate(jobs.items()):
        g = gold["jobs"][job_name]
        notify(f"Evaluating {job_name}", j_idx / (len(jobs) + 1))
        inputs = [ResumeInput(fn, text, candidate_id=fn) for fn, text in resumes.items()]
        t0 = time.perf_counter()
        report = workflow.run(job_text, inputs)
        latencies.append((time.perf_counter() - t0) / len(inputs))
        reports.append(report)

        # JD extraction
        predicted = {a for s in report.job.skills for a in s.accepted()}
        gold_all = set(g["essential"]) | set(g["desirable"])
        p, r, f = prf(predicted, gold_all)
        jd_p.append(p), jd_r.append(r), jd_f.append(f)
        pred_imp = {a: s.importance for s in report.job.skills for a in s.accepted()}
        labelled = [(sk, Importance.ESSENTIAL) for sk in g["essential"]] + [(sk, Importance.DESIRABLE) for sk in g["desirable"]]
        found = [(sk, imp) for sk, imp in labelled if sk in pred_imp]
        imp_acc.append(sum(pred_imp[sk] == imp for sk, imp in found) / len(found) if found else 0.0)
        if "expected_bias_flags" in g:
            bias_hits.append(min(1.0, len(report.job.bias_flags) / g["expected_bias_flags"]))

        # Ranking
        ranked = [a.candidate_id for a in report.assessments]
        ranked_grades = [g["grades"][cid] for cid in ranked]
        scores = [a.overall_score for a in report.assessments]
        ndcg3.append(ndcg_at_k(ranked_grades, 3))
        ndcg5.append(ndcg_at_k(ranked_grades, 5))
        p_at_3.append(precision_at_k(ranked_grades, 3))
        rhos.append(spearman(scores, ranked_grades))
        pairs.append(pairwise_accuracy(scores, ranked_grades))
        per_job[job_name] = {
            "title": report.job.title,
            "ranking": [{"candidate": cid, "score": s, "gold_grade": gr} for cid, s, gr in zip(ranked, scores, ranked_grades)],
            "jd_f1": round(f, 3),
            "ndcg@3": round(ndcg3[-1], 3),
        }

    # Resume skill extraction (job-independent, use first report's profiles)
    rp, rr, rf = [], [], []
    for c in reports[0].candidates:
        p, r, f = prf({s.skill for s in c.skills}, set(gold["resumes"][c.candidate_id]))
        rp.append(p), rr.append(r), rf.append(f)

    # Grounding + privacy over all reports
    grounded = total_ev = leaks = texts_checked = 0
    for report, (job_name, _) in zip(reports, jobs.items()):
        for a in report.assessments:
            src = redact(resumes[a.candidate_id]).text
            for m in a.requirement_matches:
                for e in m.evidence:
                    if e.startswith("Related skill"):
                        continue
                    total_ev += 1
                    grounded += is_grounded(e.rstrip("."), src, 0.75)
        blob = report.model_dump_json()
        for fn, raw in resumes.items():
            for email in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw):
                texts_checked += 1
                leaks += email in blob

    # Counterfactual fairness: score shift when only protected markers change.
    max_shift = mean_shift = 0.0
    if fairness:
        notify("Counterfactual fairness test", len(jobs) / (len(jobs) + 1))
        jd_agent, resume_agent = JobDescriptionAnalyst(), ResumeAnalyst()
        shifts = []
        for job_name, job_text in jobs.items():
            job = jd_agent.run(job_text)
            for fn, text in resumes.items():
                base = score_candidate(job, resume_agent.run(text, fn), config.scoring).overall_score
                for variant in counterfactual_variants(text):
                    alt = score_candidate(job, resume_agent.run(variant, fn), config.scoring).overall_score
                    shifts.append(abs(alt - base))
        max_shift, mean_shift = float(max(shifts)), float(np.mean(shifts))

    metrics = {
        "jd_skill_precision": np.mean(jd_p), "jd_skill_recall": np.mean(jd_r), "jd_skill_f1": np.mean(jd_f),
        "jd_importance_accuracy": np.mean(imp_acc),
        "resume_skill_precision": np.mean(rp), "resume_skill_recall": np.mean(rr), "resume_skill_f1": np.mean(rf),
        "ndcg@3": np.mean(ndcg3), "ndcg@5": np.mean(ndcg5), "precision@3": np.mean(p_at_3),
        "spearman_rho": np.mean(rhos), "pairwise_accuracy": np.mean(pairs),
        "evidence_grounding_rate": grounded / total_ev if total_ev else 1.0,
        "pii_leak_rate": leaks / texts_checked if texts_checked else 0.0,
        "bias_flag_recall": np.mean(bias_hits) if bias_hits else 1.0,
        "latency_s_per_candidate": float(np.mean(latencies)),
    }
    if fairness:
        metrics["counterfactual_max_score_shift"] = max_shift
        metrics["counterfactual_mean_score_shift"] = mean_shift
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    notify("Done", 1.0)
    return EvalResult(
        experiment_id=uuid.uuid4().hex[:10], name=name, provider=workflow.provider_label, metrics=metrics,
        per_job=per_job, config={"scoring": config.scoring.model_dump(), "provider": config.provider_spec,
                                 "llm_stages": list(config.llm_stages)},
    )
