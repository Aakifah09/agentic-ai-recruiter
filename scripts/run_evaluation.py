"""Run the evaluation suite and log the experiment to SQLite + JSON.

Usage:
    python scripts/run_evaluation.py                       # offline rules baseline
    python scripts/run_evaluation.py --provider ollama:llama3.2:latest --name llama32
    python scripts/run_evaluation.py --listed-credit 0.8 --name lenient-listing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recruiter.agents.matcher import ScoringConfig  # noqa: E402
from recruiter.evaluation import run_evaluation  # noqa: E402
from recruiter.orchestrator import WorkflowConfig  # noqa: E402
from recruiter.storage import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="rules", help="rules | ollama:<model> | openai:<model> | anthropic:<model>")
    ap.add_argument("--name", default=None, help="experiment name")
    ap.add_argument("--listed-credit", type=float, default=None, help="credit for skills only listed, not demonstrated")
    ap.add_argument("--skill-weight", type=float, default=None)
    ap.add_argument("--no-fairness", action="store_true", help="skip the counterfactual fairness test")
    ap.add_argument("--stages", default="jd,resume", help="comma list of LLM stages: jd,resume,narrative,interview")
    args = ap.parse_args()

    scoring = ScoringConfig()
    if args.listed_credit is not None:
        scoring.listed_credit = args.listed_credit
    if args.skill_weight is not None:
        scoring.skill_weight = args.skill_weight
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    config = WorkflowConfig(provider_spec=args.provider, scoring=scoring, llm_stages=stages, interview_top_k=0)

    result = run_evaluation(config, name=args.name or args.provider, fairness=not args.no_fairness,
                            progress=lambda msg, frac: print(f"[{frac:5.0%}] {msg}", flush=True))
    Store().save_experiment(result.experiment_id, result.name, result.provider, result.config, result.metrics)
    out_dir = ROOT / "data" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{result.name.replace(':', '_')}_{result.experiment_id}.json").write_text(result.to_json(), encoding="utf-8")

    print(f"\nExperiment {result.experiment_id} ({result.provider})")
    width = max(len(k) for k in result.metrics)
    for k, v in result.metrics.items():
        print(f"  {k:<{width}}  {v:.3f}")
    for job, info in result.per_job.items():
        top = ", ".join(f"{r['candidate'].split('_')[0]}({r['gold_grade']})" for r in info["ranking"][:4])
        print(f"  {job:<20} top-4 [gold grade]: {top}")


if __name__ == "__main__":
    main()
