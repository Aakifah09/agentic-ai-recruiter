"""Headless CLI: analyse a job description against a folder of resumes and write reports.

Usage:
    python scripts/run_pipeline.py --job data/samples/jobs/ai_engineer.txt --resumes data/samples/resumes
    python scripts/run_pipeline.py --job jd.pdf --resumes ./cvs --provider ollama:llama3.2:latest --stages jd,interview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recruiter.documents import SUPPORTED_EXTENSIONS, DocumentError, extract_file  # noqa: E402
from recruiter.orchestrator import RecruitmentWorkflow, ResumeInput, WorkflowConfig  # noqa: E402
from recruiter.reporting import shortlist_frame, to_docx, to_markdown  # noqa: E402
from recruiter.storage import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--resumes", required=True, type=Path, help="folder of PDF/DOCX/TXT resumes")
    ap.add_argument("--provider", default="rules")
    ap.add_argument("--stages", default="jd,interview")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--show-names", action="store_true", help="disable blind review aliases")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    resumes = []
    for p in sorted(args.resumes.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                resumes.append(ResumeInput(p.name, extract_file(p)))
            except DocumentError as exc:
                print(f"skip {p.name}: {exc}")
    job_text = extract_file(args.job)

    config = WorkflowConfig(provider_spec=args.provider, llm_stages=tuple(args.stages.split(",")),
                            interview_top_k=args.top_k, blind_review=not args.show_names)
    report = RecruitmentWorkflow(config).run(job_text, resumes, progress=lambda m, f: print(f"[{f:5.0%}] {m}", flush=True))
    Store().save_run(report, job_text)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"shortlist_{report.run_id}"
    stem.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    stem.with_suffix(".docx").write_bytes(to_docx(report))
    df = shortlist_frame(report)
    df.to_csv(stem.with_suffix(".csv"), index=False)

    print(f"\n{report.job.title} - {len(resumes)} candidates (run {report.run_id})")
    print(df[["rank", "candidate", "source_file", "overall_score", "essential_coverage", "confidence", "ai_recommendation"]]
          .to_string(index=False))
    print(f"\nReports written to {stem}.md/.docx/.csv")


if __name__ == "__main__":
    main()
