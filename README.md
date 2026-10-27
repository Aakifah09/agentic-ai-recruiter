# Agentic AI Recruiter

A multi-agent recruitment assistant that compares resumes against a job description, extracts evidence, scores and
explains each match, drafts interview questions and builds a shortlist for a **human recruiter to review**.
It runs entirely on a CPU-only Windows laptop with a local Ollama model, or with no LLM at all.

> Decision support only. The system never rejects a candidate and is designed not to infer protected
> characteristics. All sample data is fictional.

## What it demonstrates

| Area | Where |
|---|---|
| Agentic workflow: five specialised agents, explicit orchestration, trace, retries and fallbacks | [recruiter/orchestrator.py](recruiter/orchestrator.py), [recruiter/agents/](recruiter/agents/) |
| LLM integration behind a provider interface (Ollama / OpenAI / Anthropic / none) | [recruiter/llm.py](recruiter/llm.py) |
| Prompt engineering with compact response schemas and guardrail preambles, versioned | [recruiter/prompts.py](recruiter/prompts.py) |
| Structured outputs: Pydantic everywhere, JSON-schema-constrained decoding, validation-error repair loop | [recruiter/schemas.py](recruiter/schemas.py) |
| Evidence grounding and hallucination guards | `textutils.is_grounded`, `matcher.contradicts_evidence` |
| Responsible AI: redaction, blind review, bias flags, protected-topic filter, audit log | [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) |
| Quantitative evaluation: extraction F1, NDCG, Spearman, counterfactual fairness, PII leakage | [docs/EVALUATION.md](docs/EVALUATION.md) |
| Human in the loop: edit requirements and weights, instant re-score, decisions with audit trail | Shortlist & review page |
| Persistence in SQLite; Markdown / Word / CSV / JSON exports | [recruiter/storage.py](recruiter/storage.py), [recruiter/reporting.py](recruiter/reporting.py) |
| Testing: 67 pytest tests including headless Streamlit UI tests | [tests/](tests/) |

## The agents

1. **Job Description Analyst:** role, responsibilities, essential vs desirable skills, "X or Y" alternatives,
   normalised importance weights, minimum experience, education, and flags for exclusionary wording.
2. **Resume Analyst:** redacts personal data, then extracts skills with verbatim evidence. Each skill is labelled
   *demonstrated* (used in a role or project) or *listed* (skills section only). It also extracts the dated work
   history and total experience.
3. **Evidence Matcher:** deterministic weighted score (skills, relevant experience, qualifications), essential
   coverage, confidence, strengths, gaps and rationale.
4. **Interview Planner:** questions that verify claimed evidence, probe gaps and assess behaviours (STAR), each with
   "what a good answer looks like".
5. **Compliance Guardrail:** removes ungrounded evidence and any content touching protected characteristics, and
   reports JD wording issues.

Each agent can use the LLM or deterministic rules, chosen per agent in the sidebar. Any LLM error, timeout or invalid
JSON is retried once with the validation error and then falls back to rules, so a run always completes.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (Windows 11, PowerShell)

```powershell
cd C:\Users\aakif\agentic-ai-recruiter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

streamlit run streamlit_app.py          # opens http://localhost:8501
```

The packages are already installed in this machine's Python 3.11, so the venv step is optional here:
`python -m streamlit run streamlit_app.py`.

**Optional local LLM:** install [Ollama](https://ollama.com), then run `ollama pull llama3.2`. The app detects
running models automatically. On a CPU, expect about 1–2 minutes per LLM call with a 3B model. The default therefore
uses the LLM only for job analysis and interview questions.

**Optional external APIs:** `pip install openai` or `pip install anthropic`, then set `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`; the provider appears in the sidebar. No key is ever required.

## Using the app

1. **New analysis:** paste, upload (PDF, DOCX or TXT) or pick a sample job description. Upload resumes and/or
   include the 11 sample resumes, then run the workflow. The agent trace shows each step, its mode (rules or hybrid),
   and any fallback.
2. **Shortlist & review:**
   - *Shortlist:* ranked table and score-component chart.
   - *Candidate review:* evidence by requirement, strengths and gaps, the interview plan, the extracted profile, and
     a form to record a decision (advance, hold, needs info) with notes and an audit trail.
   - *Job profile:* edit skills, importance, weights and alternatives, then **re-score instantly**.
   - *Compliance:* issues found, and the safeguards applied.
   - *Agent trace* and *Export:* Markdown, Word, CSV and JSON.
3. **Evaluation:** run the benchmark against the gold labels and compare experiments.

`data/samples/uploads/` (created by `python scripts/make_sample_docs.py`) holds PDF and DOCX versions of some
sample resumes for trying the upload path.

## Command line

```powershell
# Headless run -> reports\shortlist_<run>.md / .docx / .csv
python scripts\run_pipeline.py --job data\samples\jobs\ai_engineer.txt --resumes data\samples\resumes
python scripts\run_pipeline.py --job my_jd.pdf --resumes C:\cvs --provider ollama:llama3.2:latest --stages jd,interview

# Evaluation -> SQLite experiments table + data\eval\results\*.json
python scripts\run_evaluation.py
python scripts\run_evaluation.py --provider ollama:llama3.2:latest --stages jd --no-fairness --name llama-jd

# Tests
python -m pytest
```

## Results at a glance

Rule-based baseline on the 3 × 11 sample set (details and caveats in [docs/EVALUATION.md](docs/EVALUATION.md)):

| JD skill F1 | Resume skill F1 | NDCG@3 | Pairwise acc. | Grounding | PII leaks | Counterfactual shift |
|---|---|---|---|---|---|---|
| 0.98 | 0.98 | 0.99 | 0.98 | 100% | 0% | 0.0 |

The labels were written by the developer, so read these as regression baselines, not as real-world accuracy.

## Project layout

```
streamlit_app.py            entry point: settings sidebar + navigation
app_pages/                  analyze, shortlist (review/HITL/export), evaluation, about
app_utils.py                cached store/environment, settings -> WorkflowConfig
recruiter/
  agents/                   jd_analyst, resume_analyst, matcher, interviewer, compliance, base
  orchestrator.py           workflow, progress, trace, rescore()
  llm.py  prompts.py        providers, structured calls, environment detection, prompt templates
  schemas.py                Pydantic models for every hand-off
  documents.py privacy.py   PDF/DOCX/TXT extraction, redaction and bias lexicons
  skills.py textutils.py    taxonomy, sections, date ranges, grounding
  storage.py reporting.py   SQLite, exports
  evaluation.py             metrics + benchmark runner
scripts/                    run_pipeline, run_evaluation, make_sample_docs
data/samples, data/eval     fictional jobs/resumes, gold labels, results
tests/                      67 tests (unit, agent, workflow, LLM-mocked, Streamlit AppTest)
docs/                       ARCHITECTURE, EVALUATION, RESPONSIBLE_AI
```

## Limitations

Pattern-based redaction, an English and tech-centric skill taxonomy, a small self-labelled evaluation set, and slow
LLM inference on CPU. See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md#known-limitations) and
[docs/EVALUATION.md](docs/EVALUATION.md#threats-to-validity). Using a tool like this on real applicants would require
legal review (for example under the EU AI Act, where recruitment AI is high-risk, or NYC Local Law 144), bias audits on
real data, and notices to candidates.
