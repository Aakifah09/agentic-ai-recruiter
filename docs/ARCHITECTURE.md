# Architecture

## Goals and constraints

| Constraint | Consequence |
|---|---|
| Decision support, not decision making | No "reject" output exists in the schema; a recruiter records every decision with an audit trail |
| Must run free on a Windows laptop, CPU only | Every agent has a deterministic path; LLM use is optional and per agent |
| Recruiters must be able to trust the output | Scores come from an evidence table, not from the LLM; all quotes are checked against the resume |
| Fairness and privacy | Redaction before analysis, blind aliases, a compliance agent that runs last and can remove content |

## Workflow

```
                ┌──────────────────────┐
 Job text ────► │ 1  JD Analyst        │──► JobProfile (essential/desirable skills, weights, alternatives, bias flags)
                └──────────────────────┘                  │
                ┌──────────────────────┐                  ▼
 Resumes ─► Redaction ─► │ 2  Resume Analyst │ ──► CandidateProfile (skills + verbatim evidence + strength, dated roles)
  (PDF/DOCX/TXT)         └──────────────────┘                 │
                ┌──────────────────────┐                      ▼
                │ 3  Evidence Matcher  │──► MatchAssessment (score breakdown, coverage, confidence, rationale) ──► Ranker
                └──────────────────────┘                                                                         │
                ┌──────────────────────┐                                                                         ▼
                │ 4  Interview Planner │──► InterviewPlan for top-k ───────────────────────────────────────────────┐
                └──────────────────────┘                                                                         │
                ┌──────────────────────┐                                                                         ▼
                │ 5  Compliance Guard  │──► removes ungrounded evidence, protected-topic content, flags JD wording
                └──────────────────────┘
                         │
                         ▼
               ShortlistReport ──► SQLite ──► Streamlit review UI ──► Recruiter decision + notes (audit log)
                                                    │
                                                    └─► edit job profile / scoring policy ──► rescore() (no LLM, instant)
```

The orchestrator (`recruiter/orchestrator.py`) is an explicit state machine: a fixed sequence with fan-out over
candidates. There is no free-running agent loop or framework dependency. For a screening task with a known shape this is
more predictable, cheaper on a CPU, and much easier to test. Each stage is a plain class with a `run()` method and a
Pydantic return type.

## Agent contract (`recruiter/agents/base.py`)

```
run(input) -> validated model
  ├─ provider configured for this stage?  → _run_llm()  (structured_call: JSON-schema constrained, 1 repair retry)
  │        └─ any exception                → _run_rules() and trace status = "fallback"
  └─ otherwise                             → _run_rules()
```

Every call appends an `AgentStep` (agent, mode, status, duration, detail) to the run trace shown in the UI.

## Hybrid LLM + rules design

| Agent | What the LLM adds | What stays deterministic, and why |
|---|---|---|
| JD Analyst | Skills outside the taxonomy, priority judgements | Essential vs desirable comes from the document's own sections. In testing, llama3.2 marked "nice to have" skills as required. |
| Resume Analyst | Skills described in prose, "used in work" judgement | Each quote must pass `is_grounded()` against the redacted resume or it is discarded |
| Evidence Matcher | Optional narrative | The score. `contradicts_evidence()` rejects narratives that call a missing skill a strength |
| Interview Planner | Tailored questions | Template fallback built from evidence and gaps |
| Compliance Guardrail | None | Must be auditable |

## Scoring (`recruiter/agents/matcher.py`)

```
credit(requirement) = 1.0  demonstrated in a role or project      (any accepted alternative counts)
                      0.5  only listed in a skills section          (configurable)
                      0.3  only a related skill (e.g. Azure for AWS) (configurable)
                      0    missing
skill_score         = Σ weight·credit / Σ weight
experience_score    = min(1, relevant_years / min_years)   relevant = dated roles whose text uses a job skill
qualification_score = degree level vs requirement (+ certifications)
overall             = 0.65·skill + 0.20·experience + 0.15·qualification   (weights adjustable in the UI)
```

The labels are advisory: *Strong match – recommend interview*, *Potential match – recruiter review* and *Limited
evidence – human review required*. Confidence is lowered when evidence is mostly listed-only or parsing produced
warnings.

## Persistence (`recruiter/storage.py`)

SQLite tables: `jobs`, `candidates`, `runs` (full report JSON), `reviews` (current decision), `review_audit`
(append-only history) and `experiments` (evaluation metrics + config).

## LLM providers (`recruiter/llm.py`)

`OllamaProvider` (default, local), `OpenAIProvider`, `AnthropicProvider`, or no provider (rules). Ollama calls pass
the Pydantic JSON schema as `format` so decoding is constrained, with `num_ctx=4096` and `num_predict=1200` to keep
CPU latency down. `detect_environment()` lists the installed Ollama models and the API keys present, and the UI only
offers what is available.

## CPU performance notes

Measured on this machine (Windows 11, CPU only, llama3.2 3B Q4) while other workloads were running:
one structured call takes roughly 60–180 s. A full LLM run over N resumes makes about 1 + 2N + k calls, so the
default enables the LLM only for job analysis and interview questions (1 + k calls). The rules path takes about
0.05–0.2 s per candidate.
